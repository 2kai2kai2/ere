use crate::working_nfa::{EpsilonType, WorkingNFA};
use crate::{parse_tree::Atom, working_nfa::EpsilonTransition};
use quote::{quote, ToTokens};
use std::{fmt::Write, ops::RangeInclusive};
use std::{usize, vec};

/// Sorts and combines ranges
///
/// ```
/// assert_eq!(
///     reduce_ranges(vec![4..=6, 2..=3, 5..=7, 9..=10]),
///     vec![2..=7, 9..=10],
/// );
/// ```
fn reduce_ranges<T: Ord + Clone + Copy>(
    mut ranges: Vec<RangeInclusive<T>>,
) -> Vec<RangeInclusive<T>>
where
    RangeInclusive<T>: ExactSizeIterator,
{
    ranges.sort_by_key(|range| *range.start());
    let Some((first_range, terms)) = ranges.split_first_chunk::<1>() else {
        return Vec::new();
    };
    let mut reduced_terms = Vec::new();

    let mut current_start = first_range[0].start();
    let mut current_end = first_range[0].end();
    for term in terms {
        if term.is_empty() {
            continue;
        } else if (*current_end..=term.end().clone()).len() <= 2 {
            // the next term either starts immediately after or is overlapping
            // so combine them.
            current_end = std::cmp::max(current_end, term.end());
        } else {
            reduced_terms.push(current_start.clone()..=current_end.clone());
            current_start = term.start();
            current_end = term.end();
        }
    }
    reduced_terms.push(current_start.clone()..=current_end.clone());
    return reduced_terms;
}
fn char_range_to_u8_ranges(range: RangeInclusive<char>) -> Vec<Vec<RangeInclusive<u8>>> {
    let start_char = *range.start();
    let end_char = *range.end();

    let mut out = Vec::new();
    if end_char < start_char {
        return out; // if empty
    }

    // one-byte part
    const ONE_BYTE_END: char = '\u{007F}';
    if start_char <= ONE_BYTE_END {
        let mut start_bytes = [0; 4];
        start_char.encode_utf8(&mut start_bytes);

        let end_char = std::cmp::min(end_char, ONE_BYTE_END);
        let mut end_bytes = [0; 4];
        end_char.encode_utf8(&mut end_bytes);

        out.push(vec![start_bytes[0]..=end_bytes[0]]);
    }

    // two-byte part
    const TWO_BYTE_START: char = '\u{0080}';
    const TWO_BYTE_END: char = '\u{07FF}';
    if start_char <= TWO_BYTE_END && end_char >= TWO_BYTE_START {
        let start_char = std::cmp::max(start_char, TWO_BYTE_START);
        let mut start_bytes = [0; 4];
        start_char.encode_utf8(&mut start_bytes);

        let end_char = std::cmp::min(end_char, TWO_BYTE_END);
        let mut end_bytes = [0; 4];
        end_char.encode_utf8(&mut end_bytes);

        if start_bytes[0] == end_bytes[0] {
            out.push(vec![0b110_00000..=0b110_11111, 0b10_000000..=0b10_111111]);
        } else {
            let start = if start_bytes[1] == 0b10_000000 {
                start_bytes[0]
            } else {
                // add partial-second-byte range at start
                out.push(vec![
                    start_bytes[0]..=start_bytes[0],
                    start_bytes[1]..=0b10_111111,
                ]);
                start_bytes[0] + 1
            };
            let end = if end_bytes[1] == 0b10_111111 {
                end_bytes[0]
            } else {
                // add partial-second-byte range at end
                out.push(vec![
                    end_bytes[0]..=end_bytes[0],
                    0b10_000000..=end_bytes[1],
                ]);
                end_bytes[0] - 1
            };

            if start <= end {
                out.push(vec![start..=end, 0b10_000000..=0b10_111111]);
            } else {
                debug_assert_eq!(start, end + 1);
                // this means we had something where we are split across just a single first-byte boundary
                // e.g. [0b110_00000, 0b10_111111] to [0b110_00001, 0b10_000000]
                // so we don't need full-second-byte middle ranges, only the two at the ends.
            }
        }
    }

    const THREE_BYTE_START: char = '\u{0800}';
    const THREE_BYTE_END: char = '\u{FFFF}';
    if start_char <= THREE_BYTE_END && end_char >= THREE_BYTE_START {
        let start_char = std::cmp::max(start_char, THREE_BYTE_START);
        let mut start_bytes = [0; 4];
        start_char.encode_utf8(&mut start_bytes);

        let end_char = std::cmp::min(end_char, THREE_BYTE_END);
        let mut end_bytes = [0; 4];
        end_char.encode_utf8(&mut end_bytes);

        if start_char == THREE_BYTE_START && end_char == THREE_BYTE_END {
            out.push(vec![
                0b1110_0000..=0b1110_1111,
                0b10_000000..=0b10_111111,
                0b10_000000..=0b10_111111,
            ]);
        } else {
            todo!("Multi-byte char conversion is not fully supported yet.")
        }
    }

    const FOUR_BYTE_START: char = '\u{010000}';
    if end_char >= FOUR_BYTE_START {
        let start_char = std::cmp::max(start_char, FOUR_BYTE_START);
        let mut start_bytes = [0; 4];
        start_char.encode_utf8(&mut start_bytes);

        let mut end_bytes = [0; 4];
        end_char.encode_utf8(&mut end_bytes);

        if start_char == FOUR_BYTE_START && end_char == char::MAX {
            out.push(vec![
                0b11110_000..=0b11110_111,
                0b10_000000..=0b10_111111,
                0b10_000000..=0b10_111111,
                0b10_000000..=0b10_111111,
            ]);
        } else {
            todo!("Multi-byte char conversion is not fully supported yet.")
        }
    }

    return out;
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct U8Atom(RangeInclusive<u8>);
impl U8Atom {
    pub fn check(&self, c: u8) -> bool {
        return self.0.contains(&c);
    }
}
impl std::fmt::Display for U8Atom {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.0.start() == self.0.end() {
            return write!(f, "{}", self.0.start().escape_ascii());
        } else {
            return write!(
                f,
                "[{}-{}]",
                self.0.start().escape_ascii(),
                self.0.end().escape_ascii()
            );
        }
    }
}
impl From<u8> for U8Atom {
    fn from(value: u8) -> Self {
        return U8Atom(value..=value);
    }
}
impl From<RangeInclusive<u8>> for U8Atom {
    fn from(value: RangeInclusive<u8>) -> Self {
        return U8Atom(value);
    }
}
impl TryFrom<char> for U8Atom {
    type Error = std::char::TryFromCharError;

    fn try_from(value: char) -> Result<Self, Self::Error> {
        return Ok(u8::try_from(value)?.into());
    }
}

#[derive(Debug, Clone)]
pub(crate) struct U8Transition {
    pub(crate) to: usize,
    pub(crate) symbol: U8Atom,
}
impl U8Transition {
    pub fn new(to: usize, symbol: U8Atom) -> U8Transition {
        return U8Transition { to, symbol };
    }
    pub fn with_offset(mut self, offset: usize) -> U8Transition {
        self.inplace_offset(offset);
        return self;
    }
    pub fn inplace_offset(&mut self, offset: usize) {
        self.to += offset;
    }
    pub fn add_offset(&self, offset: usize) -> U8Transition {
        return U8Transition {
            to: self.to + offset,
            symbol: self.symbol.clone(),
        };
    }
}
impl std::fmt::Display for U8Transition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        return write!(f, "-({})> {}", self.symbol, self.to);
    }
}

#[derive(Debug, Clone)]
pub struct U8State {
    pub(crate) transitions: Vec<U8Transition>,
    pub(crate) epsilons: Vec<EpsilonTransition>,
}
impl U8State {
    pub const fn new() -> U8State {
        return U8State {
            transitions: Vec::new(),
            epsilons: Vec::new(),
        };
    }
    pub fn with_transition(mut self, to: usize, symbol: U8Atom) -> U8State {
        self.transitions.push(U8Transition::new(to, symbol));
        return self;
    }
    pub fn with_epsilon(mut self, to: usize) -> U8State {
        self.epsilons.push(EpsilonTransition::new(to));
        return self;
    }
    pub fn with_epsilon_special(mut self, to: usize, special: EpsilonType) -> U8State {
        self.epsilons.push(EpsilonTransition { to, special });
        return self;
    }
    pub fn with_offset(mut self, offset: usize) -> U8State {
        self.inplace_offset(offset);
        return self;
    }
    pub fn inplace_offset(&mut self, offset: usize) {
        for t in &mut self.transitions {
            t.inplace_offset(offset);
        }
        for e in &mut self.epsilons {
            e.inplace_offset(offset);
        }
    }
    pub fn add_offset(&self, offset: usize) -> U8State {
        return U8State {
            transitions: self
                .transitions
                .iter()
                .map(|t| t.add_offset(offset))
                .collect(),
            epsilons: self.epsilons.iter().map(|e| e.add_offset(offset)).collect(),
        };
    }
}
impl std::fmt::Display for U8State {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for t in &self.transitions {
            writeln!(f, "  {t}")?;
        }
        for e in &self.epsilons {
            writeln!(f, "  {e}")?;
        }
        return Ok(());
    }
}

/// Each NFA has one start state (`0`) and one accept state (`states.len() - 1`)
#[derive(Debug, Clone)]
pub struct U8NFA {
    pub(crate) states: Vec<U8State>,
}
impl U8NFA {
    fn nfa_empty() -> U8NFA {
        let states = vec![U8State::new()];
        return U8NFA { states };
    }
    fn nfa_byte(c: &U8Atom) -> U8NFA {
        let states = vec![U8State::new().with_transition(1, c.clone()), U8State::new()];
        return U8NFA { states };
    }
    fn nfa_symbol_char(c: char) -> U8NFA {
        let mut bytes = [0u8; 4];
        c.encode_utf8(&mut bytes);
        let states = bytes
            .iter()
            .take(c.len_utf8())
            .enumerate()
            .map(|(i, byte)| U8State::new().with_transition(i + 1, (*byte).into()))
            .chain(std::iter::once(U8State::new()))
            .collect();
        return U8NFA { states };
    }
    fn nfa_symbol(c: &Atom) -> U8NFA {
        let ranges = c.to_ranges();
        let mut states = vec![U8State::new()];

        for range in ranges {
            for byte_ranges in char_range_to_u8_ranges(range) {
                let mut state = 0usize;
                for (i, byte_range) in byte_ranges.iter().enumerate() {
                    if let Some(next_state) = states[state]
                        .transitions
                        .iter()
                        .find(|a| a.symbol.0 == *byte_range)
                    {
                        state = next_state.to;
                    } else if i + 1 == byte_ranges.len() {
                        states[state]
                            .transitions
                            .push(U8Transition::new(usize::MAX, U8Atom(byte_range.clone())));
                        break; // sanity check: should be unnecessary
                    } else {
                        let new_state_idx = states.len();
                        states.push(U8State::new());
                        states[state]
                            .transitions
                            .push(U8Transition::new(new_state_idx, U8Atom(byte_range.clone())));
                        state = new_state_idx;
                    }
                }
            }
        }

        // then insert accept state, replacing [`usize::MAX`] placeholders
        let accept_state_idx = states.len();
        states.push(U8State::new());
        for state in &mut states {
            for transition in &mut state.transitions {
                if transition.to == usize::MAX {
                    transition.to = accept_state_idx;
                }
            }
        }
        // TODO: shared suffix optimizations

        return U8NFA { states };
    }
    fn nfa_union(nodes: &[U8NFA]) -> U8NFA {
        let states_count = 2 + nodes.iter().map(|n| n.states.len()).sum::<usize>();
        let mut states = vec![U8State::new()];
        for nfa in nodes {
            let sub_nfa_start = states.len();
            states[0]
                .epsilons
                .push(EpsilonTransition::new(sub_nfa_start));
            states.extend(
                nfa.states
                    .iter()
                    .map(|state| state.add_offset(sub_nfa_start)),
            );
            states
                .last_mut()
                .unwrap()
                .epsilons
                .push(EpsilonTransition::new(states_count - 1));
        }
        states.push(U8State::new());
        assert_eq!(states_count, states.len());

        return U8NFA { states };
    }
    fn nfa_capture(nfa: &U8NFA, group_num: usize) -> U8NFA {
        let states_count = 2 + nfa.states.len();
        let mut states: Vec<U8State> = std::iter::once(
            U8State::new().with_epsilon_special(1, EpsilonType::StartCapture(group_num)),
        )
        .chain(nfa.states.iter().map(|state| state.add_offset(1)))
        .chain(std::iter::once(U8State::new()))
        .collect();
        assert_eq!(states_count, states.len());
        states[states_count - 2].epsilons.push(EpsilonTransition {
            to: states_count - 1,
            special: EpsilonType::EndCapture(group_num),
        });

        return U8NFA { states };
    }
    fn nfa_concat<T: IntoIterator<Item = U8NFA>>(nodes: T) -> U8NFA {
        let mut states = vec![U8State::new().with_epsilon(1)];

        for nfa in nodes {
            let states_count = states.len();
            states.extend(
                nfa.states
                    .into_iter()
                    .map(|state| state.with_offset(states_count)),
            );
            let states_count = states.len();
            states
                .last_mut()
                .unwrap()
                .epsilons
                .push(EpsilonTransition::new(states_count));
        }

        states.push(U8State::new());
        return U8NFA { states };
    }
    fn nfa_repeat(nfa: &U8NFA, times: usize) -> U8NFA {
        return U8NFA::nfa_concat(std::iter::repeat(nfa).cloned().take(times));
    }
    fn nfa_upto(nfa: &U8NFA, times: usize, longest: bool) -> U8NFA {
        let end_state_idx = 1 + (nfa.states.len() + 1) * times;

        let mut states = vec![U8State::new()
            .with_epsilon(1)
            .with_epsilon(end_state_idx - 1)];
        for i in 0..times {
            let states_count = states.len();
            states.extend(
                nfa.states
                    .iter()
                    .map(|state| state.add_offset(states_count)),
            );
            let transition_state_idx = states.len();
            states
                .last_mut()
                .unwrap()
                .epsilons
                .push(EpsilonTransition::new(transition_state_idx));
            let mut transition_state = U8State::new();
            if i + 1 != times {
                if longest {
                    transition_state
                        .epsilons
                        .push(EpsilonTransition::new(states.len() + 1));
                }

                transition_state
                    .epsilons
                    .push(EpsilonTransition::new(end_state_idx - 1));
                if !longest {
                    transition_state
                        .epsilons
                        .push(EpsilonTransition::new(states.len() + 1));
                }
            }
            states.push(transition_state);
        }

        return U8NFA { states };
    }
    fn nfa_star(nfa: U8NFA, longest: bool) -> U8NFA {
        let end_state_idx = 1 + nfa.states.len();
        let mut start_state = U8State::new();
        if !longest {
            start_state
                .epsilons
                .push(EpsilonTransition::new(end_state_idx));
        }
        start_state.epsilons.push(EpsilonTransition::new(1));
        if longest {
            start_state
                .epsilons
                .push(EpsilonTransition::new(end_state_idx));
        }
        let mut states: Vec<U8State> = std::iter::once(start_state)
            .chain(nfa.states.into_iter().map(|state| state.with_offset(1)))
            .chain(std::iter::once(U8State::new()))
            .collect();
        states[end_state_idx - 1]
            .epsilons
            .push(EpsilonTransition::new(0));
        return U8NFA { states };
    }
    fn nfa_start() -> U8NFA {
        let states = vec![
            U8State::new().with_epsilon_special(1, EpsilonType::StartAnchor),
            U8State::new(),
        ];
        return U8NFA { states };
    }
    fn nfa_end() -> U8NFA {
        let states = vec![
            U8State::new().with_epsilon_special(1, EpsilonType::EndAnchor),
            U8State::new(),
        ];
        return U8NFA { states };
    }
    fn nfa_never() -> U8NFA {
        let states = vec![U8State::new(), U8State::new()];
        return U8NFA { states };
    }
    /// Converts from a char-based NFA
    ///
    /// Does not include any optimizations
    fn build(nfa: &WorkingNFA) -> U8NFA {
        let mut states: Vec<U8State> = Vec::new();
        let mut sub_states: Vec<U8State> = Vec::new();
        for state in &nfa.states {
            let mut new_state = U8State {
                transitions: Vec::new(),
                epsilons: state.epsilons.clone(),
            };
            // Decompose char transitions into byte transitions
            // `a -x> b` will become an expanded nfa with initial state `a` and accept state `b`
            for t in &state.transitions {
                let symbol_nfa = U8NFA::nfa_symbol(&t.symbol);
                let symbol_nfa_accept = symbol_nfa.states.len() - 1;
                let sub_states_offset = nfa.states.len() + sub_states.len() - 1;

                // Updates transition indices
                let map_transition = |sub_t: &U8Transition| {
                    if sub_t.to == symbol_nfa_accept {
                        U8Transition::new(t.to, sub_t.symbol.clone())
                    } else {
                        sub_t.add_offset(sub_states_offset)
                    }
                };

                new_state
                    .transitions
                    .extend(symbol_nfa.states[0].transitions.iter().map(map_transition));

                for sub_state in symbol_nfa
                    .states
                    .iter()
                    .skip(1)
                    .take(symbol_nfa.states.len() - 2)
                {
                    let sub_state = U8State {
                        transitions: sub_state.transitions.iter().map(map_transition).collect(),
                        epsilons: Vec::new(),
                    };
                    sub_states.push(sub_state);
                }
            }
            states.push(new_state);
        }

        states.extend_from_slice(&sub_states);
        let new_accept_state = states.len();
        states.push(U8State::new());
        states[nfa.states.len() - 1]
            .epsilons
            .push(EpsilonTransition::new(new_accept_state));
        return U8NFA { states };
    }
    /// Converts from a char-based NFA
    pub fn new(nfa: &WorkingNFA) -> U8NFA {
        let mut nfa = U8NFA::build(nfa);

        while nfa.optimize_pass() {}
        nfa.remove_unreachable();
        return nfa;
    }

    /// Helper function for removing a set of states.
    ///
    /// These states should have no incoming transitions.
    fn remove_dead_states<T: IntoIterator<Item = bool>>(&mut self, dead_states: T) {
        let state_map: Vec<usize> = dead_states
            .into_iter()
            .scan(0, |s, dead| {
                if dead {
                    return Some(usize::MAX);
                } else {
                    let out = *s;
                    *s += 1;
                    return Some(out);
                }
            })
            .collect();
        self.states = self
            .states
            .iter()
            .enumerate()
            .filter(|(i, _)| state_map[*i] != usize::MAX)
            .map(|(_, state)| state)
            .cloned()
            .collect();

        for state in &mut self.states {
            for t in &mut state.transitions {
                t.to = state_map[t.to];
            }
            for t in &mut state.epsilons {
                t.to = state_map[t.to];
            }
        }
    }

    /// Optimizes the NFA graph.
    ///
    /// Returns `true` if changes were made (meaning another pass should be tried).
    fn optimize_pass(&mut self) -> bool {
        let mut changed = false;
        let state_count = self.states.len();

        let mut dead_states = vec![false; self.states.len()];

        // Skip redundant states
        // Special transitions (anchors + capture groups) are treated similar to non-epsilon transitions
        for state_idx in 1..state_count - 1 {
            let incoming: Vec<(usize, usize)> = self
                .states
                .iter()
                .enumerate()
                .flat_map(|(s_i, s)| s.transitions.iter().enumerate().map(move |(t, _)| (s_i, t)))
                .filter(|(s, t)| self.states[*s].transitions[*t].to == state_idx)
                .collect();
            let incoming_eps: Vec<(usize, usize)> = self
                .states
                .iter()
                .enumerate()
                .flat_map(|(s_i, s)| s.epsilons.iter().enumerate().map(move |(e, _)| (s_i, e)))
                .filter(|(s, e)| self.states[*s].epsilons[*e].to == state_idx)
                .collect();

            match (
                incoming.as_slice(),
                incoming_eps.as_slice(),
                self.states[state_idx].transitions.len(),
                self.states[state_idx].epsilons.len(),
            ) {
                // `as -xes> b -e> c` can become `as -xes> c` (assuming no other transitions)
                (incoming, incoming_eps, 0, 1)
                    if self.states[state_idx].epsilons[0].special == EpsilonType::None =>
                {
                    let to = self.states[state_idx].epsilons[0].to;
                    for (s, t) in incoming {
                        self.states[*s].transitions[*t].to = to;
                    }
                    for (s, e) in incoming_eps {
                        self.states[*s].epsilons[*e].to = to;
                    }
                    dead_states[state_idx] = true;
                    self.states[state_idx].epsilons = Vec::new();
                    changed = true;
                    continue;
                }
                // `a -e> b -es> cs` can become `a -es> cs` (assuming no other transitions)
                (&[], &[(incoming_state, incoming_eps)], 0, _)
                    if self.states[incoming_state].epsilons[incoming_eps].special
                        == EpsilonType::None =>
                {
                    let outgoing_eps = std::mem::take(&mut self.states[state_idx].epsilons);
                    let after = self.states[incoming_state]
                        .epsilons
                        .split_off(incoming_eps + 1);
                    self.states[incoming_state].epsilons.pop();
                    self.states[incoming_state]
                        .epsilons
                        .extend_from_slice(&outgoing_eps);
                    self.states[incoming_state]
                        .epsilons
                        .extend_from_slice(&after);

                    dead_states[state_idx] = true;
                    changed = true;
                    continue;
                }
                _ => {}
            }

            // TODO:
            // `a -e> b -xes> cs` can become `a -xes> cs` (assuming no other transitions)
            // `a -e> b -e> a` can combine `a` and `b` (including other transitions)
            // TODO: might cause additional overhead in some cases, should we do
            // ??? `a -x> b -es> cs` can become `a -xs> cs`
            // ??? `as -es> b -x> c` can become `as -xs> c`
        }
        if !changed {
            return changed;
        }

        self.remove_dead_states(dead_states);

        return changed;
    }

    /// Finds the states that can be reached from the start via any path
    fn states_reachable_start(&self) -> Vec<bool> {
        let mut reachable = vec![false; self.states.len()];
        reachable[0] = true;
        let mut stack = vec![0];

        while let Some(state) = stack.pop() {
            for src in &self.states[state].epsilons {
                if !reachable[src.to] {
                    stack.push(src.to);
                }
                reachable[src.to] = true;
            }
            for src in &self.states[state].transitions {
                if !reachable[src.to] {
                    stack.push(src.to);
                }
                reachable[src.to] = true;
            }
        }

        return reachable;
    }
    /// Finds the states that can reach the end via any path
    fn states_reachable_end(&self) -> Vec<bool> {
        let mut reverse = vec![Vec::new(); self.states.len()];
        for (i, state) in self.states.iter().enumerate() {
            for e in &state.epsilons {
                reverse[e.to].push(i);
            }
            for t in &state.transitions {
                reverse[t.to].push(i);
            }
        }

        let mut reachable = vec![false; self.states.len()];
        reachable[self.states.len() - 1] = true;
        let mut stack = vec![self.states.len() - 1];

        while let Some(state) = stack.pop() {
            for src in &reverse[state] {
                if !reachable[*src] {
                    stack.push(*src);
                }
                reachable[*src] = true;
            }
        }

        return reachable;
    }

    /// Removes all nodes that cannot be reached or cannot reach the end.
    ///
    /// Ignores special epsilon types (so should be called after they have been resolved)
    fn remove_unreachable(&mut self) {
        let reach_start = self.states_reachable_start();
        let reach_end = self.states_reachable_end();

        // Remove transitions that involve redundant states
        for state in &mut self.states {
            state
                .epsilons
                .retain(|e| reach_start[e.to] && reach_end[e.to]);
            state
                .transitions
                .retain(|t| reach_start[t.to] && reach_end[t.to]);
        }

        // Then remove the states
        self.remove_dead_states(
            std::iter::zip(reach_start.into_iter(), reach_end.into_iter()).map(|(a, b)| !a || !b),
        );
    }

    /// Finds the number of capture groups in this NFA
    pub fn num_capture_groups(&self) -> usize {
        return self
            .states
            .iter()
            .flat_map(|state| &state.epsilons)
            .map(|eps| match eps.special {
                EpsilonType::StartCapture(n) => n,
                _ => 0,
            })
            .max()
            .unwrap_or(0)
            + 1;
    }

    /// Writes a LaTeX TikZ representation to visualize the graph.
    ///
    /// If `include_doc` is `true`, will include the headers.
    /// Otherwise, you should include `\usepackage{tikz}` and `\usetikzlibrary{automata, positioning}`.
    pub fn to_tikz(&self, include_doc: bool) -> String {
        // TODO: make the layout better
        fn escape_latex(text: String) -> String {
            return text
                .chars()
                .map(|c| match c {
                    '\\' => r"{\textbackslash}".to_string(),
                    '&' => r"\&".to_string(),
                    '%' => r"\%".to_string(),
                    '$' => r"\$".to_string(),
                    '#' => r"\#".to_string(),
                    '_' => r"\_".to_string(),
                    '{' => r"\{".to_string(),
                    '}' => r"\}".to_string(),
                    '~' => r"{\textasciitilde}".to_string(),
                    '^' => r"{\textasciicircum}".to_string(),
                    c => c.to_string(),
                })
                .collect();
        }

        let mut text_parts: Vec<String> = Vec::new();
        if include_doc {
            text_parts.push(
                "\\documentclass{standalone}\n\\usepackage{tikz}\n\\usetikzlibrary{automata, positioning}\n\\begin{document}\n"
                .into(),
            );
        }
        text_parts.push("\\begin{tikzpicture}[node distance=2cm, auto]\n".into());

        let mut transition_parts = Vec::new();

        for (i, state) in self.states.iter().enumerate() {
            if i == 0 {
                text_parts.push("\\node[state, initial](q0){$q_0$};\n".into());
            } else if i + 1 == self.states.len() {
                text_parts.push(format!(
                    "\\node[state, accepting, right of=q{}](q{i}){{$q_{{{i}}}$}};\n",
                    i - 1,
                ));
            } else {
                text_parts.push(format!(
                    "\\node[state, right of=q{}](q{i}){{$q_{{{i}}}$}};\n",
                    i - 1,
                ));
            }

            for U8Transition { to, symbol } in &state.transitions {
                let bend = match to.cmp(&i) {
                    std::cmp::Ordering::Less => "[bend left] ",
                    std::cmp::Ordering::Equal => "[loop below]",
                    std::cmp::Ordering::Greater => "[bend left] ",
                };
                transition_parts.push(format!(
                    "\\path[->] (q{i}) edge {bend} node {{{}}} (q{to});\n",
                    escape_latex(symbol.to_string()),
                ));
            }
            for EpsilonTransition { to, special } in &state.epsilons {
                let bend = match to.cmp(&i) {
                    std::cmp::Ordering::Less => "[bend left] ",
                    std::cmp::Ordering::Equal => "[loop below]",
                    std::cmp::Ordering::Greater => "[bend left] ",
                };
                let label = match special {
                    EpsilonType::None => r"$\epsilon$".to_string(),
                    EpsilonType::StartAnchor => r"{\textasciicircum}".to_string(),
                    EpsilonType::EndAnchor => r"\$".to_string(),
                    EpsilonType::StartCapture(group) => format!("{group}("),
                    EpsilonType::EndCapture(group) => format!("){group}"),
                };
                transition_parts.push(format!(
                    "\\path[->] (q{i}) edge {bend} node {{{label}}} (q{to});\n"
                ));
            }
        }
        text_parts.extend_from_slice(&transition_parts);

        text_parts.push("\\end{tikzpicture}\n".into());
        if include_doc {
            text_parts.push("\\end{document}\n".into());
        }
        return text_parts.into_iter().collect();
    }

    /// Using the classical NFA algorithm to do a simple boolean test on a string.
    pub fn test(&self, text: &str) -> bool {
        let mut list = vec![false; self.states.len()];
        let mut new_list = vec![false; self.states.len()];
        list[0] = true;

        // Adds all states reachable by epsilon transitions
        let propogate_epsilon = |list: &mut Vec<bool>, idx: usize| {
            let mut stack: Vec<usize> = list
                .iter()
                .enumerate()
                .filter_map(|(i, set)| set.then_some(i))
                .collect();

            while let Some(from) = stack.pop() {
                for EpsilonTransition { to, special } in &self.states[from].epsilons {
                    if list[from]
                        && !list[*to]
                        && (match special {
                            EpsilonType::StartAnchor => idx == 0,
                            EpsilonType::EndAnchor => idx == text.len(),
                            _ => true,
                        })
                    {
                        stack.push(*to);
                        list[*to] = true;
                    }
                }
            }
        };

        for (i, c) in text.as_bytes().iter().enumerate() {
            propogate_epsilon(&mut list, i);
            for (from, state) in self.states.iter().enumerate() {
                if !list[from] {
                    continue;
                }

                for U8Transition { to, symbol } in &state.transitions {
                    if symbol.check(*c) {
                        new_list[*to] = true;
                    }
                }
            }
            let tmp = list;
            list = new_list;
            new_list = tmp;
            new_list.fill(false);
        }
        propogate_epsilon(&mut list, text.len());
        return *list.last().unwrap_or(&false);
    }
}
impl std::fmt::Display for U8NFA {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (i, state) in self.states.iter().enumerate() {
            writeln!(f, "State {i}:")?;
            for e in &state.epsilons {
                writeln!(f, "  {e}")?;
            }
            for t in &state.transitions {
                writeln!(f, "  {t}")?;
            }
        }
        return Ok(());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{config::Config, parse_tree::ERE, simplified_tree::SimplifiedTreeNode};

    #[test]
    fn abbc_raw() {
        let nfa = U8NFA {
            states: vec![
                U8State::new().with_transition(1, b'a'.into()),
                U8State::new().with_transition(2, b'b'.into()),
                U8State::new()
                    .with_transition(3, b'c'.into())
                    .with_epsilon(1),
                U8State::new(),
            ],
        };
        println!("{}", nfa.to_tikz(true));

        assert!(nfa.test("abc"));
        assert!(nfa.test("abbc"));
        assert!(nfa.test("abbbc"));
        assert!(nfa.test("abbbbc"));

        assert!(!nfa.test("ac"));
        assert!(!nfa.test("abcc"));
        assert!(!nfa.test("bac"));
        assert!(!nfa.test("acb"));
    }

    #[test]
    fn phone_number() {
        let ere = ERE::parse_str(r"^(\+1 )?[0-9]{3}-[0-9]{3}-[0-9]{4}$").unwrap();
        let (tree, capture_groups) = SimplifiedTreeNode::from_ere(&ere, &Config::default());
        assert_eq!(capture_groups, 2);
        let nfa = WorkingNFA::new(&tree);
        let nfa = U8NFA::new(&nfa);
        println!("{}", nfa.to_tikz(true));

        assert!(nfa.test("012-345-6789"));
        assert!(nfa.test("987-654-3210"));
        assert!(nfa.test("+1 555-555-5555"));
        assert!(nfa.test("123-555-9876"));

        assert!(!nfa.test("abcd"));
        assert!(!nfa.test("0123456789"));
        assert!(!nfa.test("012--345-6789"));
        assert!(!nfa.test("(555) 555-5555"));
        assert!(!nfa.test("1 555-555-5555"));
    }

    #[test]
    fn double_loop() {
        let ere = ERE::parse_str(r"^.*(.*)*$").unwrap();
        let (tree, capture_groups) = SimplifiedTreeNode::from_ere(&ere, &Config::default());
        assert_eq!(capture_groups, 2);
        let nfa = WorkingNFA::new(&tree);
        let nfa = U8NFA::new(&nfa);
        // println!("{}", nfa.to_tikz(true));

        assert!(nfa.test(""));
        assert!(nfa.test("asdf"));
        assert!(nfa.test("1234567"));
        assert!(nfa.test("0"));

        assert!(!nfa.test("\0"));
    }

    #[test]
    fn good_anchored_start() {
        let ere = ERE::parse_str(r"^a|b*^c|d^|n").unwrap();
        let (tree, capture_groups) = SimplifiedTreeNode::from_ere(&ere, &Config::default());
        assert_eq!(capture_groups, 1);
        let nfa = WorkingNFA::new(&tree);
        let nfa = U8NFA::new(&nfa);
        // println!("{}", nfa.to_tikz(true));

        assert!(nfa.test("a"));
        assert!(nfa.test("c"));
        assert!(nfa.test("cq"));
        assert!(nfa.test("wwwnwww"));

        assert!(!nfa.test(""));
        assert!(!nfa.test("qb"));
        assert!(!nfa.test("qc"));
        assert!(!nfa.test("b"));
        assert!(!nfa.test("bc"));
        assert!(!nfa.test("bbbbbbc"));
        assert!(!nfa.test("d"));
    }

    #[test]
    fn good_anchored_end() {
        let ere = ERE::parse_str(r"a$|b$c*|$d|n").unwrap();
        let (tree, capture_groups) = SimplifiedTreeNode::from_ere(&ere, &Config::default());
        assert_eq!(capture_groups, 1);
        let nfa = WorkingNFA::new(&tree);
        let nfa = U8NFA::new(&nfa);
        println!("{}", nfa.to_tikz(true));

        assert!(nfa.test("a"));
        assert!(nfa.test("b"));
        assert!(nfa.test("qb"));
        assert!(nfa.test("wwwnwww"));

        assert!(!nfa.test(""));
        assert!(!nfa.test("bq"));
        assert!(!nfa.test("qc"));
        assert!(!nfa.test("c"));
        assert!(!nfa.test("bc"));
        assert!(!nfa.test("bcccccc"));
        assert!(!nfa.test("d"));
    }

    #[test]
    fn range_digit() {
        let ere = ERE::parse_str(r"^[[:digit:].]$").unwrap();
        let (tree, capture_groups) = SimplifiedTreeNode::from_ere(&ere, &Config::default());
        assert_eq!(capture_groups, 1);
        let nfa = WorkingNFA::new(&tree);
        let nfa = U8NFA::new(&nfa);
        println!("{}", nfa.to_tikz(true));

        assert!(nfa.test("0"));
        assert!(nfa.test("1"));
        assert!(nfa.test("9"));
        assert!(nfa.test("."));

        assert!(!nfa.test(""));
        assert!(!nfa.test("a"));
        assert!(!nfa.test("11"));
        assert!(!nfa.test("1."));
        assert!(!nfa.test(".2"));
        assert!(!nfa.test("09"));
        assert!(!nfa.test("d"));
    }

    #[test]
    fn dot() {
        let nfa = U8NFA::nfa_symbol(&&Atom::CharClass(crate::parse_tree::CharClass::Dot));
        assert!(!nfa.test("\0"));
        for c in '\u{0001}'..=char::MAX {
            let txt = c.to_string();
            let mut bytes = [0; 4];
            c.encode_utf8(&mut bytes);
            assert!(
                nfa.test(&txt),
                "Expected {c} (code point: 0x{:X}, utf8: 0x{:02X}{:02X}{:02X}{:02X}) to be matched by regex dot.", c as u32, bytes[0], bytes[1], bytes[2], bytes[3]
            );
        }

        let ere = ERE::parse_str(r"^.$").unwrap();
        let (tree, capture_groups) = SimplifiedTreeNode::from_ere(&ere, &Config::default());
        assert_eq!(capture_groups, 1);
        let nfa = WorkingNFA::new(&tree);
        let nfa = U8NFA::new(&nfa);
        println!("{}", nfa.to_tikz(true));

        assert!(nfa.test("0"));
        assert!(nfa.test("1"));
        assert!(nfa.test("a"));
        assert!(nfa.test("\u{0001}"));
        assert!(nfa.test("9"));
        assert!(nfa.test("."));
        assert!(nfa.test("\u{1234}"));

        assert!(!nfa.test(""));
        assert!(!nfa.test("\0"));
        assert!(!nfa.test("ab"));
        assert!(!nfa.test("11"));
        assert!(!nfa.test("1."));
        assert!(!nfa.test(".2"));
        assert!(!nfa.test("09"));
        assert!(!nfa.test("\u{1234}\u{4321}"));
    }
}
