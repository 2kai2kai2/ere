/// This recognizes "true" regular expressions, meaning implicit anchors and no capture groups.
/// Capture groups are treated as simple concatenation.
/// However, it also includes additional stuff like special start/end anchors
use crate::parse_tree::Atom;
use crate::simplified_tree::SimplifiedTreeNode;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum WorkingEpsilonType {
    None,
    StartAnchor,
    EndAnchor,
    StartCapture(usize),
    EndCapture(usize),
}

/// An epsilon transition for the [`WorkingNFA`]
#[derive(Debug, Clone, Copy)]
pub struct WorkingEpsilonTransition {
    pub(crate) from: usize,
    pub(crate) to: usize,
    pub(crate) special: WorkingEpsilonType,
}
impl WorkingEpsilonTransition {
    pub(crate) const fn new(from: usize, to: usize) -> WorkingEpsilonTransition {
        return WorkingEpsilonTransition {
            from,
            to,
            special: WorkingEpsilonType::None,
        };
    }
    pub(crate) const fn with_offset(self, offset: usize) -> WorkingEpsilonTransition {
        return WorkingEpsilonTransition {
            from: self.from + offset,
            to: self.to + offset,
            special: self.special,
        };
    }
    pub(crate) const fn add_offset(&self, offset: usize) -> WorkingEpsilonTransition {
        return WorkingEpsilonTransition {
            from: self.from + offset,
            to: self.to + offset,
            special: self.special,
        };
    }
}
impl std::fmt::Display for WorkingEpsilonTransition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        return write!(f, "{} -> {}", self.from, self.to);
    }
}

#[derive(Debug, Clone)]
pub(crate) struct WorkingTransition {
    pub(crate) from: usize,
    pub(crate) to: usize,
    pub(crate) symbol: Atom,
}
impl WorkingTransition {
    pub fn new(from: usize, to: usize, symbol: Atom) -> WorkingTransition {
        return WorkingTransition { from, to, symbol };
    }
    pub fn with_offset(self, offset: usize) -> WorkingTransition {
        return WorkingTransition {
            from: self.from + offset,
            to: self.to + offset,
            symbol: self.symbol,
        };
    }
    pub fn add_offset(&self, offset: usize) -> WorkingTransition {
        return WorkingTransition {
            from: self.from + offset,
            to: self.to + offset,
            symbol: self.symbol.clone(),
        };
    }
}
impl std::fmt::Display for WorkingTransition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        return write!(f, "{} -({})> {}", self.from, self.symbol, self.to);
    }
}

/// Each NFA has one start state (`0`) and one accept state (`states - 1`)
#[derive(Debug)]
pub struct WorkingNFA {
    pub(crate) transitions: Vec<WorkingTransition>,
    pub(crate) epsilons: Vec<WorkingEpsilonTransition>,
    pub(crate) states: usize,
}
impl WorkingNFA {
    const fn build_empty() -> WorkingNFA {
        return WorkingNFA {
            transitions: Vec::new(),
            epsilons: Vec::new(),
            states: 1,
        };
    }
    fn build_symbol(c: &Atom) -> WorkingNFA {
        return WorkingNFA {
            transitions: vec![WorkingTransition::new(0, 1, c.clone())],
            epsilons: Vec::new(),
            states: 2,
        };
    }
    fn build_union(nodes: &[SimplifiedTreeNode]) -> WorkingNFA {
        let sub_nfas: Vec<WorkingNFA> = nodes.iter().map(WorkingNFA::build).collect();
        let states = 2 + sub_nfas.iter().map(|n| n.states).sum::<usize>();
        let mut transitions = Vec::new();
        let mut epsilons = Vec::new();
        let mut used_states = 1usize;
        for nfa in sub_nfas {
            // Epsilon transition in
            epsilons.push(WorkingEpsilonTransition::new(0, used_states));
            // Copy internal transitions
            transitions.extend(
                nfa.transitions
                    .into_iter()
                    .map(|t| t.with_offset(used_states)),
            );
            epsilons.extend(nfa.epsilons.into_iter().map(|t| t.with_offset(used_states)));
            // Epsilon transition out
            epsilons.push(WorkingEpsilonTransition::new(
                used_states + nfa.states - 1,
                states - 1,
            ));
            used_states += nfa.states;
        }
        assert_eq!(used_states + 1, states);

        return WorkingNFA {
            transitions,
            epsilons,
            states,
        };
    }
    fn build_capture(tree: &SimplifiedTreeNode, group_num: usize) -> WorkingNFA {
        let nfa = WorkingNFA::build(tree);
        let states = nfa.states + 2;
        let mut epsilons: Vec<_> = nfa.epsilons.into_iter().map(|t| t.with_offset(1)).collect();
        let transitions: Vec<_> = nfa
            .transitions
            .into_iter()
            .map(|t| t.with_offset(1))
            .collect();
        epsilons.push(WorkingEpsilonTransition {
            from: 0,
            to: 1,
            special: WorkingEpsilonType::StartCapture(group_num),
        });
        epsilons.push(WorkingEpsilonTransition {
            from: states - 2,
            to: states - 1,
            special: WorkingEpsilonType::EndCapture(group_num),
        });
        return WorkingNFA {
            transitions,
            epsilons,
            states,
        };
    }
    fn build_concat(nodes: &[SimplifiedTreeNode]) -> WorkingNFA {
        if nodes.is_empty() {
            return WorkingNFA {
                transitions: Vec::new(),
                epsilons: Vec::new(),
                states: 1,
            };
        }

        let mut states = 1usize;
        let mut epsilons = vec![WorkingEpsilonTransition::new(0, 1)];
        let mut transitions = vec![];

        for sub in nodes {
            let nfa = WorkingNFA::build(sub);
            transitions.extend(nfa.transitions.into_iter().map(|t| t.with_offset(states)));
            epsilons.extend(nfa.epsilons.into_iter().map(|t| t.with_offset(states)));
            states += nfa.states;
            epsilons.push(WorkingEpsilonTransition::new(states - 1, states));
        }

        return WorkingNFA {
            transitions,
            epsilons,
            states: states + 1,
        };
    }
    fn build_repeat(tree: &SimplifiedTreeNode, times: usize) -> WorkingNFA {
        let mut states = 1usize;
        let mut epsilons = vec![WorkingEpsilonTransition::new(0, 1)];
        let mut transitions = vec![];
        let nfa = WorkingNFA::build(tree);

        for _ in 0..times {
            transitions.extend(nfa.transitions.iter().map(|t| t.add_offset(states)));
            epsilons.extend(nfa.epsilons.iter().map(|t| t.add_offset(states)));
            states += nfa.states;
            epsilons.push(WorkingEpsilonTransition::new(states - 1, states));
        }

        return WorkingNFA {
            transitions,
            epsilons,
            states: states + 1,
        };
    }
    fn build_upto(tree: &SimplifiedTreeNode, times: usize) -> WorkingNFA {
        let nfa = WorkingNFA::build(tree);
        let states = 2 + nfa.states * times;
        let mut used_states = 1usize;
        let mut transitions = Vec::new();
        let mut epsilons = vec![
            WorkingEpsilonTransition::new(0, 1),
            WorkingEpsilonTransition::new(0, states - 1),
        ];

        for _ in 0..times {
            transitions.extend(nfa.transitions.iter().map(|t| t.add_offset(states)));
            epsilons.extend(nfa.epsilons.iter().map(|t| t.add_offset(states)));
            used_states += nfa.states;
            epsilons.push(WorkingEpsilonTransition::new(used_states - 1, used_states));
            if used_states != states - 1 {
                epsilons.push(WorkingEpsilonTransition::new(used_states - 1, states - 1));
            }
        }

        return WorkingNFA {
            transitions,
            epsilons,
            states,
        };
    }
    fn build_star(tree: &SimplifiedTreeNode) -> WorkingNFA {
        let nfa = WorkingNFA::build(tree);
        let states = 2 + nfa.states;
        let transitions: Vec<_> = nfa
            .transitions
            .into_iter()
            .map(|t| t.with_offset(1))
            .collect();
        let mut epsilons: Vec<_> = nfa.epsilons.into_iter().map(|t| t.with_offset(1)).collect();
        epsilons.push(WorkingEpsilonTransition::new(0, 1));
        epsilons.push(WorkingEpsilonTransition::new(states - 2, 0));
        epsilons.push(WorkingEpsilonTransition::new(0, states - 1));
        return WorkingNFA {
            transitions,
            epsilons,
            states,
        };
    }
    fn build_start() -> WorkingNFA {
        return WorkingNFA {
            transitions: Vec::new(),
            epsilons: vec![WorkingEpsilonTransition {
                from: 0,
                to: 1,
                special: WorkingEpsilonType::StartAnchor,
            }],
            states: 2,
        };
    }
    fn build_end() -> WorkingNFA {
        return WorkingNFA {
            transitions: Vec::new(),
            epsilons: vec![WorkingEpsilonTransition {
                from: 0,
                to: 1,
                special: WorkingEpsilonType::EndAnchor,
            }],
            states: 2,
        };
    }
    const fn build_never() -> WorkingNFA {
        return WorkingNFA {
            transitions: Vec::new(),
            epsilons: Vec::new(),
            states: 2,
        };
    }
    /// Builds a very inefficient but valid NFA
    ///
    /// Should be optimized using [`NFA::optimize_pass`]
    fn build(tree: &SimplifiedTreeNode) -> WorkingNFA {
        return match tree {
            SimplifiedTreeNode::Empty => WorkingNFA::build_empty(),
            SimplifiedTreeNode::Symbol(c) => WorkingNFA::build_symbol(c),
            SimplifiedTreeNode::Union(nodes) => WorkingNFA::build_union(nodes),
            SimplifiedTreeNode::Capture(tree, group_num) => {
                WorkingNFA::build_capture(&tree, *group_num)
            }
            SimplifiedTreeNode::Concat(nodes) => WorkingNFA::build_concat(nodes),
            SimplifiedTreeNode::Repeat(tree, times) => WorkingNFA::build_repeat(tree, *times),
            SimplifiedTreeNode::UpTo(tree, times) => WorkingNFA::build_upto(tree, *times),
            SimplifiedTreeNode::Star(tree) => WorkingNFA::build_star(tree),
            SimplifiedTreeNode::Start => WorkingNFA::build_start(),
            SimplifiedTreeNode::End => WorkingNFA::build_end(),
            SimplifiedTreeNode::Never => WorkingNFA::build_never(),
        };
    }
    pub fn new(tree: &SimplifiedTreeNode) -> WorkingNFA {
        let mut nfa = WorkingNFA::build(tree);
        nfa.remove_start_anchors();
        nfa.remove_end_anchors();
        assert!(!nfa.has_anchors());
        while nfa.optimize_pass() {}
        nfa.remove_unreachable();
        return nfa;
    }

    /// Removes start anchors by replacing them with a never or epsilon from the start
    fn remove_start_anchors(&mut self) {
        let mut zero_len_reachable = vec![false; self.states];
        zero_len_reachable[0] = true;
        let mut changed = false;
        let mut new_epsilons = Vec::new();
        loop {
            for e in self.epsilons.iter() {
                if zero_len_reachable[e.from] && !zero_len_reachable[e.to] {
                    zero_len_reachable[e.to] = true;
                    changed = true;
                    if e.special == WorkingEpsilonType::StartAnchor {
                        new_epsilons.push(WorkingEpsilonTransition::new(0, e.to + 2));
                    }
                }
            }
            if !changed {
                break;
            }
            changed = false;
        }
        self.epsilons = self
            .epsilons
            .iter()
            .filter(|t| t.special != WorkingEpsilonType::StartAnchor)
            .cloned()
            .map(|e| e.with_offset(2))
            .chain(new_epsilons.into_iter())
            .collect();
        for t in &mut self.transitions {
            t.from += 2;
            t.to += 2;
        }
        self.states += 2;
        self.epsilons.push(WorkingEpsilonTransition::new(0, 1));
        self.epsilons.push(WorkingEpsilonTransition::new(1, 2));
        self.transitions.push(WorkingTransition::new(
            1,
            1,
            Atom::NonmatchingList(Vec::new()),
        ));
    }

    /// Removes end anchors by replacing them with a never or epsilon to the end
    fn remove_end_anchors(&mut self) {
        let mut zero_len_reachable = vec![false; self.states];
        zero_len_reachable[self.states - 1] = true;
        let mut changed = false;
        let mut new_epsilons = Vec::new();
        loop {
            for e in self.epsilons.iter() {
                if !zero_len_reachable[e.from] && zero_len_reachable[e.to] {
                    zero_len_reachable[e.from] = true;
                    changed = true;
                    if e.special == WorkingEpsilonType::EndAnchor {
                        new_epsilons.push(WorkingEpsilonTransition::new(e.from, self.states + 1));
                    }
                }
            }
            if !changed {
                break;
            }
            changed = false;
        }
        self.epsilons = self
            .epsilons
            .iter()
            .filter(|t| t.special != WorkingEpsilonType::EndAnchor)
            .cloned()
            .chain(new_epsilons.into_iter())
            .collect();
        self.states += 2;
        self.epsilons.push(WorkingEpsilonTransition::new(
            self.states - 3,
            self.states - 2,
        ));
        self.epsilons.push(WorkingEpsilonTransition::new(
            self.states - 2,
            self.states - 1,
        ));
        self.transitions.push(WorkingTransition::new(
            self.states - 2,
            self.states - 2,
            Atom::NonmatchingList(Vec::new()),
        ));
    }
    fn has_anchors(&self) -> bool {
        return self.epsilons.iter().any(|e| {
            e.special == WorkingEpsilonType::StartAnchor
                || e.special == WorkingEpsilonType::EndAnchor
        });
    }

    /// Optimizes the NFA graph
    ///
    /// Returns `true` if changes were made (meaning another pass should be tried).
    fn optimize_pass(&mut self) -> bool {
        let mut changed = false;

        let mut dead_states = vec![false; self.states];

        // Skip redundant states
        // Special transitions (anchors + capture groups) are treated similar to non-epsilon transitions
        for state in 1..self.states - 1 {
            let incoming: Vec<usize> = self
                .transitions
                .iter()
                .enumerate()
                .filter(|(_, t)| t.to == state)
                .map(|(i, _)| i)
                .collect();
            let outgoing: Vec<usize> = self
                .transitions
                .iter()
                .enumerate()
                .filter(|(_, t)| t.from == state)
                .map(|(i, _)| i)
                .collect();
            let incoming_eps: Vec<usize> = self
                .epsilons
                .iter()
                .enumerate()
                .filter(|(_, t)| t.to == state)
                .map(|(i, _)| i)
                .collect();
            let outgoing_eps: Vec<usize> = self
                .epsilons
                .iter()
                .enumerate()
                .filter(|(_, t)| t.from == state)
                .map(|(i, _)| i)
                .collect();

            match (
                incoming.as_slice(),
                incoming_eps.as_slice(),
                outgoing.as_slice(),
                outgoing_eps.as_slice(),
            ) {
                // `as -xes> b -e> c` can become `as -xes> c` (assuming no other transitions)
                (incoming, incoming_eps, &[], &[outgoing_eps])
                    if self.epsilons[outgoing_eps].special == WorkingEpsilonType::None =>
                {
                    for idx in incoming {
                        self.transitions[*idx].to = self.epsilons[outgoing_eps].to;
                    }
                    for idx in incoming_eps {
                        self.epsilons[*idx].to = self.epsilons[outgoing_eps].to;
                    }
                    dead_states[state] = true;
                    self.epsilons.swap_remove(outgoing_eps);
                    changed = true;
                    continue;
                }
                // `a -e> b -xes> cs` can become `a -xes> cs` (assuming no other transitions)
                (&[], &[incoming_eps], outgoing, outgoing_eps)
                    if self.epsilons[incoming_eps].special == WorkingEpsilonType::None =>
                {
                    for idx in outgoing {
                        self.transitions[*idx].from = self.epsilons[incoming_eps].from;
                    }
                    for idx in outgoing_eps {
                        self.epsilons[*idx].from = self.epsilons[incoming_eps].from;
                    }
                    dead_states[state] = true;
                    self.epsilons.swap_remove(incoming_eps);
                    changed = true;
                    continue;
                }
                _ => {}
            }

            // TODO:
            // `a -e> b -e> a` can combine `a` and `b` (including other transitions)
            // TODO: might cause additional overhead in some cases, should we do
            // ??? `a -x> b -es> cs` can become `a -xs> cs`
            // ??? `as -es> b -x> c` can become `as -xs> c`
        }
        if !changed {
            return changed;
        }

        let mut new_states = 0;
        let state_map: Vec<usize> = dead_states
            .into_iter()
            .scan(0, |s, dead| {
                if dead {
                    return Some(usize::MAX);
                } else {
                    let out = *s;
                    *s += 1;
                    new_states = *s;
                    return Some(out);
                }
            })
            .collect();
        self.states = new_states;

        for t in &mut self.transitions {
            t.from = state_map[t.from];
            t.to = state_map[t.to];
        }
        for t in &mut self.epsilons {
            t.from = state_map[t.from];
            t.to = state_map[t.to];
        }

        return changed;
    }

    /// Removes all nodes that cannot be reached or cannot reach the end.
    ///
    /// Ignores special epsilon types (so should be called after they have been resolved)
    fn remove_unreachable(&mut self) {
        let mut reach_start = vec![false; self.states];
        reach_start[0] = true;
        let mut reach_end = vec![false; self.states];
        reach_end[self.states - 1] = true;

        let mut changed = false;
        loop {
            for e in self.epsilons.iter() {
                if reach_start[e.from] && !reach_start[e.to] {
                    reach_start[e.to] = true;
                    changed = true;
                }
                if !reach_end[e.from] && reach_end[e.to] {
                    reach_end[e.from] = true;
                    changed = true;
                }
            }
            for t in self.transitions.iter() {
                if reach_start[t.from] && !reach_start[t.to] {
                    reach_start[t.to] = true;
                    changed = true;
                }
                if !reach_end[t.from] && reach_end[t.to] {
                    reach_end[t.from] = true;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
            changed = false;
        }

        // Remove transitions that involve redundant states
        self.transitions = self
            .transitions
            .iter()
            .filter(|t| reach_start[t.from] && reach_end[t.to])
            .cloned()
            .collect();
        self.epsilons = self
            .epsilons
            .iter()
            .filter(|e| reach_start[e.from] && reach_end[e.to])
            .cloned()
            .collect();

        // Then remove the states
        let mut new_states = 0;
        let state_map: Vec<usize> = std::iter::zip(reach_start.into_iter(), reach_end.into_iter())
            .map(|(a, b)| !a || !b)
            .scan(0, |s, dead| {
                if dead {
                    return Some(usize::MAX);
                } else {
                    let out = *s;
                    *s += 1;
                    new_states = *s;
                    return Some(out);
                }
            })
            .collect();
        self.states = new_states;

        for t in &mut self.transitions {
            t.from = state_map[t.from];
            t.to = state_map[t.to];
        }
        for t in &mut self.epsilons {
            t.from = state_map[t.from];
            t.to = state_map[t.to];
        }
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

        text_parts.push("\\node[state, initial](q0){$q_0$};\n".into());
        for i in 1..self.states - 1 {
            text_parts.push(format!(
                "\\node[state, right of=q{}](q{}){{$q_{{{}}}$}};\n",
                i - 1,
                i,
                i,
            ));
        }
        text_parts.push(format!(
            "\\node[state, accepting, right of=q{}](q{}){{$q_{{{}}}$}};\n",
            self.states - 2,
            self.states - 1,
            self.states - 1,
        ));

        for WorkingTransition { from, to, symbol } in &self.transitions {
            let bend = match to.cmp(from) {
                std::cmp::Ordering::Less => "[bend left] ",
                std::cmp::Ordering::Equal => "[loop below]",
                std::cmp::Ordering::Greater => "[bend left] ",
            };
            text_parts.push(format!(
                "\\path[->] (q{from}) edge {bend} node {{{}}} (q{to});\n",
                escape_latex(symbol.to_string()),
            ));
        }
        for WorkingEpsilonTransition { from, to, .. } in &self.epsilons {
            let bend = match to.cmp(from) {
                std::cmp::Ordering::Less => "[bend left] ",
                std::cmp::Ordering::Equal => "[loop below]",
                std::cmp::Ordering::Greater => "[bend left] ",
            };
            text_parts.push(format!(
                "\\path[->] (q{from}) edge {bend} node {{$\\epsilon$}} (q{to});\n"
            ));
        }

        text_parts.push("\\end{tikzpicture}\n".into());
        if include_doc {
            text_parts.push("\\end{document}\n".into());
        }
        return text_parts.into_iter().collect();
    }

    /// Using the classical NFA algorithm to do a simple boolean test on a string.
    pub fn test(&self, text: &str) -> bool {
        let mut list = vec![false; self.states];
        let mut new_list = vec![false; self.states];
        list[0] = true;

        // Adds all states reachable by epsilon transitions
        let propogate_epsilon = |list: &mut Vec<bool>| loop {
            let mut has_new = false;
            for t in &self.epsilons {
                if list[t.from] && !list[t.to] {
                    list[t.to] = true;
                    has_new = true;
                }
            }
            if !has_new {
                break;
            }
        };
        propogate_epsilon(&mut list);

        for c in text.chars() {
            for WorkingTransition { from, to, symbol } in &self.transitions {
                if list[*from] && symbol.check(c) {
                    new_list[*to] = true;
                }
            }
            let tmp = list;
            list = new_list;
            new_list = tmp;
            new_list.fill(false);
            propogate_epsilon(&mut list);
        }
        return *list.last().unwrap_or(&false);
    }
}
impl std::fmt::Display for WorkingNFA {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "{} states:", self.states)?;
        for t in &self.transitions {
            writeln!(f, "  {t}")?;
        }
        return Ok(());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse_tree::ERE;

    #[test]
    fn abbc_raw() {
        let nfa = WorkingNFA {
            transitions: vec![
                WorkingTransition::new(0, 1, 'a'.into()),
                WorkingTransition::new(1, 2, 'b'.into()),
                WorkingTransition::new(2, 3, 'c'.into()),
            ],
            epsilons: vec![WorkingEpsilonTransition::new(2, 1)],
            states: 4,
        };
        // println!("{}", nfa.to_tikz(true));

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
        let tree = SimplifiedTreeNode::from(ere);
        let nfa = WorkingNFA::new(&tree);
        // println!("{}", nfa.to_tikz(true));

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
        let tree = SimplifiedTreeNode::from(ere);
        let nfa = WorkingNFA::new(&tree);
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
        let tree = SimplifiedTreeNode::from(ere);
        let nfa = WorkingNFA::new(&tree);
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
        let tree = SimplifiedTreeNode::from(ere);
        let nfa = WorkingNFA::new(&tree);
        // println!("{}", nfa.to_tikz(true));

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
}
