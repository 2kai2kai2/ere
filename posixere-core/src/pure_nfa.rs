/// This recognizes "true" regular expressions, meaning implicit anchors and no capture groups.
/// Capture groups are treated as simple concatenation.
use crate::parse_tree::Atom;
use crate::simplified_tree::SimplifiedTreeNode;

#[derive(thiserror::Error, Debug)]
enum PureNFAError {
    #[error("The PureNFA does not support anchors.")]
    UnexpectedAnchor,
}

#[derive(Debug, Clone, Copy)]
pub struct PureNFAEpsilonTransition {
    from: usize,
    to: usize,
}
impl PureNFAEpsilonTransition {
    pub fn new(from: usize, to: usize) -> PureNFAEpsilonTransition {
        return PureNFAEpsilonTransition { from, to };
    }
    pub fn with_offset(self, offset: usize) -> PureNFAEpsilonTransition {
        return PureNFAEpsilonTransition {
            from: self.from + offset,
            to: self.to + offset,
        };
    }
    pub fn add_offset(&self, offset: usize) -> PureNFAEpsilonTransition {
        return PureNFAEpsilonTransition {
            from: self.from + offset,
            to: self.to + offset,
        };
    }
}
impl std::fmt::Display for PureNFAEpsilonTransition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        return write!(f, "{} -> {}", self.from, self.to);
    }
}

#[derive(Debug, Clone)]
pub struct PureNFATransition {
    from: usize,
    to: usize,
    symbol: Atom,
}
impl PureNFATransition {
    pub fn new(from: usize, to: usize, symbol: Atom) -> PureNFATransition {
        return PureNFATransition { from, to, symbol };
    }
    pub fn with_offset(self, offset: usize) -> PureNFATransition {
        return PureNFATransition {
            from: self.from + offset,
            to: self.to + offset,
            symbol: self.symbol,
        };
    }
    pub fn add_offset(&self, offset: usize) -> PureNFATransition {
        return PureNFATransition {
            from: self.from + offset,
            to: self.to + offset,
            symbol: self.symbol.clone(),
        };
    }
}
impl std::fmt::Display for PureNFATransition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        return write!(f, "{} -({})> {}", self.from, self.symbol, self.to);
    }
}

/// Each NFA has one start state (`0`) and one accept state (`states - 1`)
#[derive(Debug)]
struct PureNFA {
    transitions: Vec<PureNFATransition>,
    epsilons: Vec<PureNFAEpsilonTransition>,
    states: usize,
}
impl PureNFA {
    /// Builds a very inefficient but valid NFA
    /// 
    /// Should be optimized using [`PureNFA::optimize_pass`]
    fn build_basic(tree: &SimplifiedTreeNode) -> Result<PureNFA, PureNFAError> {
        match tree {
            SimplifiedTreeNode::Empty => {
                return Ok(PureNFA {
                    transitions: Vec::new(),
                    epsilons: Vec::new(),
                    states: 1,
                });
            }
            SimplifiedTreeNode::Symbol(c) => {
                return Ok(PureNFA {
                    transitions: vec![PureNFATransition::new(0, 1, c.clone())],
                    epsilons: Vec::new(),
                    states: 2,
                });
            }
            SimplifiedTreeNode::Union(simplified_tree_nodes) => {
                let sub_nfas: Vec<PureNFA> = simplified_tree_nodes
                    .iter()
                    .map(PureNFA::build_basic)
                    .collect::<Result<_, _>>()?;
                let states = 2 + sub_nfas.iter().map(|n| n.states).sum::<usize>();
                let mut transitions = Vec::new();
                let mut epsilons = Vec::new();
                let mut used_states = 1usize;
                for nfa in sub_nfas {
                    // Epsilon transition in
                    epsilons.push(PureNFAEpsilonTransition::new(0, used_states));
                    // Copy internal transitions
                    transitions.extend(
                        nfa.transitions
                            .into_iter()
                            .map(|t| t.with_offset(used_states)),
                    );
                    epsilons.extend(nfa.epsilons.into_iter().map(|t| t.with_offset(used_states)));
                    // Epsilon transition out
                    epsilons.push(PureNFAEpsilonTransition::new(
                        used_states + nfa.states - 1,
                        states - 1,
                    ));
                    used_states += nfa.states;
                }
                assert_eq!(used_states + 1, states);

                return Ok(PureNFA {
                    transitions,
                    epsilons,
                    states,
                });
            }
            SimplifiedTreeNode::Capture(simplified_tree_node, _) => {
                return PureNFA::build_basic(&simplified_tree_node);
            }
            SimplifiedTreeNode::Concat(simplified_tree_nodes) => {
                if simplified_tree_nodes.is_empty() {
                    return Ok(PureNFA {
                        transitions: Vec::new(),
                        epsilons: Vec::new(),
                        states: 1,
                    });
                }

                let mut states = 1usize;
                let mut epsilons = vec![PureNFAEpsilonTransition::new(0, 1)];
                let mut transitions = vec![];

                for sub in simplified_tree_nodes {
                    let nfa = PureNFA::build_basic(sub)?;
                    transitions.extend(nfa.transitions.into_iter().map(|t| t.with_offset(states)));
                    epsilons.extend(nfa.epsilons.into_iter().map(|t| t.with_offset(states)));
                    states += nfa.states;
                    epsilons.push(PureNFAEpsilonTransition::new(states - 1, states));
                }

                return Ok(PureNFA {
                    transitions,
                    epsilons,
                    states: states + 1,
                });
            }
            SimplifiedTreeNode::Repeat(simplified_tree_node, times) => {
                let mut states = 1usize;
                let mut epsilons = vec![PureNFAEpsilonTransition::new(0, 1)];
                let mut transitions = vec![];
                let nfa = PureNFA::build_basic(simplified_tree_node)?;

                for _ in 0..*times {
                    transitions.extend(nfa.transitions.iter().map(|t| t.add_offset(states)));
                    epsilons.extend(nfa.epsilons.iter().map(|t| t.add_offset(states)));
                    states += nfa.states;
                    epsilons.push(PureNFAEpsilonTransition::new(states - 1, states));
                }

                return Ok(PureNFA {
                    transitions,
                    epsilons,
                    states: states + 1,
                });
            }
            SimplifiedTreeNode::UpTo(simplified_tree_node, times) => {
                let nfa = PureNFA::build_basic(simplified_tree_node)?;
                let states = 2 + nfa.states * *times;
                let mut used_states = 1usize;
                let mut transitions = Vec::new();
                let mut epsilons = vec![
                    PureNFAEpsilonTransition::new(0, 1),
                    PureNFAEpsilonTransition::new(0, states - 1),
                ];

                for _ in 0..*times {
                    transitions.extend(nfa.transitions.iter().map(|t| t.add_offset(states)));
                    epsilons.extend(nfa.epsilons.iter().map(|t| t.add_offset(states)));
                    used_states += nfa.states;
                    epsilons.push(PureNFAEpsilonTransition::new(used_states - 1, used_states));
                    if used_states != states - 1 {
                        epsilons.push(PureNFAEpsilonTransition::new(used_states - 1, states - 1));
                    }
                }

                return Ok(PureNFA {
                    transitions,
                    epsilons,
                    states,
                });
            }
            SimplifiedTreeNode::Star(simplified_tree_node) => {
                let nfa = PureNFA::build_basic(simplified_tree_node)?;
                let states = 2 + nfa.states;
                let transitions: Vec<_> = nfa
                    .transitions
                    .into_iter()
                    .map(|t| t.with_offset(1))
                    .collect();
                let mut epsilons: Vec<_> =
                    nfa.epsilons.into_iter().map(|t| t.with_offset(1)).collect();
                epsilons.push(PureNFAEpsilonTransition::new(0, 1));
                epsilons.push(PureNFAEpsilonTransition::new(states - 2, 0));
                epsilons.push(PureNFAEpsilonTransition::new(0, states - 1));
                return Ok(PureNFA {
                    transitions,
                    epsilons,
                    states,
                });
            }
            SimplifiedTreeNode::Start => return Err(PureNFAError::UnexpectedAnchor),
            SimplifiedTreeNode::End => return Err(PureNFAError::UnexpectedAnchor),
            SimplifiedTreeNode::Never => {
                return Ok(PureNFA {
                    transitions: Vec::new(),
                    epsilons: Vec::new(),
                    states: 2,
                });
            }
        }
    }
    pub fn new(tree: &SimplifiedTreeNode) -> Result<PureNFA, PureNFAError> {
        let mut nfa = PureNFA::build_basic(tree)?;
        while nfa.optimize_pass() {}
        return Ok(nfa);
    }
    /// Optimizes the NFA graph
    ///
    /// Returns `true` if changes were made (meaning another pass should be tried).
    fn optimize_pass(&mut self) -> bool {
        let mut changed = false;

        let mut dead_states = vec![false; self.states];

        // Skip redundant states
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
                // `as -xes> b -e> c` can become `as -xes> c`
                (incoming, incoming_eps, &[], &[outgoing_eps]) => {
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
                // `a -e> b -xes> cs` can become `a -xes> cs`
                (&[], &[incoming_eps], outgoing, outgoing_eps) => {
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

        for PureNFATransition { from, to, symbol } in &self.transitions {
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
        for PureNFAEpsilonTransition { from, to } in &self.epsilons {
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

    /// Using the classical NFA algorithm.
    pub fn check(&self, text: &str) -> bool {
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
            for PureNFATransition { from, to, symbol } in &self.transitions {
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
impl std::fmt::Display for PureNFA {
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
        let nfa = PureNFA {
            transitions: vec![
                PureNFATransition::new(0, 1, 'a'.into()),
                PureNFATransition::new(1, 2, 'b'.into()),
                PureNFATransition::new(2, 3, 'c'.into()),
            ],
            epsilons: vec![PureNFAEpsilonTransition::new(2, 1)],
            states: 4,
        };
        // println!("{}", nfa.to_tikz(true));

        assert!(nfa.check("abc"));
        assert!(nfa.check("abbc"));
        assert!(nfa.check("abbbc"));
        assert!(nfa.check("abbbbc"));

        assert!(!nfa.check("ac"));
        assert!(!nfa.check("abcc"));
        assert!(!nfa.check("bac"));
        assert!(!nfa.check("acb"));
    }

    #[test]
    fn phone_number() {
        let ere = ERE::parse_str(r"(\+1 )?[0-9]{3}-[0-9]{3}-[0-9]{4}").unwrap();
        let tree = SimplifiedTreeNode::from(ere);
        let nfa = PureNFA::new(&tree).unwrap();
        // println!("{}", nfa.to_tikz(true));

        assert!(nfa.check("012-345-6789"));
        assert!(nfa.check("987-654-3210"));
        assert!(nfa.check("+1 555-555-5555"));
        assert!(nfa.check("123-555-9876"));

        assert!(!nfa.check("abcd"));
        assert!(!nfa.check("0123456789"));
        assert!(!nfa.check("012--345-6789"));
        assert!(!nfa.check("(555) 555-5555"));
        assert!(!nfa.check("1 555-555-5555"));
    }
}
