use crate::parse_tree::*;

/// For translation between our parse tree and https://en.wikipedia.org/wiki/Thompson%27s_construction
#[derive(Debug, PartialEq, Eq, Clone)]
pub(crate) enum SimplifiedTreeNode {
    /// Translates to `epsilon`
    Empty,
    Symbol(Atom),
    Union(Vec<SimplifiedTreeNode>),
    /// `Capture(child, group_num)`
    Capture(Box<SimplifiedTreeNode>, usize),
    Concat(Vec<SimplifiedTreeNode>),
    /// `Repeat(child, times)`
    Repeat(Box<SimplifiedTreeNode>, usize),
    /// `UpTo(child, max_times)`
    UpTo(Box<SimplifiedTreeNode>, usize),
    Star(Box<SimplifiedTreeNode>),
    Start,
    End,
    Never,
}
impl SimplifiedTreeNode {
    pub fn optional(self) -> SimplifiedTreeNode {
        return self.union(SimplifiedTreeNode::Empty);
    }
    pub fn union(self, other: SimplifiedTreeNode) -> SimplifiedTreeNode {
        if let SimplifiedTreeNode::Union(mut u) = self {
            u.push(other);
            return SimplifiedTreeNode::Union(u);
        }
        return SimplifiedTreeNode::Union(vec![self, other]);
    }
    pub fn concat(self, other: SimplifiedTreeNode) -> SimplifiedTreeNode {
        if let SimplifiedTreeNode::Concat(mut c) = self {
            c.push(other);
            return SimplifiedTreeNode::Concat(c);
        }
        return SimplifiedTreeNode::Concat(vec![self, other]);
    }
    pub fn repeat(self, count: usize) -> SimplifiedTreeNode {
        return SimplifiedTreeNode::Repeat(self.into(), count);
    }
    pub fn upto(self, count: usize) -> SimplifiedTreeNode {
        return SimplifiedTreeNode::UpTo(self.into(), count);
    }
    pub fn star(self) -> SimplifiedTreeNode {
        return SimplifiedTreeNode::Star(self.into());
    }

    // /// A non-optimized, backtracking implementation
    // /// - `start` is the index of the original `text` it starts at
    // ///
    // /// # Returns
    // /// The number of matched symbols, or `None` if no match is found.
    // fn _check(&self, text: &str, start: usize, variation: usize) -> Option<usize> {
    //     return match self {
    //         SimplifiedTreeNode::Empty => Some(0),
    //         SimplifiedTreeNode::Symbol(atom) => atom.check(text.chars().next()?).then_some(1),
    //         SimplifiedTreeNode::Union(vec) => todo!(),
    //         SimplifiedTreeNode::Capture(node, _) => node._check(text, start),
    //         SimplifiedTreeNode::Concat(vec) => {
    //             let mut size = 0;
    //             for node in vec {
    //                 size += node._check(&text[size..], start + size)?;
    //             }
    //             Some(size)
    //         }
    //         SimplifiedTreeNode::Repeat(node, n) => {
    //             let mut size = 0;
    //             for _ in 0..(*n) {
    //                 size += node._check(&text[size..], start + size)?;
    //             }
    //             Some(size)
    //         }
    //         SimplifiedTreeNode::UpTo(node, n) => todo!(),
    //         SimplifiedTreeNode::Star(node) => todo!(),
    //         SimplifiedTreeNode::Start if start == 0 => Some(0),
    //         SimplifiedTreeNode::Start => None,
    //         SimplifiedTreeNode::End if start == text.len() => Some(0),
    //         SimplifiedTreeNode::End => None,
    //         SimplifiedTreeNode::Never => None,
    //     };
    // }
    // /// A non-optimized, backtracking implementation
    // pub fn check(&self, text: &str) -> bool {
    //     for i in 0..text.len() {
    //         if let Some(_) = self._check(&text[..i], i) {
    //             return true;
    //         }
    //     }
    //     return self._check("", text.len()).is_some();
    // }
}
impl SimplifiedTreeNode {
    fn from_ere(value: &ERE, mut group_num: usize) -> (SimplifiedTreeNode, usize) {
        let parts = value
            .0
            .iter()
            .map(|part| {
                let (new_node, new_group_num) =
                    SimplifiedTreeNode::from_ere_branch(&part, group_num);
                group_num = new_group_num;
                new_node
            })
            .collect();
        return (SimplifiedTreeNode::Union(parts), group_num);
    }
    fn from_ere_branch(value: &EREBranch, mut group_num: usize) -> (SimplifiedTreeNode, usize) {
        let parts = value
            .0
            .iter()
            .map(|part| {
                let (new_node, new_group_num) = SimplifiedTreeNode::from_ere_part(&part, group_num);
                group_num = new_group_num;
                new_node
            })
            .collect();
        return (SimplifiedTreeNode::Concat(parts), group_num);
    }
    fn from_ere_part(value: &EREPart, group_num: usize) -> (SimplifiedTreeNode, usize) {
        return match value {
            EREPart::Single(expr) => SimplifiedTreeNode::from_ere_expression(expr, group_num),
            EREPart::Quantified(expr, quantifier) => {
                let (child, group_num) = SimplifiedTreeNode::from_ere_expression(expr, group_num);
                let part = match quantifier {
                    Quantifier::Star => child.star(),
                    Quantifier::Plus => child.clone().concat(child.star()),
                    Quantifier::QuestionMark => child.optional(),
                    Quantifier::Multiple(n) => child.repeat(*n as usize),
                    Quantifier::Range(n, None) => {
                        child.clone().repeat(*n as usize).concat(child.star())
                    }
                    Quantifier::Range(n, Some(m)) => match m.checked_sub(*n) {
                        None => SimplifiedTreeNode::Never,
                        Some(0) => child.repeat(*n as usize),
                        Some(r) => child
                            .clone()
                            .repeat(*n as usize)
                            .concat(child.upto(r as usize)),
                    },
                };
                (part, group_num)
            }
            EREPart::Start => (SimplifiedTreeNode::Start, group_num),
            EREPart::End => (SimplifiedTreeNode::End, group_num),
        };
    }
    fn from_ere_expression(value: &EREExpression, group_num: usize) -> (SimplifiedTreeNode, usize) {
        return match value {
            EREExpression::Atom(atom) => (atom.clone().into(), group_num),
            EREExpression::Subexpression(ere) => {
                let (capture, next_group_num) = SimplifiedTreeNode::from_ere(ere, group_num + 1);
                (
                    SimplifiedTreeNode::Capture(capture.into(), group_num),
                    next_group_num,
                )
            }
        };
    }
}
impl From<ERE> for SimplifiedTreeNode {
    fn from(value: ERE) -> Self {
        let (root, _) = SimplifiedTreeNode::from_ere(&value, 0);
        return root;
    }
}
impl From<Atom> for SimplifiedTreeNode {
    fn from(value: Atom) -> Self {
        return SimplifiedTreeNode::Symbol(value);
    }
}
