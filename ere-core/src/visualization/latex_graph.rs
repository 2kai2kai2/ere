pub fn escape_latex(text: String) -> String {
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

pub struct LatexGraphTransition {
    pub(crate) to: usize,
    /// The label is a valid latex-encoded string to be inserted at the label.
    pub(crate) label: String,
}

pub struct LatexGraphState {
    /// The label is a valid latex-encoded string to be inserted at the label.
    pub(crate) label: String,
    pub(crate) transitions: Vec<LatexGraphTransition>,
    pub(crate) initial: bool,
    pub(crate) accept: bool,
}

/// Used for tikz visualizations of NFA-like graphs
pub struct LatexGraph {
    pub(crate) states: Vec<LatexGraphState>,
}
impl LatexGraph {
    /// Writes a LaTeX TikZ representation to visualize the graph.
    ///
    /// If `include_doc` is `true`, will include the headers.
    /// Otherwise, you should include `\usepackage{tikz}` and `\usetikzlibrary{automata, positioning}`.
    pub fn to_tikz(&self, include_doc: bool) -> String {
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
            let mut modifiers = String::new();
            if state.initial {
                modifiers += ", initial";
            }
            if state.accept {
                modifiers += ", accepting"
            }
            if i == 0 {
                text_parts.push(format!("\\node[state{modifiers}](q0){{$q_0$}};\n"));
            } else {
                text_parts.push(format!(
                    "\\node[state{modifiers}, right of=q{}](q{i}){{$q_{{{}}}$}};\n",
                    i - 1,
                    state.label
                ));
            }

            for LatexGraphTransition { to, label } in &state.transitions {
                let bend = match to.cmp(&i) {
                    std::cmp::Ordering::Less => "[bend left] ",
                    std::cmp::Ordering::Equal => "[loop below]",
                    std::cmp::Ordering::Greater => "[bend left] ",
                };
                transition_parts.push(format!(
                    "\\path[->] (q{i}) edge {bend} node {{{label}}} (q{to});\n",
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
}
