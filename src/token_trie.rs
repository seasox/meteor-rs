use log::debug;

use anyhow::{bail, Result};
use rand::distributions::WeightedIndex;
use std::collections::BTreeMap;
use std::fmt::{Debug, Formatter};

type TokenId = u32;
type Token = Vec<u8>;
type Label = Token;
type Prob = f64;

pub struct TokenTrie {
    root: TrieNode,
    // TODO implement thread-safe lookup
    //lookup: HashMap<TokenId, Weak<RefCell<TrieNode>>>,
}

pub trait Trie {
    type Node;
    fn lookup(&self, token: &TokenId) -> Option<&Self::Node>;
    fn new(tokens: Vec<(TokenId, Token)>) -> Result<Self>
    where
        Self: Sized;
    fn update(&mut self, probabilities: Vec<(TokenId, Prob)>);
    fn tokens(&self) -> Vec<TokenId>;
    fn probabilities(&self) -> Vec<Prob>;
    fn distribution(&self) -> Distribution;
}

impl Default for TokenTrie {
    fn default() -> Self {
        TokenTrie {
            root: TrieNode::new(None, None, None),
            //lookup: Default::default(),
        }
    }
}

#[derive(Debug, PartialEq)]
pub struct Distribution {
    reprs: Vec<TokenId>,
    tokens: Vec<Vec<TokenId>>,
    probs: Vec<Prob>,
}

impl Default for Distribution {
    fn default() -> Self {
        Distribution {
            reprs: vec![],
            tokens: vec![],
            probs: vec![],
        }
    }
}

impl Distribution {
    fn extend(&mut self, other: Distribution) {
        self.reprs.extend(other.reprs);
        self.tokens.extend(other.tokens);
        self.probs.extend(other.probs);
    }
}

impl Trie for TokenTrie {
    type Node = TrieNode;
    /// Look up a TokenId in this trie
    fn lookup(&self, token: &TokenId) -> Option<&TrieNode> {
        self.root.lookup(token)
    }

    /// Constructor for TokenTrie
    fn new(tokens: Vec<(TokenId, Token)>) -> Result<Self> {
        let root = TrieNode::new(None, None, None);
        let mut trie = TokenTrie { root };
        for (token_id, token) in tokens {
            trie.root.insert(token.clone(), token.clone(), token_id)?;
        }
        Ok(trie)
    }

    /// Update this trie with new token probabilities
    fn update(&mut self, probabilities: Vec<(TokenId, Prob)>) {
        // update all probabilities according to arg
        self.root.update(&probabilities)
    }

    fn tokens(&self) -> Vec<TokenId> {
        self.root.tokens()
    }

    fn probabilities(&self) -> Vec<Prob> {
        self.root.probabilities()
    }

    fn distribution(&self) -> Distribution {
        self.root.distribution()
    }
}

type EdgeMap<T, R> = BTreeMap<T, R>;

pub struct TrieNode {
    edges: EdgeMap<Label, TrieNode>,
    pub token: Option<Token>,
    pub token_id: Option<TokenId>,
    pub probability: Option<Prob>,
}

impl TrieNode {
    fn new(token: Option<Token>, token_id: Option<TokenId>, probability: Option<Prob>) -> Self {
        Self {
            edges: EdgeMap::new(),
            token,
            token_id,
            probability,
        }
    }
    fn lookup(&self, id: &TokenId) -> Option<&TrieNode> {
        if self.token_id.is_some_and(|x| x.eq(id)) {
            return Some(self);
        }
        self.edges
            .iter()
            .map(|(_, n)| n.lookup(id))
            .find(|n| n.is_some())
            .map(|n| n.unwrap())
    }

    fn tokens(&self) -> Vec<TokenId> {
        let mut ret: Vec<TokenId> = self.token_id.iter().cloned().collect();
        for edge in self.edges.values() {
            ret.extend(edge.tokens());
        }
        ret
    }
    fn probabilities(&self) -> Vec<Prob> {
        let mut ret: Vec<Prob> = self.probability.iter().cloned().collect();
        for edge in self.edges.values() {
            ret.extend(edge.probabilities());
        }
        ret
    }
    fn distribution(&self) -> Distribution {
        match self.token_id {
            None => {
                // the current node is a split node. Bubble down
                let mut distr = Distribution::default();
                for edge in self.edges.values() {
                    distr.extend(edge.distribution());
                }
                distr
            }
            Some(token_id) => {
                return if self.edges.is_empty() {
                    Distribution {
                        reprs: vec![token_id.clone()],
                        tokens: vec![vec![token_id.clone()]],
                        probs: vec![self.probability.iter().cloned().sum()],
                    }
                } else {
                    // the current node has a probability assigned and child nodes -> not uniquely decodable
                    let probs = vec![self.probabilities().iter().sum()];
                    Distribution {
                        reprs: vec![token_id],
                        tokens: vec![self.tokens()],
                        probs,
                    }
                };
            }
        }
    }
    fn insert(&mut self, label: Label, token: Token, token_id: TokenId) -> Result<()> {
        let child = self
            .edges
            .iter_mut()
            .find(|(e, _)| !find_common_prefix(e, &label).is_empty());
        return match child {
            Some((edge, node)) => {
                // this edge matches the label to insert. Update this node with token_id
                if label.eq(edge) {
                    debug!("Found node {:?} for label {:?}", node, &label);
                    if node.token.is_some() {
                        bail!("During insert of token {:?} with ID {:?}: Matched node already initialized with {:?}, ID {:?}", token, token_id, node.token, node.token_id);
                    }
                    node.token = Some(token.clone());
                    node.token_id = Some(token_id.clone());
                    //lookup_update(token_id, node.clone());
                    return Ok(());
                }
                let prefix = find_common_prefix(edge, &label);
                assert!(!prefix.is_empty());
                if edge.eq(&prefix) {
                    debug!("Bubble down");
                    node.insert(
                        label[prefix.len()..].to_vec(),
                        token,
                        token_id,
                        //lookup_update,
                    )?;
                    Ok(())
                } else {
                    // label != edge && edge != prefix => label or edge (or both) have a distinct suffix: insert split
                    debug!(
                        "Insert split node for {:?} -> {} and {:?}",
                        label[prefix.len()..].to_vec(),
                        token_id,
                        edge[prefix.len()..].to_vec()
                    );
                    let mut split_node = TrieNode::new(None, None, None);
                    {
                        let edge = edge.clone();
                        split_node.edges.insert(
                            edge[prefix.len()..].to_vec(),
                            self.edges.remove(&edge).unwrap(),
                        );
                        if label.eq(&prefix) {
                            // set token and id on split node
                            assert_eq!(split_node.token_id, None);
                            split_node.token = Some(token);
                            split_node.token_id = Some(token_id);
                            //lookup_update(token_id, &mut split_node);
                        } else {
                            // add child node for suffix of label
                            let token_node = TrieNode::new(Some(token), Some(token_id), None);
                            //lookup_update(token_id, &mut token_node);
                            split_node
                                .edges
                                .insert(label[prefix.len()..].to_vec(), token_node);
                        }
                    }
                    self.edges.insert(prefix, split_node);
                    Ok(())
                }
            }
            None => {
                debug!("Insert new edge {:?}->{:?}", &label, &token_id);
                let t = TrieNode::new(Some(token), Some(token_id), None);
                //lookup_update(token_id, &mut t);
                self.edges.insert(label, t);
                Ok(())
            }
        };
    }
    fn update(&mut self, probabilities: &Vec<(TokenId, Prob)>) {
        self.probability = match self.token_id {
            Some(id) => probabilities
                .iter()
                .find(|(t, _)| id.eq(t))
                .map(|(_, p)| *p),
            None => None,
        };
        self.edges
            .iter_mut()
            .for_each(|(_, n)| n.update(probabilities));
    }
}

impl TrieNode {
    fn format_debug(&self, self_label: &Label, level: usize, max_depth: Option<usize>) -> String {
        let mut label = format!("{:?}", self_label);
        if let Some(token_id) = &self.token_id {
            label += &format!(" ({:?}", token_id);
            if let Some(token) = &self.token {
                label += &format!(", {:?}", token);
            }
            if let Some(prob) = self.probability {
                label += &format!(", {:?}", prob);
            }
            label += &")".to_owned();
        }
        let indent = "  ".repeat(level);
        let mut tr = label;
        let depth = self.depth();
        if let Some(max_depth) = max_depth {
            if level > max_depth && depth > 0 {
                tr += &format!("\n{}├── ... ({} levels)", indent, depth);
                return tr;
            }
        }
        for (label, n) in &self.edges {
            tr += &format!(
                "\n{}├── {}",
                indent,
                n.format_debug(label, level + 1, max_depth)
            )
        }
        tr
    }

    fn depth(&self) -> usize {
        let mut max_child_depth = 0;
        for edge_node in self.edges.values() {
            let child_depth = edge_node.depth();
            if child_depth > max_child_depth {
                max_child_depth = child_depth;
            }
        }
        max_child_depth + 1
    }
}

fn find_common_prefix<T: PartialEq + Clone>(vec1: &[T], vec2: &[T]) -> Vec<T> {
    let min_length = std::cmp::min(vec1.len(), vec2.len());
    let mut common_prefix: Vec<T> = Vec::new();

    for i in 0..min_length {
        if vec1[i] == vec2[i] {
            common_prefix.push(vec1[i].clone());
        } else {
            break;
        }
    }
    common_prefix
}

impl Debug for TokenTrie {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.root)
    }
}

impl Debug for TrieNode {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        const MAX_DEPTH: Option<usize> = None;
        write!(f, "{}", self.format_debug(&vec![], 0, MAX_DEPTH))
    }
}

#[cfg(test)]
mod tests {
    use crate::token_trie::{Distribution, Prob, Token, TokenId, TokenTrie, Trie};
    use anyhow::{Context, Result};
    use std::collections::HashMap;

    fn init() {
        let _ = env_logger::builder().is_test(true).try_init();
    }

    fn create_tokens(tokens: Vec<&str>) -> Vec<(TokenId, Token)> {
        let mut ret = vec![];
        for i in 0..tokens.len() {
            ret.push((i as TokenId, tokens[i].as_bytes().to_vec()));
        }
        ret
    }

    #[test]
    fn test_webex() -> Result<()> {
        init();
        let args = create_tokens(vec!["Alice", "found", "an", "ant", "at", "the", "tree"]);
        let probs: HashMap<TokenId, Prob> = HashMap::from([
            (args[0].0.clone(), 0.3),
            (args[1].0.clone(), 0.1),
            (args[2].0.clone(), 0.2),
            (args[3].0.clone(), 0.1),
            (args[4].0.clone(), 0.05),
            (args[5].0.clone(), 0.05),
            (args[6].0.clone(), 0.2),
        ]);
        let mut trie = TokenTrie::new(args)?;
        println!("{:?}", trie);
        trie.update(probs.into_iter().collect());
        let d = trie.distribution();
        let reprs = d.reprs;
        let tokens = d.tokens;
        assert_eq!(reprs, [0, 2, 4, 1, 5, 6]);
        for i in 0..reprs.len() {
            if i == 1 {
                assert_eq!(tokens[i], [2, 3]);
            } else {
                assert!(tokens[i][0] == reprs[i] && tokens[i].len() == 1);
            }
        }
        Ok(())
    }

    #[test]
    fn test_new() -> Result<()> {
        init();
        let args: Vec<(TokenId, Token)> = vec![(0, vec![1, 2, 3, 4]), (1, vec![1, 5, 6, 7, 8])];
        let tokens: Vec<TokenId> = args.iter().map(|t| t.0.clone()).collect();
        let trie = TokenTrie::new(args).with_context(|| "TokenTrie constructor failed")?;
        assert_eq!(trie.tokens(), tokens);
        println!("{:?}", trie);
        Ok(())
    }

    #[test]
    fn test_new_overlapping() -> Result<()> {
        init();
        let args: Vec<(TokenId, Token)> = vec![(0, vec![1, 2, 3, 4]), (1, vec![1, 2, 3, 5])];
        let tokens: Vec<TokenId> = args.iter().map(|t| t.0.clone()).collect();
        let trie = TokenTrie::new(args).with_context(|| "TokenTrie constructor failed")?;
        assert_eq!(trie.tokens(), tokens);
        println!("{:?}", trie);
        Ok(())
    }

    #[test]
    fn test_root_preserve() -> Result<()> {
        init();
        let args: Vec<(TokenId, Token)> = vec![(0, vec![1])];
        let tokens: Vec<TokenId> = args.iter().map(|t| t.0.clone()).collect();
        let trie = TokenTrie::new(args).with_context(|| "TokenTrie constructor failed")?;
        assert_eq!(trie.tokens(), tokens);
        Ok(())
    }
    #[test]
    fn test_resample() -> Result<()> {
        let args = create_tokens(vec!["Alice", "an", "ant"]);
        let mut trie = TokenTrie::new(args)?;
        trie.update(vec![(0, 1f64), (1, 2f64), (2, 5f64)]);

        let d = trie.distribution();
        let reprs = d.reprs;
        let tokens = d.tokens;
        let probs = d.probs;

        assert_eq!(reprs[1], 1);
        assert_eq!(tokens[1], vec![1, 2]);
        assert_eq!(probs[1], 7f64);

        let lookup_node = trie.lookup(&reprs[1]).unwrap();
        assert_eq!(tokens[1], lookup_node.tokens());
        assert_eq!(probs[1], lookup_node.probabilities().iter().sum::<f64>());

        Ok(())
    }

    #[test]
    fn test_resample_deep() -> Result<()> {
        let args = create_tokens(vec!["a", "alice", "an", "ant", "bob"]);
        let mut trie = TokenTrie::new(args)?;
        trie.update(vec![(0, 1f64), (1, 3f64), (2, 3f64), (3, 4f64), (4, 5f64)]);

        let d = trie.distribution();
        let reprs = d.reprs;
        let tokens = d.tokens;
        let probs = d.probs;

        assert_eq!(reprs[0], 0);
        assert_eq!(tokens[0], vec![0, 1, 2, 3]);
        assert_eq!(probs[0], 11f64);

        Ok(())
    }

    #[test]
    fn test_resample_multi_split() -> Result<()> {
        let args = create_tokens(vec!["alice", "an", "albert", "ant", "bob", "a"]);
        let mut trie = TokenTrie::new(args)?;
        trie.update(vec![
            (0, 3f64),
            (1, 3f64),
            (2, 4f64),
            (3, 5f64),
            (4, 7f64),
            (5, 1f64),
        ]);

        let d = trie.distribution();
        let reprs = d.reprs;
        let tokens = d.tokens;
        let probs = d.probs;

        assert_eq!(reprs[0], 5);
        assert_eq!(tokens[0], vec![5, 2, 0, 1, 3]);
        assert_eq!(probs[0], 16f64);

        let lookup_node = trie.lookup(&reprs[0]).unwrap();
        assert_eq!(tokens[0], lookup_node.tokens());
        assert_eq!(probs[0], lookup_node.probabilities().iter().sum::<f64>());

        let d = lookup_node.distribution();
        let reprs = d.reprs;
        let tokens = d.tokens;
        let probs = d.probs;

        assert_eq!(tokens[0], vec![5, 2, 0, 1, 3]);
        assert_eq!(reprs[0], 5);
        assert_eq!(probs[0], 16f64);

        let st_lookup_node = trie.lookup(&reprs[0]).unwrap();
        assert_eq!(tokens[0], st_lookup_node.tokens());
        assert_eq!(probs[0], st_lookup_node.probabilities().iter().sum::<f64>());

        Ok(())
    }

    #[test]
    fn test_resample_multi_split_pseudo() -> Result<()> {
        let tokens = create_tokens(vec!["alice", "an", "albert", "ant", "bob"]);
        let mut trie = TokenTrie::new(tokens)?;
        trie.update(vec![(0, 3f64), (1, 3f64), (2, 4f64), (3, 5f64), (4, 7f64)]);

        println!("{:?}", trie);

        let d = trie.distribution();

        println!("{:?}", d);

        assert_eq!(d.reprs[2], 1);
        assert_eq!(d.tokens[2], vec![1, 3]);
        assert_eq!(d.probs[2], 8f64);

        let lookup_node = trie.lookup(&d.reprs[2]);
        assert!(lookup_node.is_some());

        let d = lookup_node.unwrap().distribution();
        let reprs = d.reprs;
        let tokens = d.tokens;
        let probs = d.probs;

        assert_eq!(tokens[0], vec![1, 3]);
        assert_eq!(reprs[0], 1);
        assert_eq!(probs[0], 8f64);
        assert_eq!(probs.len(), 1);
        assert_eq!(reprs.len(), 1);

        let st_lookup_node = trie.lookup(&reprs[0]);
        assert!(st_lookup_node.is_some());
        assert_eq!(
            probs[0],
            st_lookup_node.unwrap().probabilities().iter().sum::<f64>()
        );

        Ok(())
    }

    #[test]
    fn test_empty_label_split() -> Result<()> {
        let tokens = create_tokens(vec!["ABC", "AB"]);
        let mut trie = TokenTrie::new(tokens)?;
        let probs = vec![(0, 2f64), (1, 3f64)];
        trie.update(probs);

        let lookup_node = trie.lookup(&1).unwrap();
        assert_eq!(lookup_node.token_id, Some(1));
        assert_eq!(lookup_node.probability, Some(3f64));
        assert!(!lookup_node.edges.contains_key(&vec![]));

        Ok(())
    }

    #[test]
    fn test_from_labels() -> Result<()> {
        let tokens = create_tokens(vec!["Alice", "an", "ant", "a"]);
        let mut trie = TokenTrie::new(tokens)?;
        let probs = vec![(0, 1.0), (1, 4.0), (2, 7.0), (3, 22.0)];
        trie.update(probs);
        println!("{:?}", trie);

        let d = trie.distribution();

        println!("{:?}", d);
        assert_eq!(d.reprs, vec![0, 3]);
        assert_eq!(d.tokens, vec![vec![0], vec![3, 1, 2]]);
        assert_eq!(d.probs, vec![1.0, 33.0]);

        let lookup_node = trie.lookup(&3).unwrap();
        let d = lookup_node.distribution();

        println!("{:?}", &d);

        assert_eq!(d.reprs, vec![3]);
        assert_eq!(d.tokens, vec![vec![3, 1, 2]]);
        assert_eq!(d.probs, vec![33.0]);

        Ok(())
    }

    #[test]
    fn test_update_reset() -> Result<()> {
        let tokens = create_tokens(vec!["Alice", "an", "ant", "a"]);
        let mut trie = TokenTrie::new(tokens)?;
        let probs = vec![(0, 1.0), (1, 4.0), (2, 7.0), (3, 22.0)];
        trie.update(probs);

        println!("{:?}", trie);

        let d = trie.distribution();
        let reprs = d.reprs;
        let tokens = d.tokens;
        let probs = d.probs;

        assert_eq!(reprs, vec![0, 3]);
        assert_eq!(tokens, vec![vec![0], vec![3, 1, 2]]);
        assert_eq!(probs, vec![1.0, 33.0]);

        trie.update(vec![]);
        assert_eq!(
            trie.distribution(),
            Distribution {
                reprs: vec![0, 3],
                tokens: vec![vec![0], vec![3, 1, 2]],
                probs: vec![0f64, 0f64],
            }
        );

        trie.update(vec![(2, 22.0)]);
        let d = trie.distribution();
        let reprs = d.reprs;
        let tokens = d.tokens;
        let probs = d.probs;

        assert_eq!(reprs, vec![0, 3]);
        assert_eq!(tokens, vec![vec![0], vec![3, 1, 2]]);
        assert_eq!(probs, vec![0.0, 22.0]);

        Ok(())
    }
}
