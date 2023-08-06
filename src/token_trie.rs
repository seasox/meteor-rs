use log::debug;

use anyhow::{bail, Result};
use rand::distributions::{WeightedError, WeightedIndex};
use std::collections::BTreeMap;
use std::fmt::{Debug, Formatter};

type TokenId = u32;
type Token = Vec<u8>;
type Label = Token;
pub type Prob = f32;

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
            root: TrieNode::new(None, None),
            //lookup: Default::default(),
        }
    }
}

#[derive(Debug, PartialEq)]
pub struct Distribution {
    pub reprs: Vec<TokenId>,
    pub tokens: Vec<Vec<TokenId>>,
    pub probs: Vec<Prob>,
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

impl Distribution {
    pub fn weighted_index(&self) -> Result<WeightedIndex<Prob>, WeightedError> {
        WeightedIndex::new(&self.probs)
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
        let root = TrieNode::new(None, None);
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
    fn new(token: Option<Token>, token_id: Option<TokenId>) -> Self {
        Self {
            edges: EdgeMap::new(),
            token,
            token_id,
            probability: None,
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

    pub fn tokens(&self) -> Vec<TokenId> {
        let mut ret: Vec<TokenId> = match self.probability {
            None => vec![],
            Some(p) => {
                if p > 0.0 {
                    self.token_id.iter().cloned().collect()
                } else {
                    vec![]
                }
            }
        };
        for edge in self.edges.values() {
            ret.extend(edge.tokens());
        }
        ret
    }
    pub fn probabilities(&self) -> Vec<Prob> {
        let mut ret: Vec<Prob> = self.probability.iter().cloned().collect();
        for edge in self.edges.values() {
            ret.extend(edge.probabilities());
        }
        ret
    }
    pub fn distribution(&self) -> Distribution {
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
                    debug!("Found node for label {:?}", &label);
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
                    let mut split_node = TrieNode::new(None, None);
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
                            let token_node = TrieNode::new(Some(token), Some(token_id));
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
                let t = TrieNode::new(Some(token), Some(token_id));
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

    fn create_probs(tokens: &Vec<(TokenId, Token)>, probs: Vec<i32>) -> Vec<(TokenId, Prob)> {
        assert_eq!(tokens.len(), probs.len()); // test impl error, not test failure!
        let probs: Vec<Prob> = probs.iter().map(|x| *x as Prob).collect();
        tokens.iter().map(|x| x.0).zip(probs).collect()
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
        let tokens = create_tokens(vec!["a", "abde"]);
        let mut trie = TokenTrie::new(tokens).with_context(|| "TokenTrie constructor failed")?;
        trie.update(vec![(0, 0.5), (1, 0.5)]);

        assert_eq!(trie.tokens(), vec![0, 1]);
        println!("{:?}", trie);
        Ok(())
    }

    #[test]
    fn test_new_overlapping() -> Result<()> {
        init();
        let args = create_tokens(vec!["abcd", "abce"]);
        let mut trie =
            TokenTrie::new(args.clone()).with_context(|| "TokenTrie constructor failed")?;
        let probs = create_probs(&args, vec![1, 1]);
        trie.update(probs);
        assert_eq!(trie.tokens(), vec![0, 1]);
        println!("{:?}", trie);
        Ok(())
    }

    #[test]
    fn test_root_preserve() -> Result<()> {
        init();
        let tokens = create_tokens(vec!["a"]);
        let mut trie = TokenTrie::new(tokens).with_context(|| "TokenTrie constructor failed")?;
        trie.update(vec![(0, 1.0)]);

        assert_eq!(trie.tokens(), vec![0]);
        Ok(())
    }
    #[test]
    fn test_resample() -> Result<()> {
        let args = create_tokens(vec!["Alice", "an", "ant"]);
        let probs = create_probs(&args, vec![1, 2, 5]);
        let mut trie = TokenTrie::new(args)?;
        trie.update(probs);

        let d = trie.distribution();
        let reprs = d.reprs;
        let tokens = d.tokens;
        let probs = d.probs;

        assert_eq!(reprs[1], 1);
        assert_eq!(tokens[1], vec![1, 2]);
        assert_eq!(probs[1], 7 as Prob);

        let lookup_node = trie.lookup(&reprs[1]).unwrap();
        assert_eq!(tokens[1], lookup_node.tokens());
        assert_eq!(probs[1], lookup_node.probabilities().iter().sum::<Prob>());

        Ok(())
    }

    #[test]
    fn test_resample_deep() -> Result<()> {
        let args = create_tokens(vec!["a", "alice", "an", "ant", "bob"]);
        let probs = create_probs(&args, vec![1, 3, 3, 4, 5]);
        let mut trie = TokenTrie::new(args)?;
        trie.update(probs);

        let d = trie.distribution();
        let reprs = d.reprs;
        let tokens = d.tokens;
        let probs = d.probs;

        assert_eq!(reprs[0], 0);
        assert_eq!(tokens[0], vec![0, 1, 2, 3]);
        assert_eq!(probs[0], 11 as Prob);

        Ok(())
    }

    #[test]
    fn test_resample_multi_split() -> Result<()> {
        let args = create_tokens(vec!["alice", "an", "albert", "ant", "bob", "a"]);
        let probs = create_probs(&args, vec![3, 3, 4, 5, 7, 1]);
        let mut trie = TokenTrie::new(args)?;
        trie.update(probs);

        let d = trie.distribution();
        let reprs = d.reprs;
        let tokens = d.tokens;
        let probs = d.probs;

        assert_eq!(reprs[0], 5);
        assert_eq!(tokens[0], vec![5, 2, 0, 1, 3]);
        assert_eq!(probs[0], 16 as Prob);

        let lookup_node = trie.lookup(&reprs[0]).unwrap();
        assert_eq!(tokens[0], lookup_node.tokens());
        assert_eq!(probs[0], lookup_node.probabilities().iter().sum::<Prob>());

        let d = lookup_node.distribution();
        let reprs = d.reprs;
        let tokens = d.tokens;
        let probs = d.probs;

        assert_eq!(tokens[0], vec![5, 2, 0, 1, 3]);
        assert_eq!(reprs[0], 5);
        assert_eq!(probs[0], 16 as Prob);

        let st_lookup_node = trie.lookup(&reprs[0]).unwrap();
        assert_eq!(tokens[0], st_lookup_node.tokens());
        assert_eq!(
            probs[0],
            st_lookup_node.probabilities().iter().sum::<Prob>()
        );

        Ok(())
    }

    #[test]
    fn test_resample_multi_split_pseudo() -> Result<()> {
        let tokens = create_tokens(vec!["alice", "an", "albert", "ant", "bob"]);
        let probs = create_probs(&tokens, vec![3, 3, 4, 5, 7]);
        let mut trie = TokenTrie::new(tokens)?;
        trie.update(probs);

        println!("{:?}", trie);

        let d = trie.distribution();

        println!("{:?}", d);

        assert_eq!(d.reprs[2], 1);
        assert_eq!(d.tokens[2], vec![1, 3]);
        assert_eq!(d.probs[2], 8 as Prob);

        let lookup_node = trie.lookup(&d.reprs[2]);
        assert!(lookup_node.is_some());

        let d = lookup_node.unwrap().distribution();
        let reprs = d.reprs;
        let tokens = d.tokens;
        let probs = d.probs;

        assert_eq!(tokens[0], vec![1, 3]);
        assert_eq!(reprs[0], 1);
        assert_eq!(probs[0], 8 as Prob);
        assert_eq!(probs.len(), 1);
        assert_eq!(reprs.len(), 1);

        let st_lookup_node = trie.lookup(&reprs[0]);
        assert!(st_lookup_node.is_some());
        assert_eq!(
            probs[0],
            st_lookup_node.unwrap().probabilities().iter().sum::<Prob>()
        );

        Ok(())
    }

    #[test]
    fn test_empty_label_split() -> Result<()> {
        let tokens = create_tokens(vec!["ABC", "AB"]);
        let probs = create_probs(&tokens, vec![2, 3]);
        let mut trie = TokenTrie::new(tokens)?;
        trie.update(probs);

        let lookup_node = trie.lookup(&1).unwrap();
        assert_eq!(lookup_node.token_id, Some(1));
        assert_eq!(lookup_node.probability, Some(3 as Prob));
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
                tokens: vec![vec![], vec![]],
                probs: vec![0 as Prob, 0 as Prob],
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

    #[test]
    fn test_trie_order() -> Result<()> {
        let tokens = create_tokens(vec!["a", "b", "ab", "ba", "bab"]);
        let probs = create_probs(&tokens, vec![1, 2, 3, 4, 5]);
        let mut trie = TokenTrie::new(tokens)?;
        trie.update(probs);

        assert_eq!(trie.tokens(), vec![0, 2, 1, 3, 4]);
        assert_eq!(trie.probabilities(), vec![1f32, 3f32, 2f32, 4f32, 5f32]);
        let trie = trie.lookup(&1).expect("Token not found");
        assert_eq!(trie.tokens(), vec![1, 3, 4]);
        assert_eq!(trie.probabilities(), vec![2f32, 4f32, 5f32]);
        Ok(())
    }

    #[test]
    fn test_trie_zero_prob_token() -> Result<()> {
        let tokens = create_tokens(vec!["a", "b"]);
        let probs = vec![(tokens[0].0, 1.0)];
        let mut trie = TokenTrie::new(tokens)?;
        trie.update(probs);

        assert_eq!(trie.probabilities(), vec![1.0]);
        assert_eq!(trie.tokens(), vec![0]);
        Ok(())
    }
}
