use std::collections::BTreeMap;
use std::fmt::{Debug, Formatter};

use anyhow::{bail, Result};

type TokenId = u32;
type Token = Vec<u8>;
type Label = Token;
pub type Prob = u64;

pub trait Trie {
    type Node;
    fn lookup(&self, token: &TokenId) -> Option<&Self::Node>;
    fn update(&mut self, probabilities: &[(TokenId, Prob)]);
    fn tokens(&self) -> Vec<TokenId>;
    fn probabilities(&self) -> Vec<Prob>;
    fn distribution(&self) -> Distribution;
}

pub struct TokenTrie {
    root: TrieNode,
    // TODO implement thread-safe lookup
    //lookup: HashMap<TokenId, Weak<RefCell<TrieNode>>>,
}

impl TokenTrie {
    /// Constructor for TokenTrie
    pub fn new(tokens: Vec<(TokenId, Token)>) -> Result<Self> {
        let root = TrieNode::new(None, None);
        let mut trie = TokenTrie { root };
        for (token_id, token) in tokens {
            trie.root.insert(token.clone(), token.clone(), token_id)?;
        }
        Ok(trie)
    }
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

impl Distribution {
    pub(crate) fn find_token_prefix_index(
        &self,
        get_token: impl Fn(usize) -> Vec<u8>,
        token: &[u8],
        special_token_ids: &[TokenId],
    ) -> Option<usize> {
        for (idx, t) in self
            .reprs
            .iter()
            .enumerate()
            .filter(|(_, id)| !special_token_ids.contains(id))
        {
            let t = get_token(*t as usize);
            if token.starts_with(&t) {
                return Some(idx);
            }
        }
        None
    }
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

    /// Update this trie with new token probabilities
    fn update(&mut self, probabilities: &[(TokenId, Prob)]) {
        // update all probabilities according to arg
        self.root.update(probabilities)
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
                if p > 0 {
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
        let mut ret = self
            .probability
            .map_or(vec![], |p| if p > 0 { vec![p] } else { vec![] });
        for edge in self.edges.values() {
            ret.extend(edge.probabilities());
        }
        ret
    }
    pub fn distribution(&self) -> Distribution {
        match (self.token_id, self.probability) {
            (None, _) | (_, None) | (_, Some(0)) => {
                // the current node is a split node. Bubble down
                let mut distr = Distribution::default();
                for edge in self.edges.values() {
                    distr.extend(edge.distribution());
                }
                distr
            }
            (Some(token_id), Some(prob)) => {
                assert!(prob > 0);
                return if self.edges.is_empty() {
                    Distribution {
                        reprs: vec![token_id.clone()],
                        tokens: vec![vec![token_id.clone()]],
                        probs: vec![prob],
                    }
                } else {
                    // the current node has a token ID assigned and child nodes -> not uniquely decodable
                    let probs = vec![self.probabilities().iter().sum()];
                    if probs.iter().any(|p| *p > 0 as Prob) {
                        Distribution {
                            reprs: vec![token_id],
                            tokens: vec![self.tokens()],
                            probs,
                        }
                    } else {
                        Default::default()
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
                    //debug!("Found node for label {:?}", &label);
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
                    //debug!("Bubble down");
                    node.insert(
                        label[prefix.len()..].to_vec(),
                        token,
                        token_id,
                        //lookup_update,
                    )?;
                    Ok(())
                } else {
                    // label != edge && edge != prefix => label or edge (or both) have a distinct suffix: insert split
                    /*debug!(
                        "Insert split node for {:?} -> {} and {:?}",
                        label[prefix.len()..].to_vec(),
                        token_id,
                        edge[prefix.len()..].to_vec()
                    );*/
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
                //debug!("Insert new edge {:?}->{:?}", &label, &token_id);
                let t = TrieNode::new(Some(token), Some(token_id));
                //lookup_update(token_id, &mut t);
                self.edges.insert(label, t);
                Ok(())
            }
        };
    }
    fn update(&mut self, probabilities: &[(TokenId, Prob)]) {
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

#[derive(Copy, Clone)]
enum TrieFormatFilter {
    //All,
    WithProbOnly,
}

impl TrieFormatFilter {
    fn filter(&self, node: &TrieNode) -> bool {
        return match self {
            //TrieFormatFilter::All => true,
            TrieFormatFilter::WithProbOnly => node.probabilities().iter().any(|p| *p > 0),
        };
    }
}

impl TokenTrie {
    fn format_debug(
        &self,
        self_label: &Label,
        level: usize,
        max_depth: Option<usize>,
        filter: TrieFormatFilter,
    ) -> String {
        format!(
            "{}",
            self.root.format_debug(self_label, level, max_depth, filter)
        )
    }
}

impl TrieNode {
    fn format_debug(
        &self,
        self_label: &Label,
        level: usize,
        max_depth: Option<usize>,
        filter: TrieFormatFilter,
    ) -> String {
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
        let mut tr = if level > 0 {
            let indent = "  ".repeat(level - 1);
            format!("\n{}├─", indent)
        } else {
            String::new()
        };
        tr += label.as_str();
        let depth = self.depth();
        if let Some(max_depth) = max_depth {
            if level > max_depth && depth > 0 {
                tr += &format!("... ({} levels)", depth);
                return tr;
            }
        }
        for (label, n) in &self.edges {
            let child_incl = filter.filter(n);
            if child_incl {
                tr += &n.format_debug(label, level + 1, max_depth, filter);
            }
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
        const MAX_DEPTH: Option<usize> = None;
        const FILTER: TrieFormatFilter = TrieFormatFilter::WithProbOnly;
        write!(f, "{}", self.format_debug(&vec![], 0, MAX_DEPTH, FILTER))
    }
}

#[cfg(test)]
mod tests {
    use anyhow::{Context, Result};
    use log::{debug, info};

    use crate::token_trie::{Distribution, Prob, Token, TokenId, TokenTrie, Trie};

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

    fn create_probs<P: Copy + Into<Prob>>(
        tokens: &[(TokenId, Token)],
        probs: Vec<P>,
    ) -> Vec<(TokenId, Prob)> {
        assert!(tokens.len() >= probs.len()); // test impl error, not test failure!
        let probs: Vec<Prob> = probs.iter().map(|x| <P as Into<Prob>>::into(*x)).collect();
        tokens.iter().map(|x| x.0).zip(probs).collect()
    }

    #[test]
    fn test_webex() -> Result<()> {
        init();
        let args = create_tokens(vec!["Alice", "found", "an", "ant", "at", "the", "tree"]);
        let probs = create_probs(&args, vec![30u32, 10, 20, 10, 5, 5, 20]);
        let mut trie = TokenTrie::new(args)?;
        println!("{:?}", trie);
        trie.update(&probs);
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
        let probs = create_probs(&tokens, vec![1u8, 1]);
        let mut trie = TokenTrie::new(tokens).with_context(|| "TokenTrie constructor failed")?;
        trie.update(&probs);

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
        let probs = create_probs(&args, vec![1u8, 1]);
        trie.update(&probs);
        assert_eq!(trie.tokens(), vec![0, 1]);
        println!("{:?}", trie);
        Ok(())
    }

    #[test]
    fn test_root_preserve() -> Result<()> {
        init();
        let tokens = create_tokens(vec!["a"]);
        let probs = create_probs(&tokens, vec![1u8]);
        let mut trie = TokenTrie::new(tokens).with_context(|| "TokenTrie constructor failed")?;
        trie.update(&probs);

        assert_eq!(trie.tokens(), vec![0]);
        Ok(())
    }
    #[test]
    fn test_resample() -> Result<()> {
        init();
        let args = create_tokens(vec!["Alice", "an", "ant"]);
        let probs = create_probs(&args, vec![1u8, 2, 5]);
        let mut trie = TokenTrie::new(args)?;
        trie.update(&probs);

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
        init();
        let args = create_tokens(vec!["a", "alice", "an", "ant", "bob"]);
        let probs = create_probs(&args, vec![1u8, 3, 3, 4, 5]);
        let mut trie = TokenTrie::new(args)?;
        trie.update(&probs);

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
        init();
        let args = create_tokens(vec!["alice", "an", "albert", "ant", "bob", "a"]);
        let probs = create_probs(&args, vec![3u8, 3, 4, 5, 7, 1]);
        let mut trie = TokenTrie::new(args)?;
        trie.update(&probs);

        let d = trie.distribution();
        let reprs = d.reprs;
        let tokens = d.tokens;
        let probs = d.probs;

        assert_eq!(reprs[0], 5);
        assert_eq!(tokens[0], vec![5, 2, 0, 1, 3]);
        assert_eq!(probs[0], 16);

        let lookup_node = trie.lookup(&reprs[0]).unwrap();
        assert_eq!(tokens[0], lookup_node.tokens());
        assert_eq!(probs[0], lookup_node.probabilities().iter().sum::<Prob>());

        let d = lookup_node.distribution();
        let reprs = d.reprs;
        let tokens = d.tokens;
        let probs = d.probs;

        assert_eq!(tokens[0], vec![5, 2, 0, 1, 3]);
        assert_eq!(reprs[0], 5);
        assert_eq!(probs[0], 16);

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
        init();
        let tokens = create_tokens(vec!["alice", "an", "albert", "ant", "bob"]);
        let probs = create_probs(&tokens, vec![3u8, 3, 4, 5, 7]);
        let mut trie = TokenTrie::new(tokens)?;
        trie.update(&probs);

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
        init();
        let tokens = create_tokens(vec!["ABC", "AB"]);
        let probs = create_probs(&tokens, vec![2u8, 3]);
        let mut trie = TokenTrie::new(tokens)?;
        trie.update(&probs);

        let lookup_node = trie.lookup(&1).unwrap();
        assert_eq!(lookup_node.token_id, Some(1));
        assert_eq!(lookup_node.probability, Some(3 as Prob));
        assert!(!lookup_node.edges.contains_key(&vec![]));

        Ok(())
    }

    #[test]
    fn test_from_labels() -> Result<()> {
        init();
        let tokens = create_tokens(vec!["Alice", "an", "ant", "a"]);
        let probs = create_probs(&tokens, vec![1u8, 4, 7, 22]);
        let mut trie = TokenTrie::new(tokens)?;
        trie.update(&probs);
        println!("{:?}", trie);

        let d = trie.distribution();

        println!("{:?}", d);
        assert_eq!(d.reprs, vec![0, 3]);
        assert_eq!(d.tokens, vec![vec![0], vec![3, 1, 2]]);
        assert_eq!(d.probs, vec![1, 33]);

        let lookup_node = trie.lookup(&3).unwrap();
        let d = lookup_node.distribution();

        println!("{:?}", &d);

        assert_eq!(d.reprs, vec![3]);
        assert_eq!(d.tokens, vec![vec![3, 1, 2]]);
        assert_eq!(d.probs, vec![33]);

        Ok(())
    }

    #[test]
    fn test_zero_prob_repr() -> Result<()> {
        init();
        let tokens = create_tokens(vec!["Alice", "an", "at", "a"]);
        let mut trie = TokenTrie::new(tokens.clone())?;
        let probs = create_probs(&tokens, vec![1u8, 4, 2, 0]);
        trie.update(&probs);

        println!("{:?}", trie);

        let d = trie.distribution();
        let reprs = d.reprs;
        let tokens = d.tokens;
        let probs = d.probs;

        assert_eq!(tokens, vec![vec![0], vec![1], vec![2]]);
        assert_eq!(probs, vec![1u64, 4, 2]);
        assert_eq!(reprs, vec![0, 1, 2]);

        Ok(())
    }

    #[test]
    fn test_update_reset() -> Result<()> {
        init();
        let tokens = create_tokens(vec!["Alice", "an", "ant", "a"]);
        let mut trie = TokenTrie::new(tokens.clone())?;
        let probs = create_probs(&tokens, vec![1u8, 4, 7, 22]);
        trie.update(&probs);

        println!("{:?}", trie);

        let d = trie.distribution();
        let reprs = d.reprs;
        let tokens = d.tokens;
        let probs = d.probs;

        assert_eq!(reprs, vec![0, 3]);
        assert_eq!(tokens, vec![vec![0], vec![3, 1, 2]]);
        assert_eq!(probs, vec![1, 33]);

        trie.update(&vec![]);
        assert_eq!(
            trie.distribution(),
            Distribution {
                reprs: vec![],
                tokens: vec![],
                probs: vec![],
            }
        );

        trie.update(&vec![(2, 22)]);
        let d = trie.distribution();
        let reprs = d.reprs;
        let tokens = d.tokens;
        let probs = d.probs;

        println!("{:?}", trie);

        assert_eq!(reprs, vec![2]);
        assert_eq!(tokens, vec![vec![2]]);
        assert_eq!(probs, vec![22]);

        Ok(())
    }

    #[test]
    fn test_trie_order() -> Result<()> {
        init();
        let tokens = create_tokens(vec!["a", "b", "ab", "ba", "bab"]);
        let probs = create_probs(&tokens, vec![1u8, 2, 3, 4, 5]);
        let mut trie = TokenTrie::new(tokens)?;
        trie.update(&probs);

        assert_eq!(trie.tokens(), vec![0, 2, 1, 3, 4]);
        assert_eq!(trie.probabilities(), vec![1, 3, 2, 4, 5]);
        let trie = trie.lookup(&1).expect("Token not found");
        assert_eq!(trie.tokens(), vec![1, 3, 4]);
        assert_eq!(trie.probabilities(), vec![2, 4, 5]);
        Ok(())
    }

    #[test]
    fn test_trie_none_prob_token() -> Result<()> {
        init();
        let tokens = create_tokens(vec!["a", "b"]);
        let probs = create_probs(&tokens, vec![1u8]);
        let mut trie = TokenTrie::new(tokens)?;
        trie.update(&probs);

        println!("{:?}", trie);

        assert_eq!(trie.probabilities(), vec![1]);
        assert_eq!(trie.tokens(), vec![0]);
        Ok(())
    }

    #[test]
    fn test_trie_zero_prob_token() -> Result<()> {
        init();
        let tokens = create_tokens(vec!["a", "b"]);
        let probs = create_probs(&tokens, vec![1u8, 0]);
        let mut trie = TokenTrie::new(tokens)?;
        trie.update(&probs);

        println!("{:?}", trie);

        assert_eq!(trie.probabilities(), vec![1]);
        assert_eq!(trie.tokens(), vec![0]);
        Ok(())
    }

    #[test]
    fn test_distribution() -> Result<()> {
        init();
        let tokens: Vec<(TokenId, Token)> =
            create_tokens(vec!["Alice", "found", "an", "ant", "at", "the", "tree"]);
        let probs = create_probs(&tokens, [1u8; 7].to_vec());
        let mut trie = TokenTrie::new(tokens.clone())?;
        trie.update(&probs);

        debug!("{:?}", trie);

        let dist = trie.distribution();

        info!("{:?}", dist);
        assert_eq!(dist.reprs, vec![0, 2, 4, 1, 5, 6]);
        assert_eq!(dist.probs, vec![1, 2, 1, 1, 1, 1]);
        assert_eq!(
            dist.tokens,
            vec![vec![0], vec![2, 3], vec![4], vec![1], vec![5], vec![6]]
        );

        Ok(())
    }

    #[test]
    fn test_distribution_lookup() -> Result<()> {
        init();
        let tokens: Vec<(TokenId, Token)> =
            create_tokens(vec!["Alice", "found", "an", "ant", "at", "the", "tree"]);
        fn get_token(tokens: &[(TokenId, Token)]) -> impl Fn(usize) -> Vec<u8> + '_ {
            |idx| tokens[idx].1.clone()
        }
        let get_token = get_token(&tokens);
        let probs = create_probs(&tokens, [1u8; 7].to_vec());
        let mut trie = TokenTrie::new(tokens.clone())?;
        trie.update(&probs);

        debug!("{:?}", trie);

        let dist = trie.distribution();

        debug!("{:?}", dist);

        assert_eq!(
            dist.find_token_prefix_index(&get_token, "ant".as_bytes(), &vec![])
                .map(|x| dist.reprs[x]),
            Some(2)
        );
        assert_eq!(
            dist.find_token_prefix_index(&get_token, "tree".as_bytes(), &vec![])
                .map(|x| dist.reprs[x]),
            Some(6)
        );
        assert_eq!(
            dist.find_token_prefix_index(&get_token, "Bob".as_bytes(), &vec![])
                .map(|x| dist.reprs[x]),
            None
        );
        Ok(())
    }

    #[test]
    fn test_distribution_special_tokens() -> Result<()> {
        init();
        let tokens: Vec<(TokenId, Token)> = create_tokens(vec![
            "Alice",
            "found",
            "an",
            "ant",
            "ant__SPECIAL", /* <-- special token */
            "at",
            "the",
            "tree",
            "__SPECIAL", /* <-- special token */
        ]);
        fn get_token(tokens: &[(TokenId, Token)]) -> impl Fn(usize) -> Vec<u8> + '_ {
            |idx| tokens[idx].1.clone()
        }
        let get_token = get_token(&tokens);
        let probs = create_probs(&tokens, [1u8; 7].to_vec());
        let mut trie = TokenTrie::new(tokens.clone())?;
        trie.update(&probs);

        debug!("{:?}", trie);

        let dist = trie.distribution();
        let special_tokens = vec![7];

        debug!("{:?}", dist);

        assert_eq!(
            dist.find_token_prefix_index(&get_token, "__SPECIAL".as_bytes(), &special_tokens)
                .map(|x| dist.reprs[x]),
            None
        );
        Ok(())
    }
}
