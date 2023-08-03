use log::debug;

use anyhow::{bail, Context, Result};
use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap};
use std::fmt::{Debug, Formatter};
use std::rc::{Rc, Weak};

type TokenId = u32;
type Token = Vec<u8>;
type Label = Token;
type Prob = f64;

pub struct TokenTrie {
    root: Rc<RefCell<TrieNode>>,
    lookup: HashMap<TokenId, Weak<RefCell<TrieNode>>>,
}

trait TNode {}

pub trait Trie {
    type Node;
    fn lookup(&self, token: &TokenId) -> Result<Rc<RefCell<Self::Node>>>;
    fn new(tokens: Vec<(TokenId, Token)>) -> Result<Self>
    where
        Self: Sized;
    fn update(&self, probabilities: Vec<(TokenId, Prob)>) -> Result<()>;
    fn tokens(&self) -> Vec<TokenId>;
    fn probabilities(&self) -> Vec<Prob>;
    fn distribution(&self) -> Vec<(TokenId, Vec<TokenId>, Prob)>;
}

impl Trie for TokenTrie {
    type Node = TrieNode;
    /// Look up a TokenId in this trie
    fn lookup(&self, token: &TokenId) -> Result<Rc<RefCell<TrieNode>>> {
        let res = self
            .lookup
            .get(token)
            .with_context(|| format!("Not found: {:?}", token))?;
        res.upgrade().with_context(|| "Upgrade failed")
    }

    /// Constructor for TokenTrie
    fn new(tokens: Vec<(TokenId, Token)>) -> Result<Self> {
        let root = Rc::new(RefCell::new(TrieNode::new(None, None, None)));
        let mut lookup = HashMap::new();
        let mut lookup_update = |id, node: Rc<RefCell<TrieNode>>| {
            assert_eq!(Some(id), node.borrow().token_id);
            lookup.insert(id, Rc::downgrade(&node));
        };
        for (token_id, token) in tokens {
            root.borrow_mut()
                .insert(token.clone(), token, token_id, &mut lookup_update)?;
        }
        Ok(TokenTrie { root, lookup })
    }

    /// Update this trie with new token probabilities
    fn update(&self, probabilities: Vec<(TokenId, Prob)>) -> Result<()> {
        // clear all probabilities
        for (tok, t) in &self.lookup {
            t.upgrade()
                .with_context(|| format!("Reference for {:?} went out of scope", tok))?
                .borrow_mut()
                .probability = None;
        }
        // set probabilities according to parameters
        for (t, p) in probabilities {
            let t_rc = self
                .lookup
                .get(&t)
                .with_context(|| "Token not found")?
                .upgrade()
                .with_context(|| format!("Reference {:?} went out of scope", t))?;
            t_rc.borrow_mut().probability = Some(p);
        }
        Ok(())
    }

    fn tokens(&self) -> Vec<TokenId> {
        self.root.borrow().tokens()
    }

    fn probabilities(&self) -> Vec<Prob> {
        self.root.borrow().probabilities()
    }

    fn distribution(&self) -> Vec<(TokenId, Vec<TokenId>, Prob)> {
        self.root.borrow().distribution()
    }
}

type EdgeMap<T, R> = BTreeMap<T, R>;

pub struct TrieNode {
    edges: EdgeMap<Label, Rc<RefCell<TrieNode>>>,
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
    fn tokens(&self) -> Vec<TokenId> {
        let mut ret: Vec<TokenId> = self.token_id.iter().cloned().collect();
        for edge in self.edges.values() {
            ret.extend(edge.borrow().tokens());
        }
        ret
    }
    fn probabilities(&self) -> Vec<Prob> {
        let mut ret: Vec<Prob> = self.probability.iter().cloned().collect();
        for edge in self.edges.values() {
            ret.extend(edge.borrow().probabilities());
        }
        ret
    }
    fn distribution(&self) -> Vec<(TokenId, Vec<TokenId>, Prob)> {
        match self.token_id {
            None => {
                // the current node is a split node. Bubble down
                let mut distr = vec![];
                for edge in self.edges.values() {
                    distr.extend(edge.borrow().distribution());
                }
                distr
            }
            Some(token_id) => {
                return if self.edges.is_empty() {
                    vec![(
                        token_id.clone(),
                        vec![token_id.clone()],
                        self.probability.iter().cloned().sum(),
                    )]
                } else {
                    // the current node has a probability assigned and child nodes -> not uniquely decodable
                    let probs = self.probabilities().iter().sum();
                    vec![(token_id, self.tokens(), probs)]
                };
            }
        }
    }
    fn insert<LookupUpdateFn>(
        &mut self,
        label: Label,
        token: Token,
        token_id: TokenId,
        mut lookup_update: LookupUpdateFn,
    ) -> Result<()>
    where
        LookupUpdateFn: FnMut(TokenId, Rc<RefCell<TrieNode>>),
    {
        let edges = self.edges.clone();
        let child: Option<(&Label, &Rc<RefCell<TrieNode>>)> = edges
            .iter()
            .find(|(e, _)| !find_common_prefix(e, &label).is_empty())
            .clone();
        return match child {
            Some((edge, node)) => {
                // this edge matches the label to insert. Update this node with token_id
                if label.eq(edge) {
                    debug!("Found node {:?} for label {:?}", node.borrow(), &label);
                    if node.borrow().token.is_some() {
                        bail!("During insert of token {:?} with ID {:?}: Matched node already initialized with {:?}, ID {:?}", token, token_id, node.borrow().token, node.borrow().token_id);
                    }
                    let mut node_mut = node.borrow_mut();
                    node_mut.token = Some(token.clone());
                    node_mut.token_id = Some(token_id.clone());
                    drop(node_mut); // we have to drop manually here. Otherwise lookup_update panics
                    lookup_update(token_id, node.clone());
                    return Ok(());
                }
                let prefix = find_common_prefix(edge, &label);
                assert!(!prefix.is_empty());
                if edge.eq(&prefix) {
                    debug!("Bubble down");
                    node.borrow_mut().insert(
                        label[prefix.len()..].to_vec(),
                        token,
                        token_id,
                        lookup_update,
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
                    let split_node = Rc::new(RefCell::new(TrieNode::new(None, None, None)));
                    {
                        let mut split_node_ref = split_node.borrow_mut();
                        split_node_ref.edges.insert(
                            edge[prefix.len()..].to_vec(),
                            self.edges.remove(edge).unwrap(),
                        );
                        if label.eq(&prefix) {
                            // set token and id on split node
                            assert_eq!(split_node_ref.token_id, None);
                            split_node_ref.token = Some(token);
                            split_node_ref.token_id = Some(token_id);
                            drop(split_node_ref);
                            lookup_update(token_id, split_node.clone());
                        } else {
                            // add child node for suffix of label
                            let token_node = Rc::new(RefCell::new(TrieNode::new(
                                Some(token),
                                Some(token_id),
                                None,
                            )));
                            split_node_ref
                                .edges
                                .insert(label[prefix.len()..].to_vec(), token_node.clone());
                            lookup_update(token_id, token_node);
                        }
                    }
                    self.edges.insert(prefix, split_node);
                    Ok(())
                }
            }
            None => {
                debug!("Insert new edge {:?}->{:?}", &label, &token_id);
                let t = Rc::new(RefCell::new(TrieNode::new(
                    Some(token),
                    Some(token_id),
                    None,
                )));
                self.edges.insert(label, t.clone());
                lookup_update(token_id, t);
                Ok(())
            }
        };
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
                n.borrow().format_debug(label, level + 1, max_depth)
            )
        }
        tr
    }

    fn depth(&self) -> usize {
        let mut max_child_depth = 0;
        for edge_node in self.edges.values() {
            let child_depth = edge_node.borrow().depth();
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
        write!(f, "{:?}", self.root.borrow())
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
    use crate::token_trie::{Prob, Token, TokenId, TokenTrie, Trie};
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
        let trie = TokenTrie::new(args)?;
        println!("{:?}", trie);
        trie.update(probs.into_iter().collect())?;
        let d = trie.distribution();
        let reprs: Vec<TokenId> = d.iter().map(|x| x.0.clone()).collect();
        let tokens: Vec<Vec<TokenId>> = d.iter().map(|x| x.1.clone()).collect();
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
        let trie = TokenTrie::new(args)?;
        trie.update(vec![(0, 1f64), (1, 2f64), (2, 5f64)])?;

        let d = trie.distribution();
        let reprs: Vec<TokenId> = d.iter().cloned().map(|x| x.0).collect();
        let tokens: Vec<Vec<TokenId>> = d.iter().cloned().map(|x| x.1).collect();
        let probs: Vec<Prob> = d.iter().cloned().map(|x| x.2).collect();

        assert_eq!(reprs[1], 1);
        assert_eq!(tokens[1], vec![1, 2]);
        assert_eq!(probs[1], 7f64);

        let lookup_node = trie.lookup(&reprs[1])?;
        let lookup_ref = lookup_node.borrow();
        assert_eq!(tokens[1], lookup_ref.tokens());
        assert_eq!(probs[1], lookup_ref.probabilities().iter().sum::<f64>());

        Ok(())
    }

    #[test]
    fn test_resample_deep() -> Result<()> {
        let args = create_tokens(vec!["a", "alice", "an", "ant", "bob"]);
        let trie = TokenTrie::new(args)?;
        trie.update(vec![(0, 1f64), (1, 3f64), (2, 3f64), (3, 4f64), (4, 5f64)])?;

        let d = trie.distribution();
        let reprs: Vec<TokenId> = d.iter().cloned().map(|x| x.0).collect();
        let tokens: Vec<Vec<TokenId>> = d.iter().cloned().map(|x| x.1).collect();
        let probs: Vec<Prob> = d.iter().cloned().map(|x| x.2).collect();

        assert_eq!(reprs[0], 0);
        assert_eq!(tokens[0], vec![0, 1, 2, 3]);
        assert_eq!(probs[0], 11f64);

        Ok(())
    }

    #[test]
    fn test_resample_multi_split() -> Result<()> {
        let args = create_tokens(vec!["alice", "an", "albert", "ant", "bob", "a"]);
        let trie = TokenTrie::new(args)?;
        trie.update(vec![
            (0, 3f64),
            (1, 3f64),
            (2, 4f64),
            (3, 5f64),
            (4, 7f64),
            (5, 1f64),
        ])?;

        let d = trie.distribution();
        let reprs: Vec<TokenId> = d.iter().cloned().map(|x| x.0).collect();
        let tokens: Vec<Vec<TokenId>> = d.iter().cloned().map(|x| x.1).collect();
        let probs: Vec<Prob> = d.iter().cloned().map(|x| x.2).collect();

        assert_eq!(reprs[0], 5);
        assert_eq!(tokens[0], vec![5, 2, 0, 1, 3]);
        assert_eq!(probs[0], 16f64);

        let lookup_node = trie.lookup(&reprs[0])?;
        assert_eq!(tokens[0], lookup_node.borrow().tokens());
        assert_eq!(
            probs[0],
            lookup_node.borrow().probabilities().iter().sum::<f64>()
        );

        let d = lookup_node.borrow().distribution();
        let st_reprs: Vec<TokenId> = d.iter().cloned().map(|x| x.0).collect();
        let st_tokens: Vec<Vec<TokenId>> = d.iter().cloned().map(|x| x.1).collect();
        let st_probs: Vec<Prob> = d.iter().cloned().map(|x| x.2).collect();

        assert_eq!(st_tokens[0], vec![5, 2, 0, 1, 3]);
        assert_eq!(st_reprs[0], 5);
        assert_eq!(st_probs[0], 16f64);

        let st_lookup_node = trie.lookup(&st_reprs[0])?;
        assert_eq!(st_tokens[0], st_lookup_node.borrow().tokens());
        assert_eq!(
            st_probs[0],
            st_lookup_node.borrow().probabilities().iter().sum::<f64>()
        );

        Ok(())
    }

    #[test]
    fn test_resample_multi_split_pseudo() -> Result<()> {
        let tokens = create_tokens(vec!["alice", "an", "albert", "ant", "bob"]);
        let trie = TokenTrie::new(tokens)?;
        trie.update(vec![(0, 3f64), (1, 3f64), (2, 4f64), (3, 5f64), (4, 7f64)])?;

        println!("{:?}", trie);

        let d = trie.distribution();
        let reprs: Vec<TokenId> = d.iter().cloned().map(|x| x.0).collect();
        let tokens: Vec<Vec<TokenId>> = d.iter().cloned().map(|x| x.1).collect();
        let probs: Vec<Prob> = d.iter().cloned().map(|x| x.2).collect();

        println!("{:?}", d);

        assert_eq!(reprs[2], 1);
        assert_eq!(tokens[2], vec![1, 3]);
        assert_eq!(probs[2], 8f64);

        let lookup_node = trie.lookup(&reprs[2]);
        assert!(lookup_node.is_ok());

        let d = lookup_node.unwrap().borrow().distribution();
        let st_reprs: Vec<TokenId> = d.iter().cloned().map(|x| x.0).collect();
        let st_tokens: Vec<Vec<TokenId>> = d.iter().cloned().map(|x| x.1).collect();
        let st_probs: Vec<Prob> = d.iter().cloned().map(|x| x.2).collect();

        assert_eq!(st_tokens[0], vec![1, 3]);
        assert_eq!(st_reprs[0], 1);
        assert_eq!(st_probs[0], 8f64);
        assert_eq!(st_probs.len(), 1);
        assert_eq!(st_reprs.len(), 1);

        let st_lookup_node = trie.lookup(&st_reprs[0]);
        assert!(st_lookup_node.is_ok());
        assert_eq!(
            st_probs[0],
            st_lookup_node
                .unwrap()
                .borrow()
                .probabilities()
                .iter()
                .sum::<f64>()
        );

        Ok(())
    }

    #[test]
    fn test_empty_label_split() -> Result<()> {
        let tokens = create_tokens(vec!["ABC", "AB"]);
        let trie = TokenTrie::new(tokens)?;
        let probs = vec![(0, 2f64), (1, 3f64)];
        trie.update(probs)?;

        let lookup_node = trie.lookup(&1)?;
        assert_eq!(lookup_node.borrow().token_id, Some(1));
        assert_eq!(lookup_node.borrow().probability, Some(3f64));
        assert!(!lookup_node.borrow().edges.contains_key(&vec![]));

        Ok(())
    }

    #[test]
    fn test_from_labels() -> Result<()> {
        let tokens = create_tokens(vec!["Alice", "an", "ant", "a"]);
        let trie = TokenTrie::new(tokens)?;
        let probs = vec![(0, 1.0), (1, 4.0), (2, 7.0), (3, 22.0)];
        trie.update(probs)?;
        println!("{:?}", trie);

        let d = trie.distribution();
        let reprs: Vec<TokenId> = d.iter().cloned().map(|x| x.0).collect();
        let tokens: Vec<Vec<TokenId>> = d.iter().cloned().map(|x| x.1).collect();
        let probs: Vec<Prob> = d.iter().cloned().map(|x| x.2).collect();

        println!("{:?}", d);
        assert_eq!(reprs, vec![0, 3]);
        assert_eq!(tokens, vec![vec![0], vec![3, 1, 2]]);
        assert_eq!(probs, vec![1.0, 33.0]);

        let lookup_node = trie.lookup(&3)?;
        let d = lookup_node.borrow().distribution();
        let reprs: Vec<TokenId> = d.iter().cloned().map(|x| x.0).collect();
        let tokens: Vec<Vec<TokenId>> = d.iter().cloned().map(|x| x.1).collect();
        let probs: Vec<Prob> = d.iter().cloned().map(|x| x.2).collect();

        println!("{:?}", d);

        assert_eq!(reprs, vec![3]);
        assert_eq!(tokens, vec![vec![3, 1, 2]]);
        assert_eq!(probs, vec![33.0]);

        Ok(())
    }

    #[test]
    fn test_update_reset() -> Result<()> {
        let tokens = create_tokens(vec!["Alice", "an", "ant", "a"]);
        let trie = TokenTrie::new(tokens)?;
        let probs = vec![(0, 1.0), (1, 4.0), (2, 7.0), (3, 22.0)];
        trie.update(probs)?;

        println!("{:?}", trie);

        let d = trie.distribution();
        let reprs: Vec<TokenId> = d.iter().cloned().map(|x| x.0).collect();
        let tokens: Vec<Vec<TokenId>> = d.iter().cloned().map(|x| x.1).collect();
        let probs: Vec<Prob> = d.iter().cloned().map(|x| x.2).collect();

        assert_eq!(reprs, vec![0, 3]);
        assert_eq!(tokens, vec![vec![0], vec![3, 1, 2]]);
        assert_eq!(probs, vec![1.0, 33.0]);

        trie.update(vec![])?;
        assert_eq!(
            trie.distribution(),
            vec![(0, vec![0], 0f64), (3, vec![3, 1, 2], 0f64)]
        );

        trie.update(vec![(2, 22.0)])?;
        let d = trie.distribution();
        let reprs: Vec<TokenId> = d.iter().cloned().map(|x| x.0).collect();
        let tokens: Vec<Vec<TokenId>> = d.iter().cloned().map(|x| x.1).collect();
        let probs: Vec<Prob> = d.iter().cloned().map(|x| x.2).collect();

        assert_eq!(reprs, vec![0, 3]);
        assert_eq!(tokens, vec![vec![0], vec![3, 1, 2]]);
        assert_eq!(probs, vec![0.0, 22.0]);

        Ok(())
    }
}
