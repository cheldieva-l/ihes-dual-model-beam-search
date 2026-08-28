# Method

## 1. Permutation convention

An IHES state and every generator are permutations of `0..71`. Applying generator `g` to state `s` is implemented exactly as:

```text
next_state = s[g]
```

Thus a path is an ordered right-composition of generators. The inverse of a path reverses its move order and replaces each move by its named inverse. All 18 official generators participate in the search.

## 2. MLP score and global beam

The checkpoint-compatible network one-hot encodes all 72 state entries, uses two fully connected layers with batch normalization, applies the configured residual blocks, and emits one scalar. At each depth the search:

1. expands every retained state by all 18 generators;
2. scores children in GPU batches;
3. deduplicates equal states, retaining the best-scoring occurrence;
4. selects the global best `K` unique states.

Two deterministic 64-bit Zobrist hashes index states. Hash equality is never treated as state equality in correctness-sensitive operations: collision groups and frontier meetings are verified by exact comparison of all 72 entries.

## 3. Symmetry frame

For a relabelling permutation `R`, let `R_inv = argsort(R)`. A state is represented in the rotated frame as:

```text
rotate_R(s)[i] = R[s[R_inv[i]]]
```

For each generator `g`, the conjugated generator is:

```text
g_R[i] = R_inv[g[R[i]]]
```

The 48 supplied rotations are accepted only when every `g_R` exactly equals one of the 18 official generators. This produces an explicit move table from the rotated frame back to original coordinates. Every transformed solution is replayed on the original state.

## 4. Reverse search

For a rotated start `s_R`, reverse search starts from `inverse(s_R)`. If it finds a path `B` to identity, the corresponding direct path is `inverse(B)`, meaning reversed order with inverse moves. That frame path is then conjugated back to original move names and replayed.

The reverse-neighbour tensor from the IHES asset is used only as a validation oracle. Rows inside each 18-row puzzle block are treated as an unordered set because the observed file does not preserve official generator order.

## 5. Bidirectional projection

The direct and reverse frontiers are not in the same projection. For direct start `s` and reverse state `r`, the exact map into direct coordinates is:

```text
M_s(r)[q] = s[r[q]]
```

The implementation applies `M_s` to every row of the complete reverse frontier before hashing and intersection. A candidate meeting must satisfy exact 72-entry equality. No known midpoint or midpoint hash is supplied to the blind join.

If the forward path to the meeting is `A` and the reverse path is `B`, the full frame solution is:

```text
A + inverse(B)
```

It is conjugated back to the original frame and replayed from the original puzzle state.

## 6. Known-path diagnostic

Given a known solution `P` only for evaluation, the direct diagnostic follows prefixes of `P`. The reverse diagnostic follows prefixes of `inverse(P)`, which correspond to inverted suffixes of `P`. For complementary depths `d` and `len(P)-d`, the implementation verifies:

```text
direct_state[d] == M_s(reverse_state[len(P)-d])
```

At every depth it logs whether the exact known state:

- was generated from the prior retained frontier;
- survived the ordinary global top-K selection;
- was placed in the final slot by diagnostic protection.

Protection is permitted only after exact candidate equality proves generation. It cannot resurrect a state that was never generated. Protected runs are diagnostic and must not be used as ordinary blind-search evidence. Normal search keeps protection disabled.

These observations establish a checkable invariant for one run. They do not prove that a reference path survives for every puzzle, model, beam width, or depth.

## 7. Submission invariant

A replacement is written into the official two-column sample file. The validator then parses and replays every path for all 1,003 test states. It rejects missing IDs, duplicate IDs, extra columns, unknown generators, incorrect terminal states, or an invalid replacement.

