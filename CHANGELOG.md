# Changelog

## v0.2.1 

### Fixes
Update the code to match the new syntaxis of `bitarray` package
(mostly renaming `ba.itersearch(...)` functions into `ba.search(...)` ).

Plus minor bugfixes.

### New functionality (backend version)

Add `support_surplus` parameter into  `iter_keys_of_intent` function.
Now one can find **∆-equivalent keys of an intent**,
that are minimal descriptions that describe the same objects as the intent 
plus at most `support_surplus` more additional objects.
The function in question is called `csp.mine_equivalence_classes.iter_keys_of_intent(...)`.

Implement **MRG-Exp** algorithm (Carpathia-G-Rare) for **Minimal Rare Itemset** mining.
A minimal rare itemset (or a minimal rare description) is a minimal subset of attributes
    that describes less than (or equal to) `max_support` objects.
The function is called `csp.mine_equivalence_classes.iter_minimal_rare_itemsets_via_mrgexp(...)`.

Implement **Clustering via MRG-Exp** algorithm for **Minimal Broad Clustering** mining.
A minimal broad clustering is a minimal subset of attributes that, together,
cover more than (or equal to) `min_coverage` objects.
The function is called `csp.mine_equivalence_classes.iter_minimal_broad_clusterings_via_mrgexp(...)`.

Add **level-wise equivalence class iteration**.
The level-wise optimisation is supposed to fasten up the iteration of the equivalence class.
However, currently, it only fastens up the iteration through the equivalence class of the maximal description
(that often describes no objects).   
The function is called `csp.mine_equivalence_classes.iter_equivalence_class_levelwise(...)`.
The older and more straightforward iteration procedure is implemented in 
`csp.mine_equivalence_classes.iter_equivalence_class(...)`.
