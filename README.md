# caspailleur

Lightweight python package to explore binary data in FCA terms.


## Structure
```mermaid
  graph TD;
      S[itemsets];
      A[intents]; B[keys]; C[passkeys]; D[intents ordering]; 
      E[pseudo-intents]; F[proper premises];
      G[linearity index]; H[distributivity index];
      S --> A
      A --> B;
      A --> C;
      A --> D;
      A --> E; 
      B --> E;  
      B --> F; A --> F; 
      A --> G; D --> G;
      D --> H; A --> H; 
```

```mermaid
  graph TD;
      S["<b>itemsets</b><br><small><tt>np2bas(...)</tt></small>"];
      A["<b>intents</b><br><small><tt>list_intents_via_LCM(...)</tt></small>"];
      B["<b>keys</b><br><small><tt>list_keys(...)</tt></small>"];
      C["<b>passkeys</b><br><small><tt>list_passkeys(...)</tt></small>"];
      D["<b>intents ordering</b><br><small><tt>sort_intents_inclusion(...)</tt></small>"]; 
      E["<b>pseudo-intents</b><br><small><tt>list_pseudo_intents_via_keys(...)</tt></small>"];
      F["<b>proper premises</b><br><small><tt>iter_proper_premises_via_keys(...)</tt></small>"];
      G["<b>linearity index</b><br><small><tt>linearity_index(...)</tt></small>"];
      H["<b>distributivity index</b><br><small><tt>distributivity_index(...)</tt></small>"];
      
      click S "https://github.com/EgorDudyrev/caspailleur/blob/b35a37f559ecceedd70a5b72301707a4ca94201c/caspailleur/base_functions.py#L43" "Link to np2bas function"
      click A "https://github.com/EgorDudyrev/caspailleur/blob/b35a37f559ecceedd70a5b72301707a4ca94201c/caspailleur/mine_equivalence_classes.py#L12" "Open function list_intents_via_LCM"
      click B "https://github.com/EgorDudyrev/caspailleur/blob/b35a37f559ecceedd70a5b72301707a4ca94201c/caspailleur/mine_equivalence_classes.py#L107" "Open function list_keys"
      click C "https://github.com/EgorDudyrev/caspailleur/blob/b35a37f559ecceedd70a5b72301707a4ca94201c/caspailleur/mine_equivalence_classes.py#L159" "Open function list_passkeys"
      click D "https://github.com/EgorDudyrev/caspailleur/blob/b35a37f559ecceedd70a5b72301707a4ca94201c/caspailleur/order.py#L28" "Open function sort_intents_inclusion"
      click E "https://github.com/EgorDudyrev/caspailleur/blob/b35a37f559ecceedd70a5b72301707a4ca94201c/caspailleur/implication_bases.py#L59" "Open function list_pseudo_intents_via_keys"
      click F "https://github.com/EgorDudyrev/caspailleur/blob/b35a37f559ecceedd70a5b72301707a4ca94201c/caspailleur/implication_bases.py#L45" "Open function iter_proper_premises_via_keys"
      click G "https://github.com/EgorDudyrev/caspailleur/blob/b35a37f559ecceedd70a5b72301707a4ca94201c/caspailleur/indices.py#L9" "Open function linearity_index"
      click H "https://github.com/EgorDudyrev/caspailleur/blob/b35a37f559ecceedd70a5b72301707a4ca94201c/caspailleur/indices.py#L25" "Open function distributivity_index"
      
      S --> A
      A --> B;
      A --> C;
      A --> D;
      A --> E; 
      B --> E;  
      B --> F; A --> F; 
      A --> G; D --> G;
      D --> H; A --> H; 
```


## Funding

The package development is supported by ANR project SmartFCA [(ANR-21-CE23-0023)](https://anr.fr/Projet-ANR-21-CE23-0023).

SmartFCA ([https://www.smartfca.org/](https://www.smartfca.org/)) is a big platform that will contain many extensions
of Formal Concept Analysis including pattern structures, Relational Concept Analysis, Graph-FCA and others. 
While caspailleur is a small python package that covers only the basic notions of FCA. 