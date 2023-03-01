# caspailleur

Lightweight python package to explore binary data in FCA terms.


## Structure

```mermaid
  graph TD;
      S["<b>itemsets</b><br><small><tt>casp.np2bas(...)</tt></small>"];
      A["<b>intents</b><br><small><tt>casp.list_intents_via_LCM(...)</tt></small>"];
      B["<b>keys</b><br><small><tt>casp.list_keys(...)</tt></small>"];
      C["<b>passkeys</b><br><small><tt>casp.list_passkeys(...)</tt></small>"];
      D["<b>intents ordering</b><br><small><tt>casp.sort_intents_inclusion(...)</tt></small>"]; 
      E["<b>pseudo-intents</b><br><small><tt>casp.list_pseudo_intents_via_keys(...)</tt></small>"];
      F["<b>proper premises</b><br><small><tt>csp.iter_proper_premises_via_keys(...)</tt></small>"];
      G["<b>linearity index</b><br><small><tt>casp.linearity_index(...)</tt></small>"];
      H["<b>distributivity index</b><br><small><tt>casp.distributivity_index(...)</tt></small>"];
      
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
