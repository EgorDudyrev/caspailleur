# Basics of FCA
* **Object**: An index of a row in the data
* **Attribute**: An index of a (binary) column in the data
* **Formal Context**: A binary dataset represented as a triplet of objects, attributes and their connections
* **Description**: A subset of attributes
* **Extent**: The maximal subset of objects described by some description
* **Intent**: The maximal subset of attributes describing some objects. 
Also, the maximal subset of attributes describing the same objects as some given description. 
* **Concept**: A pair of corresponding extent and intent, so a pair of a maximal subset of objects and their maximal description.
* **Concept Lattice**: A set of all concepts in the data ordered by generality


* **Order on concepts**: Concepts are ordered by their generality.
So concept _A_  is less than concept _B_ if _A_ is less general than _B_.
That is, if _B_ covers all the objects from _A_, or if _A_ contains all the attributes from _B_.

* **Sub concepts**: All concepts that are less general than some given concept
* **Super concepts**: All concepts that are more general than some given concept
* **Previous concepts**: The most general sub concepts
* **Next concepts**: The least general super concepts

# Minimal descriptions
* **Key**: A minimal subset of attributes describing some objects (_there may be many keys for the same subset of objects_) 
* **Passkey**: A shortest subset of attributes describing some objects (_there may be many passkeys for the same subset of objects_)

# Implications
* **Premise**: The left part of implication _A => B_, so the condition of the implication
* **Conclusion**: The right part of implication _A => B_, so what is implied by the implication 
* **Saturation**: The process of enriching a description with a given set of implications.
For example, given description {color_is_green} and implications {} => {fruit}, {fruit, color_is_green} => {form_is_oval},
description {color_is_green} can be saturated into description {color_is_green, fruit}, because everything implies {fruit},
And then description {color_is_green, fruit} can be saturated into {color_is_green, fruit, form_is_oval} based on the second given implication.
* **Proper Premise**: A description that implies some attributes, not implied by its subdescriptions.
* **Pseudo-intent**: A description that implies some attributes, not implied by its subdescriptions and saturated w.r.t. the other pseudo-intents.

* **Canonical Direct basis** (also **Proper Premise basis** or **Karell basis**): 
A set of implications where every premise is a proper premise. Such set of implication is direct, that is one can saturate a description passing every implication only once. 
* **Canonical basis** (also **Pseudo-intent basis** or **Duquenne-Guiges basis**):
A set of implications where every premise is a pseudo-intent. Such set of implication is the smallest possible set of implications covering all the implications in the data.
* **Unit basis**: A set of implications where every conclusion is a single attribute and not a subset of attributes. 


# Interestingness measures
* **Support**: The number of objects described by a description (or concept, or implication). 
In `caspailleur` the term "support" is synonymous to "frequency" which is the percentage of objects described by a description.
* **Delta-stability**: The minimal number of objects a description will lose if added at least one other attribute.
