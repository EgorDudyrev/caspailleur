"""
This module contains functions to test whether
a set of attributes belongs to a given characteristic attribute set class
(e.g. pseudo-intents, minimal generators, etc.)
"""
from typing import List, FrozenSet

from .base_functions import closure, is_psubset_of, powerset


def is_closed(B: FrozenSet[int], crosses_per_columns: List[FrozenSet[int]]) -> bool:
    """Test whether `B` is closed w.r.t. `crosses_per_columns`"""
    return B == closure(B, crosses_per_columns)


def is_pseudo_intent(
        P: FrozenSet[int],
        crosses_per_columns: List[FrozenSet[int]],
        sub_pseudo_intents: List[FrozenSet[int]] = None,
        is_closed_: bool = None,
) -> bool:
    """Test whether `P` is a pseudo-intent w.r.t. `crosses_per_columns`

    An attribute set $P$ is a __pseudo-intent__ iff
    * $P \neq P''$
    * $Q'' \subset P$ for every pseudo-intent $Q \subset P$
    """  # NOQA: W605 invalid escape sequence '\s'
    if is_closed_ is None:
        is_closed_ = is_closed(P, crosses_per_columns)

    if is_closed_:
        return False

    if sub_pseudo_intents is None:
        sub_pseudo_intents = []
        for D in powerset(P):
            if frozenset(D) == P:
                continue
            if is_pseudo_intent(D, crosses_per_columns, sub_pseudo_intents):
                sub_pseudo_intents.append(D)

    answer = all(
        is_psubset_of(closure(Q, crosses_per_columns), P)
        for Q in sub_pseudo_intents
        if is_psubset_of(Q, P)
    )
    return answer


def is_proper_premise(
        Q: FrozenSet[int],
        crosses_per_columns: List[FrozenSet[int]],
        is_closed_: bool =None
) -> bool:
    """Test whether `Q` is a proper premise w.r.t. `crosses_per_columns`

    Acc. to S. Kuznetsov "ML on the Basis of FCA":

    "Recall that a set $Q$ is a __proper premise__ of a context $K = (G, M, I)$ if
    * $Q ⊂ M$
    * $Q'' \neq Q$, and
    * $(Q \setminus \{n\})'' \neq Q, \quad \forall n \in Q$
    """  # NOQA: W605 invalid escape sequence '\s'
    if is_closed_ is None:
        is_closed_ = is_closed(Q, crosses_per_columns)
    if is_closed_:
        return False

    intent = closure(Q, crosses_per_columns)
    union_closure = set(Q)
    for n in Q:
        union_closure |= closure(Q-{n}, crosses_per_columns)
    return union_closure != intent


def is_minimal_gen(D: FrozenSet[int], crosses_per_columns: List[FrozenSet[int]]) -> bool:
    """Test whether `D` is a minimal generator w.r.t. `crosses_per_columns`

    $D \subseteq M$ is a __minimal generator__ iff $$\nexists m \in D: (D\setminus \{m\})'' = D''$$
    or
    $$\nexists m \in D: (D\setminus \{m\})' = D'$$
    """
    intent = closure(D, crosses_per_columns)
    return all(closure(D-{m}, crosses_per_columns) != intent for m in D)


def is_minimum_gen(
        D: FrozenSet[int],
        crosses_per_columns: List[FrozenSet[int]],
        minimal_gens: List[FrozenSet[int]] = None,
        is_minimal_gen_: bool = None
) -> bool:
    """Test whether `D` is a minimum generator w.r.t. `crosses_per_columns`

    $D \subseteq M$ is a __minimum generator__ iff
    * $D$ is a minimal generator
    * $\nexists E \subseteq M$ s.t. $E$ is a minimal generator and $|E| > |D|$
    """  # NOQA: W605 invalid escape sequence '\s'
    if is_minimal_gen_ is None:
        is_minimal_gen_ = is_minimal_gen(D, crosses_per_columns)
    if is_minimal_gen_ is False:
        return False

    if minimal_gens is None:
        descr_iterator = powerset(range(len(crosses_per_columns)))
    else:
        descr_iterator = sorted(minimal_gens, key=lambda E: len(E))
    intent = closure(D, crosses_per_columns)
    first_mingen = next(E for E in descr_iterator if closure(E, crosses_per_columns) == intent)
    return len(D) == len(first_mingen)


def is_key(D: frozenset, crosses_per_columns: List[frozenset], is_minimal_gen_=None) -> bool:
    """Test whether `D` is a key w.r.t. `crosses_per_columns` (the same as minimal generator)"""
    return is_minimal_gen(D, crosses_per_columns) if is_minimal_gen_ is None else is_minimal_gen_


def is_passkey(
        D: FrozenSet[int],
        crosses_per_columns: List[FrozenSet[int]],
        keys: List[FrozenSet[int]] = None,
        is_minimum_gen_=None
) -> bool:
    """Test whether `D` is a passkey w.r.t. `crosses_per_columns` (the same as minimum generator)"""
    return is_minimum_gen(D, crosses_per_columns, keys) if is_minimum_gen_ is None else is_minimum_gen_
