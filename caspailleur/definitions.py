"""
This module contains functions to test whether
a set of attributes belongs to a given characteristic attribute set class
(e.g. pseudo-intents, minimal generators, etc.)
"""
from typing import FrozenSet, Union
from bitarray import frozenbitarray as fbarray


from .base_functions import closure, is_psubset_of, powerset


def is_closed(B: Union[FrozenSet[int], fbarray], crosses_per_columns: Union[list[FrozenSet[int]], list[fbarray]]) -> bool:
    """Test whether `B` is closed w.r.t. `crosses_per_columns`"""
    return B == set(closure(B, crosses_per_columns))


def is_pseudo_intent(
        P: FrozenSet[int],
        crosses_per_columns: Union[list[FrozenSet[int]], list[fbarray]],
        sub_pseudo_intents: list[FrozenSet[int]] = None,
        is_closed_: bool = None,
) -> bool:
    """Test whether `P` is a pseudo-intent w.r.t. `crosses_per_columns`

    An attribute set P is a pseudo-intent iff
    * P is not closed, and
    * for every smaller pseudo-intent Q, the closure of Q is in P
    """
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
        is_psubset_of(frozenset(closure(Q, crosses_per_columns)), P)
        for Q in sub_pseudo_intents
        if is_psubset_of(Q, P)
    )
    return answer


def is_proper_premise(
        Q: FrozenSet[int],
        crosses_per_columns: list[FrozenSet[int]],
        is_closed_: bool = None
) -> bool:
    """Test whether `Q` is a proper premise w.r.t. `crosses_per_columns`


    An attribute set Q is a proper premise
    * Q is not closed, and
    * Q | closure(Q-{n1}) | ... | closure(Q - {nk}) != closure(Q), where n1, ..., nk are elements of Q
    """
    if is_closed_ is None:
        is_closed_ = is_closed(Q, crosses_per_columns)
    if is_closed_:
        return False

    intent = set(closure(Q, crosses_per_columns))
    union_closure = set(Q)
    for n in Q:
        union_closure |= set(closure(Q-{n}, crosses_per_columns))
    return union_closure != intent


def is_minimal_gen(D: FrozenSet[int], crosses_per_columns: list[FrozenSet[int]]) -> bool:
    """Test whether `D` is a minimal generator w.r.t. `crosses_per_columns`

    An attribute set D is a minimal generator if there is no subset of D with the same closure as D
    """
    intent = set(closure(D, crosses_per_columns))
    return all(set(closure(D-{m}, crosses_per_columns)) != intent for m in D)


def is_minimum_gen(
        D: FrozenSet[int],
        crosses_per_columns: list[FrozenSet[int]],
        minimal_gens: list[FrozenSet[int]] = None,
        is_minimal_gen_: bool = None
) -> bool:
    """Test whether `D` is a minimum generator w.r.t. `crosses_per_columns`


    An attribute set D is a minimum generator if there is no attribute set of smaller size with the same closure as D
    """
    if is_minimal_gen_ is None:
        is_minimal_gen_ = is_minimal_gen(D, crosses_per_columns)
    if is_minimal_gen_ is False:
        return False

    if minimal_gens is None:
        descr_iterator = powerset(range(len(crosses_per_columns)))
    else:
        descr_iterator = sorted(minimal_gens, key=lambda E: len(E))
    intent = set(closure(D, crosses_per_columns))
    first_mingen = next(E for E in descr_iterator if set(closure(E, crosses_per_columns)) == intent)
    return len(D) == len(first_mingen)


def is_key(D: frozenset, crosses_per_columns: list[frozenset], is_minimal_gen_=None) -> bool:
    """Test whether `D` is a key w.r.t. `crosses_per_columns` (the same as minimal generator)"""
    return is_minimal_gen(D, crosses_per_columns) if is_minimal_gen_ is None else is_minimal_gen_


def is_passkey(
        D: FrozenSet[int],
        crosses_per_columns: list[FrozenSet[int]],
        keys: list[FrozenSet[int]] = None,
        is_minimum_gen_=None
) -> bool:
    """Test whether `D` is a passkey w.r.t. `crosses_per_columns` (the same as minimum generator)"""
    return is_minimum_gen(D, crosses_per_columns, keys) if is_minimum_gen_ is None else is_minimum_gen_
