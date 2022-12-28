from caspailleur import definitions as defs
from caspailleur.base_functions import powerset


def test_is_closed():
    crosses_per_columns = [
        {1, 2},     # a
        {3, 4},     # b
        {2, 3, 4},  # c
        {1, 4},     # d
        set(),      # e
    ]
    attrs = 'abcde'
    closed_sets_true = {'bc', 'bcd', 'abcde', 'a', 'c', 'd', 'ac', 'ad', ''}

    powset = powerset(range(len(attrs)))
    closed_sets = {B for B in powset if defs.is_closed(B, crosses_per_columns)}
    closed_sets_verb = {''.join([attrs[m] for m in sorted(B)]) for B in closed_sets}
    assert closed_sets_verb == closed_sets_true


def test_is_pseudo_intent():
    crosses_per_columns = [
        {1, 2},     # a
        {3, 4},     # b
        {2, 3, 4},  # c
        {1, 4},     # d
        set(),      # e
    ]
    attrs = 'abcde'
    pseudo_intents_true = {'b', 'e', 'cd', 'abc'}

    powset = list(powerset(range(len(attrs))))

    pseudo_intents = []
    for B in powset:
        if defs.is_pseudo_intent(B, crosses_per_columns, pseudo_intents):
            pseudo_intents.append(B)
    pseudo_intents_verb = {''.join([attrs[m] for m in sorted(B)]) for B in pseudo_intents}
    assert pseudo_intents_verb == pseudo_intents_true

    pseudo_intents = {B for B in powset if defs.is_pseudo_intent(B, crosses_per_columns)}
    pseudo_intents_verb = {''.join([attrs[m] for m in sorted(B)]) for B in pseudo_intents}
    assert pseudo_intents_verb == pseudo_intents_true


def test_is_proper_premise():
    crosses_per_columns = [
        {1, 2},  # a
        {3, 4},  # b
        {2, 3, 4},  # c
        {1, 4},  # d
        set(),  # e
    ]
    attrs = 'abcde'
    pprems_true = {'b', 'e', 'cd', 'ab', 'acd'}
    powset = powerset(range(len(attrs)))
    pprems = {B for B in powset if defs.is_proper_premise(B, crosses_per_columns)}
    pprems_verb = {''.join([attrs[m] for m in sorted(B)]) for B in pprems}
    assert pprems_verb == pprems_true


def test_is_minimal_gen():
    crosses_per_columns = [
        {1, 2},  # a
        {3, 4},  # b
        {2, 3, 4},  # c
        {1, 4},  # d
        set(),  # e
    ]
    attrs = 'abcde'
    mingens_true = {'a', 'c', 'd', 'ac', 'ad', '', 'bd', 'b', 'e', 'cd', 'ab', 'acd'}
    powset = powerset(range(len(attrs)))
    mingens = {B for B in powset if defs.is_minimal_gen(B, crosses_per_columns)}
    mingens_verb = {''.join([attrs[m] for m in sorted(B)]) for B in mingens}
    assert mingens_verb == mingens_true


def test_is_minimum_gen():
    crosses_per_columns = [
        {1, 2},  # a
        {3, 4},  # b
        {2, 3, 4},  # c
        {1, 4},  # d
        set(),  # e
    ]
    attrs = 'abcde'
    mingens_true = {'a', 'c', 'd', 'ac', 'ad', '', 'bd', 'b', 'e', 'cd'}
    powset = list(powerset(range(len(attrs))))

    mingens = {B for B in powset if defs.is_minimum_gen(B, crosses_per_columns)}
    mingens_verb = {''.join([attrs[m] for m in sorted(B)]) for B in mingens}
    assert mingens_verb == mingens_true

    minimal_gens = {'a', 'c', 'd', 'ac', 'ad', '', 'bd', 'b', 'e', 'cd', 'ab', 'acd'}
    minimal_gens_unverb = {frozenset({attrs.index(m) for m in D}) for D in minimal_gens}
    mingens = {B for B in powset if defs.is_minimum_gen(B, crosses_per_columns, minimal_gens_unverb)}
    mingens_verb = {''.join([attrs[m] for m in sorted(B)]) for B in mingens}
    assert mingens_verb == mingens_true


def test_is_key():
    crosses_per_columns = [
        {1, 2},  # a
        {3, 4},  # b
        {2, 3, 4},  # c
        {1, 4},  # d
        set(),  # e
    ]
    attrs = 'abcde'
    keys_true = {'a', 'c', 'd', 'ac', 'ad', '', 'bd', 'b', 'e', 'cd', 'ab', 'acd'}
    powset = powerset(range(len(attrs)))
    keys = {B for B in powset if defs.is_key(B, crosses_per_columns)}
    keys_verb = {''.join([attrs[m] for m in sorted(B)]) for B in keys}
    assert keys_verb == keys_true


def test_is_passkey():
    crosses_per_columns = [
        {1, 2},  # a
        {3, 4},  # b
        {2, 3, 4},  # c
        {1, 4},  # d
        set(),  # e
    ]
    attrs = 'abcde'
    pkeys_true = {'a', 'c', 'd', 'ac', 'ad', '', 'bd', 'b', 'e', 'cd'}
    powset = list(powerset(range(len(attrs))))

    pkeys = {B for B in powset if defs.is_passkey(B, crosses_per_columns)}
    pkeys_verb = {''.join([attrs[m] for m in sorted(B)]) for B in pkeys}
    assert pkeys_verb == pkeys_true

    keys = {'a', 'c', 'd', 'ac', 'ad', '', 'bd', 'b', 'e', 'cd', 'ab', 'acd'}
    keys_unverb = {frozenset({attrs.index(m) for m in D}) for D in keys}
    pkeys = {B for B in powset if defs.is_passkey(B, crosses_per_columns, keys_unverb)}
    pkeys_verb = {''.join([attrs[m] for m in sorted(B)]) for B in pkeys}
    assert pkeys_verb == pkeys_true
