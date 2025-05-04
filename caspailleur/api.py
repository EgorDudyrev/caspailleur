"""Module with high-level functions to work with Caspailleur

The proposed functions are:
* iter_descriptions(data) to iterate all possible descriptions (and their characteristics) in the data
* mine_descriptions(data) to get the Pandas DataFrame with all possible descriptions in the data
    (note that "the number of all descriptions" = 2**"the number of binary attributes",
    so the resulting DataFrame might be huge even for small data)
* mine_concepts(data) to get all the concepts (and their characteristics) in the data
* mine_implications(data, basis_name) to get a basis of implications for the data
"""
from typing import Iterator, Iterable, Literal, Union, Optional, get_args
import pandas as pd
from bitarray import frozenbitarray as fbarray

from .base_functions import powerset, extension, intention
from .io import ContextType, to_named_bitarrays, transpose_context, isets2bas, verbalise, to_absolute_number
from .order import topological_sorting, sort_intents_inclusion, inverse_order
from . import indices as idxs
from . import definitions
from . import mine_equivalence_classes as mec, implication_bases as ibases


MINE_DESCRIPTIONS_COLUMN = Literal[
    "description", "extent", "intent",
    "support", "delta_stability",
    "is_closed", "is_key", "is_passkey", "is_proper_premise", "is_pseudo_intent"
]

MINE_CONCEPTS_COLUMN = Literal[
    "extent", "intent", "support", "delta_stability",
    "keys", "passkeys", "proper_premises", "pseudo_intents",
    "previous_concepts", "next_concepts", "sub_concepts",  "super_concepts",
]

MINE_IMPLICATIONS_COLUMN = Literal[
    "premise", "conclusion", "conclusion_full", "extent", "support", "delta_stability"
]

BASIS_NAME = Literal[
    "Proper Premise", "Canonical Direct", "Karell",
    "Pseudo-Intent", "Canonical", "Duquenne-Guigues", "Minimum",
]


def _setup_colnames_to_compute(
        all_columns,
        columns_to_compute: Union[list[str], Literal['all'], None],
        dependencies: dict[Union[str, tuple[str, bool]], set[str]], return_all_computed: bool
) -> tuple[set[str], list[str]]:
    columns_to_return = list(get_args(all_columns))
    if columns_to_compute is not None and columns_to_compute != 'all':
        columns_to_return = list(columns_to_compute)
    columns_to_compute = set(columns_to_return)

    assert columns_to_compute <= set(get_args(all_columns)), \
        f"The following elements were asked for but cannot be computed {columns_to_compute - set(all_columns)}. " \
        f"However, only the following columns can be chosen: {all_columns}."

    for premise, conclusion in dependencies.items():
        if not isinstance(premise, str):
            if not premise[1]:
                continue
            premise = premise[0]
        columns_to_compute.update(conclusion if premise in columns_to_compute or premise == '' else set())

    if return_all_computed:
        columns_to_return += sorted(set(columns_to_compute) - set(columns_to_return), key=list(get_args(all_columns)).index)
    return columns_to_compute, columns_to_return


def iter_descriptions(
        data: ContextType,
        to_compute: Union[list[MINE_DESCRIPTIONS_COLUMN], Literal['all']] = 'all'
) -> Iterator[dict]:
    """Iterate all possible descriptions in the data one by one

    Parameters
    ----------
    data: ContextType
        Data in a format, supported by Caspailleur.io module.
        For example, Pandas DataFrame with binary values,
        or list of lists of strings (where every list of strings represents an itemset).
    to_compute: list[MINE_DESCRIPTIONS_COLUMN] or 'all'
        A list of characteristics to compute (by default: "all")
        The list of all possible characteristics is defined in caspailleur.api.MINE_DESCRIPTIONS_COLUMN value.

    Returns
    -------
    descriptions: Iterator[dict]
        Iterator of descriptions. Every description and its characteristics are represented with a dictionary
        where the keys are defined by `to_compute` parameter.


    Notes
    -----
    Every description is a subset of attributes,
    and the maximal amount of possible descriptions is 2**(number of attributes).
    Thus, the resulting iterator will contain the exponential amount of elements,
    unless you define some additional early stopping outside of this function.
    """
    bitarrays, objects, attributes = to_named_bitarrays(data)
    attr_extents = transpose_context(bitarrays)
    to_compute, cols_to_return = _setup_colnames_to_compute(MINE_DESCRIPTIONS_COLUMN, to_compute, {}, True)

    sub_pseudo_intents = []

    def get_vals_for_column(column_name: MINE_DESCRIPTIONS_COLUMN):
        if column_name == 'description':
            return set(verbalise(description_idxs, attributes))
        if column_name == 'extent':
            return set(verbalise(extent_ba, objects))
        if column_name == 'intent':
            return set(verbalise(definitions.closure(description_idxs, attr_extents), attributes))
        if column_name == 'support':
            return idxs.support_by_description(description_idxs, attr_extents, extent_ba)
        if column_name == 'delta_stability':
            return idxs.delta_stability_by_description(description_idxs, attr_extents)
        if column_name == 'is_closed':
            return definitions.is_closed(description_idxs, attr_extents)
        if column_name == 'is_key':
            return definitions.is_key(description_idxs, attr_extents)
        if column_name == 'is_passkey':
            return definitions.is_passkey(description_idxs, attr_extents)
        if column_name == 'is_proper_premise':
            return definitions.is_proper_premise(description_idxs, attr_extents)
        if column_name == 'is_pseudo_intent':
            return definitions.is_pseudo_intent(description_idxs, attr_extents, sub_pseudo_intents)
        raise NotImplementedError(f"Something's gone wrong in the code, asked for column: {column_name}")

    for description_idxs in powerset(range(len(attributes))):
        extent_ba = fbarray(extension(description_idxs, attr_extents))
        stats = {colname: get_vals_for_column(colname) for colname in cols_to_return}

        yield stats
        if 'is_pseudo_intent' in cols_to_return and stats['is_pseudo_intent']:
            sub_pseudo_intents.append(description_idxs)


def mine_descriptions(
        data: ContextType,
        min_support: Union[int, float] = 0,
        to_compute: Union[list[MINE_DESCRIPTIONS_COLUMN], Literal['all']] = 'all',
        return_every_computed_column: bool = False
) -> pd.DataFrame:
    """Mine all frequent descriptions and their characteristics

    The frequency of a description is defined by its "support" index,
    which is the amount (or the percentage) of objects described by the description.

    Parameters
    ----------
    data: ContextType
        Data in a format, supported by Caspailleur.io module.
        For example, Pandas DataFrame with binary values,
        or list of lists of strings (where every list of strings represents an itemset).
    min_support: int or float
        Minimal value of the support for a description, i.e. how many objects it describes.
        Can be defined by an integer (so the number of objects) or by a float (so the percentage of objects).
    to_compute: list[MINE_DESCRIPTIONS_COLUMN] or 'all'
        A list of characteristics to compute (by default: "all")
        The list of all possible characteristics is defined in caspailleur.api.MINE_DESCRIPTIONS_COLUMN value.
    return_every_computed_column: bool
        A flag whether to return every computed column or only the ones defined by `to_compute` parameter.
        For example, the computation of column 'is_proper_premise' requires the computation of column 'is_key'.
        Thus, if you have asked for column 'is_proper_premise' you can get the column 'is_key' ``for free''.

    Returns
    -------
    descriptions_df: pandas.DataFrame
        Pandas DataFrame where every row represents a description, and every column represents its characteristics
        defined by `to_compute` and `return_every_computed_column` parameters.

    Notes
    -----
    If you want to look at only the most stable descriptions, then use "mine_concepts" function,
    as all stable descriptions are concept intents.

    Every description is a subset of attributes.
    So the maximal amount of possible descriptions is 2**(number of attributes).
    Make sure to specify the `min_support` parameter when dealing with the data with dozens of attributes,
    otherwise you might end up with a resulting DataFrame of hundreds of thousands rows.
    """
    ####################################################
    # Computing what columns and parameters to compute #
    ####################################################
    col_dependencies: dict[MINE_DESCRIPTIONS_COLUMN, set[MINE_DESCRIPTIONS_COLUMN]] = {
        'is_pseudo_intent': {'is_proper_premise', 'intent'},
        'is_proper_premise': {'intent', 'is_key'},
        'is_key': {'intent'},
        'is_passkey': {'intent'},
        'is_closed': {'intent'},
        ('', min_support > 0): {'support'},  # always compute support if min_support > 0
        'support': {'extent'},
        'delta_stability': {'extent'},
        'intent': {'extent'},
    }
    to_compute, cols_to_return = _setup_colnames_to_compute(
        MINE_DESCRIPTIONS_COLUMN, to_compute, col_dependencies, return_every_computed_column)

    ################################
    # Compute the required columns #
    ################################
    bitarrays, objects, attributes = to_named_bitarrays(data)
    attr_extents = transpose_context(bitarrays)
    n_objects = len(objects)
    min_support = to_absolute_number(min_support, n_objects)

    descriptions_ba = list(isets2bas(powerset(range(len(attributes))), len(attributes)))
    if 'extent' in to_compute:
        extents_ba = [fbarray(extension(descr, attr_extents)) for descr in descriptions_ba]
    if min_support > 0:
        descriptions_ba, extents_ba = zip(*[(descr, ext) for descr, ext in zip(descriptions_ba, extents_ba)
                                              if ext.count() >= min_support])
    column_description = [verbalise(descr, attributes) for descr in descriptions_ba]
    if 'extent' in to_compute:
        column_extent = [verbalise(extent, objects) for extent in extents_ba]
    if 'intent' in to_compute:
        intents_ba = mec.list_intents_via_LCM(bitarrays, min_supp=min_support)
        ext_int_map = {fbarray(extension(intent, attr_extents)): intent_i for intent_i, intent in enumerate(intents_ba)}
        column_intent = [verbalise(intents_ba[ext_int_map[extent]], attributes) for extent in extents_ba]
    if 'is_key' in to_compute:
        keys_ba = mec.list_keys(intents_ba)
    if 'is_passkey' in to_compute:
        passkeys_ba = mec.list_passkeys(intents_ba)
    if 'is_proper_premise' in to_compute:
        ppremises_ba = dict(ibases.iter_proper_premises_via_keys(intents_ba, keys_ba))
    if 'is_pseudo_intent' in to_compute:
        pintents_ba = dict(ibases.list_pseudo_intents_via_keys(ppremises_ba.items(), intents_ba))
    if 'support' in to_compute:
        column_support = [extent.count() for extent in extents_ba]
    if 'delta_stability' in to_compute:
        column_delta_stability = [idxs.delta_stability_by_description(descr, attr_extents, extent)
                                  for descr, extent in zip(descriptions_ba, extents_ba)]
    if 'is_closed' in to_compute:
        column_is_closed = [intents_ba[ext_int_map[extent]] == descr_ba
                            for descr_ba, extent in zip(descriptions_ba, extents_ba)]
    if 'is_key' in to_compute:
        column_is_key = [descr_ba in keys_ba for descr_ba in descriptions_ba]
    if 'is_passkey' in to_compute:
        column_is_passkey = [descr_ba in passkeys_ba for descr_ba in descriptions_ba]
    if 'is_proper_premise' in to_compute:
        column_is_proper_premise = [descr_ba in ppremises_ba for descr_ba in descriptions_ba]
    if 'is_pseudo_intent' in to_compute:
        column_is_pseudo_intent = [descr_ba in pintents_ba for descr_ba in descriptions_ba]

    locals_ = locals()
    return pd.DataFrame({f: locals_[f"column_{f}"] for f in cols_to_return}).rename_axis('description_id')


def mine_concepts(
        data: ContextType,
        to_compute: Union[list[MINE_CONCEPTS_COLUMN], Literal['all']] = 'all',
        return_every_computed_column: bool = False,
        min_support: Union[int, float] = 0,
        min_delta_stability: Union[int, float] = 0, n_stable_concepts: Optional[int] = None,
        use_tqdm: bool = False,
) -> pd.DataFrame:
    """Compute the frequent concepts in the data

    Parameters
    ----------
    data: ContextType
        Data in a format, supported by Caspailleur.io module.
        For example, Pandas DataFrame with binary values,
        or list of lists of integers (where every list of integers represents an itemset).
    to_compute: list[MINE_CONCEPT_COLUMN] or 'all'
        A list of characteristics to compute (by default, compute 'all' possible columns)
        The list of all possible characteristics is defined in caspailleur.api.MINE_CONCEPTS_COLUMN value.
    return_every_computed_column: bool
        A flag whether to return every computed column or only the ones defined by `to_compute` parameter.
        For example, the computation of column 'is_proper_premise' requires the computation of column 'is_key'.
        Thus, if you have asked for column 'is_proper_premise' you can get the column 'is_key' ``for free''.
    min_support: int or float
        Minimal value of the support for a concept, i.e. how many objects it describes.
        Can be defined by an integer (so the number of objects) or by a float (so the percentage of objects).
    min_delta_stability: int or float
        Minimal value of the delta-stability of a concept,
        i.e. the minimal amount of objects a concept will lose when made a bit more precise.
        Can be defined by an integer (so the number of objects) or by a float (so the percentage of objects).
    n_stable_concepts: int or None
        Select only `n` concepts with the highest (presumably) delta-stability.
        The parameter can be used together with or instead of `min_delta_stability`
        when the exact value the of required min_delta_stability is not known.
        The found `n` stable concepts are not necessarily the concepts with the _highest_ delta stability.
        Yet their delta-stability is high enough.
    use_tqdm: bool
        A flag whether to use tqdm progress bar for long computations.
        Is used for computing pseudo-intents.

    Returns
    -------
    descriptions_df: pandas.DataFrame
        Pandas DataFrame where every row represents a concept, and every column represents its characteristics
        defined by `to_compute` and `return_every_computed_column` parameters.

    Notes
    -----
    The number of concepts might be exponential to the number of objects and attributes in the data.
    Make sure to specify the `min_support` parameter when dealing with the data with dozens of attributes,
    otherwise you might end up with a resulting DataFrame of hundreds of thousands rows.

    """
    ##################################################
    # Computing what columns and parameters to compute
    ##################################################
    # whether to compute all concepts whose support is higher min_support
    compute_only_stable_concepts = min_delta_stability != 0 or n_stable_concepts is not None
    col_dependencies: dict[MINE_CONCEPTS_COLUMN, set[MINE_CONCEPTS_COLUMN]] = {
        'pseudo_intents': {'proper_premises', 'intent'},
        'proper_premises': {'intent', 'keys'},
        'keys': {'intent'},
        'passkeys': {'intent'},
        ('keys', compute_only_stable_concepts): {'extent'},  # extents are not required when computing join semilattice
        ('passkeys', compute_only_stable_concepts): {'extent'},
        'support': {'extent'},
        'delta_stability': {'intent', 'extent'},
        'extent': {'intent'},
        'super_concepts': {'sub_concepts'},
        'sub_concepts': {'previous_concepts'},
        'next_concepts': {'previous_concepts'},
        'previous_concepts': {'intent'},
    }
    to_compute, cols_to_return = _setup_colnames_to_compute(
        MINE_CONCEPTS_COLUMN, to_compute, col_dependencies, return_every_computed_column)

    #################################
    # Running the (long) computations
    #################################
    itemsets_ba, objects, attributes = to_named_bitarrays(data)
    attr_extents = transpose_context(itemsets_ba)
    itemsets_ba, attr_extents = list(map(fbarray, itemsets_ba)), list(map(fbarray, attr_extents))

    def verbalise_descriptions(bas):
        return [verbalise(ba, attributes) for ba in bas]

    def group_by_concept(pairs: Iterable[tuple[fbarray, int]], n_cncpts: int) -> list[list[set[str]]]:
        per_concept = [[] for _ in range(n_cncpts)]
        for ba, cncpt_i in pairs:
            per_concept[cncpt_i].append(ba)
        return per_concept

    def compute_intents(
            attr_extents_, itemsets_ba_,
            min_support_, compute_only_stable_concepts_, min_delta_stability_, n_stable_concepts_
    ):
        if compute_only_stable_concepts_:
            n_objects = len(attr_extents_[0])
            stable_extents = mec.list_stable_extents_via_gsofia(
                attr_extents_, n_objects, min_delta_stability_, n_stable_concepts_,
                min_supp=to_absolute_number(min_support_, n_objects), n_attributes=len(attr_extents_)
            )
            intents_ba = [intention(extent, attr_extents_) for extent in stable_extents]
        else:
            intents_ba = mec.list_intents_via_LCM(itemsets_ba_, min_supp=min_support_)
        intents_ba = topological_sorting(intents_ba)[0]
        n_concepts = len(intents_ba)
        return intents_ba, n_concepts

    def compute_extents(intents_ba_, attr_extents_):
        return [fbarray(extension(intent, attr_extents_)) for intent in intents_ba_]

    def compute_keys(extents_ba_, intents_ba_, attr_extents_, compute_only_stable_concepts_):
        if compute_only_stable_concepts_:
            return mec.list_keys_for_extents(extents_ba_, attr_extents_)
        return mec.list_keys(intents_ba_)

    def compute_passkeys(extents_ba_, intents_ba_, attr_extents_, compute_only_stable_concepts_):
        if compute_only_stable_concepts_:
            return mec.list_passkeys_for_extents(extents_ba_, attr_extents_)
        return mec.list_passkeys(intents_ba_)

    intents_ba, n_concepts = None, None
    if 'intent' in to_compute:
        intents_ba, n_concepts = compute_intents(
            attr_extents, itemsets_ba,
            min_support, compute_only_stable_concepts, min_delta_stability, n_stable_concepts
        )
    extents_ba = compute_extents(intents_ba, attr_extents) if 'extent' in to_compute else None
    keys_ba = compute_keys(extents_ba, intents_ba, attr_extents, compute_only_stable_concepts) if 'keys' in to_compute else None
    passkeys_ba = compute_passkeys(extents_ba, intents_ba, attr_extents, compute_only_stable_concepts) if 'passkeys' in to_compute else None
    proper_premises_ba = dict(ibases.iter_proper_premises_via_keys(intents_ba, keys_ba)) if 'proper_premises' in to_compute else None
    pseudo_intents_ba = None
    if 'pseudo_intents' in to_compute:
        pseudo_intents_ba = dict(ibases.list_pseudo_intents_via_keys(
            proper_premises_ba.items(), intents_ba, use_tqdm=use_tqdm, n_keys=len(proper_premises_ba)))

    # Columns for order on concepts
    previous_concepts, sub_concepts = None, None
    if 'previous_concepts' in to_compute:
        previous_concepts, sub_concepts = sort_intents_inclusion(intents_ba, return_transitive_order=True)
    next_concepts = inverse_order(previous_concepts) if 'next_concepts' in to_compute else None
    super_concepts = inverse_order(sub_concepts) if 'super_concepts' in to_compute else None

    ###################################
    # Verbalise the columns to return #
    ###################################
    if 'intent' in cols_to_return:
        column_intent = verbalise_descriptions(intents_ba)
    if 'extent' in cols_to_return:
        column_extent = [verbalise(extent, objects) for extent in extents_ba]
    if 'keys' in cols_to_return:
        column_keys = [verbalise_descriptions(keys) for keys in group_by_concept(keys_ba.items(), n_concepts)]
    if 'passkeys' in cols_to_return:
        column_passkeys = [verbalise_descriptions(pkeys) for pkeys in group_by_concept(passkeys_ba.items(), n_concepts)]
    if 'proper_premises' in cols_to_return:
        column_proper_premises = [verbalise_descriptions(pps) for pps in
                                  group_by_concept(proper_premises_ba.items(), n_concepts)]
    if 'pseudo_intents' in cols_to_return:
        column_pseudo_intents = [verbalise_descriptions(pis) for pis in
                                 group_by_concept(pseudo_intents_ba.items(), n_concepts)]
    if 'support' in cols_to_return:
        column_support = [extent.count() for extent in extents_ba]
    if 'delta_stability' in cols_to_return:
        column_delta_stability = [idxs.delta_stability_by_description(descr, attr_extents, extent_ba)
                                  for descr, extent_ba in zip(intents_ba, extents_ba)]

    for colname in ['super_concepts', 'sub_concepts', 'next_concepts', 'previous_concepts']:
        if colname in cols_to_return:
            locals()[f"column_{colname}"] = [set(ba.search(True)) for ba in locals()[colname]]

    locals_ = locals()
    return pd.DataFrame({f: locals_[f"column_{f}"] for f in cols_to_return}).rename_axis('concept_id')


def mine_implications(
        data: ContextType, basis_name: BASIS_NAME = 'Proper Premise',
        unit_base: bool = False,
        to_compute: Optional[Union[list[MINE_IMPLICATIONS_COLUMN], Literal['all']]] = 'all',
        return_every_computed_column: bool = False,
        min_support: Union[int, float] = 0,
        min_delta_stability: Union[int, float] = 0, n_stable_concepts: Optional[int] = None
) -> pd.DataFrame:
    """Compute an implication basis (i.e. a set of non-redundant implications) for the given data

    An implication is a pair A -> B meaning that
     whenever all attributes from A are found in a description, this description also contains every attribute from B.
    Here, set of attributes A is called a premise, set of attributes B is called a conclusion.

    Parameters
    ----------
    data: ContextType
        Data in a format, supported by Caspailleur.io module.
        For example, Pandas DataFrame with binary values,
        or list of lists of strings (where every list of strings represents an itemset).
    basis_name: BASIS_NAME
        The name of the basis two compute. The possible values are provided in caspailleur.BASIS_NAME variable
         and discussed in the Notes section below.
    unit_base: bool
        A flag whether to return the "unit" version of the basis,
         where every implication's conclusion consists of a single attribute.
        See the Notes section below for more examples.
    to_compute: list[MINE_IMPLICATIONS_COLUMN] or 'all'
        A list of characteristics to compute (by default, compute 'all')
        The list of all possible characteristics is defined in caspailleur.api.MINE_IMPLICATIONS_COLUMN value.
    return_every_computed_column: bool
        A flag whether to return every computed column or only the ones defined by `to_compute` parameter.
        For example, the computation of column 'support' requires the computation of column 'extent'.
        Thus, if you have asked for column 'support' you can get the column 'extent' ``for free''.
    min_support: int or float
        Minimal value of the support for an implication, i.e. how many objects it describes.
        Can be defined by an integer (so the number of objects) or by a float (so the percentage of objects).
    min_delta_stability: int or float
        Minimal delta stability of concepts, whose minimal/maximal descriptions serve as premises/conclusions
         of implications. Raising up the value of min_delta_stability might be useful for finding implications in
          big data, where finding all concepts is not feasible.
        The value can be provided as an integer (measuring delta-stability with the number of objects) or as a float
         (measuring delta-stability with the percentage of objects w.r.t. total number of objects in the data).
        NOTE: The use of min_delta_stability for finding implication was not extensively studied in the literature.
        The resulting set of implication should be valid, but might not reflect some important implications.
    n_stable_concepts: int or None
        Parameter analogous to min_delta_stability,  except that it sets up the maximal value of concepts
         with high delta stability, and not the threshold for minimal delta stability.
        Can be either an integer (representing maximal number of high-stability concepts) or None (for all concepts).

    Returns
    -------
    descriptions_df: pandas.DataFrame
        Pandas DataFrame where every row represents an implication, and every column represents its characteristics
        defined by `to_compute` parameter.

    Notes
    -----
    *Basis name*
    There are two implication bases that can be mined via Caspailleur.
    One is called "Canonical Direct" basis and sometimes referred to as "Proper Premise" basis or "Karell" basis.
    The other is called "Canonical" basis and sometimes referred to as "Duquenne-Guigues",
     "Minimum" or "Pseudo-Intent" basis.

    Canonical Direct basis is "direct",
    i.e. one can find the closure of a subset of attributes with one pass over the implications.
    Also, every premise of the Canonical Direct basis is a minimal description of all attributes
    in the conclusion of the premise. Thus, no subset of a premise implies the same conclusion.

    Canonical basis contains the smallest amount of implication possible.
    Thus, it can be considered as the most dense and the most concise way to represent the data.
    However, such basis is not "direct"
    and some premises of this basis are not minimal subsets of attributes corresponding to their conclusions.

    Note that the Caspailleur's algorithm for computing the bases are not proven to be State-of-the-Art,
    as they were never extensively tested versus the other algorithms.
    But, they work fast enough for daily occasions.


    *Unit base*
    Unit and non-unit bases are two ways of representing the same set of implications.
    In the unit base, every implication's conclusion consist of a single attribute,
     but there can be many implications with the same premise.
    In the non-unit base, every implication's premise is unique, but there can be many attributes in the conclusion.

    For example, non unit-base:
    {a} -> {b, c}
    {d, e} -> {f}

    and the equivalent unit base:
    {a} -> b
    {a} -> c
    {d, e} -> f
    """
    assert basis_name in get_args(BASIS_NAME), \
        f"You asked for '{basis_name}' basis. But only the following bases are supported: {BASIS_NAME}"
    if basis_name in {'Canonical Direct', "Karell"}:
        basis_name = 'Proper Premise'
    if basis_name in {'Canonical', 'Duquenne-Guigues', 'Minimum'}:
        basis_name = 'Pseudo-Intent'

    dependencies = {
        "support": {'extent'},
        "delta_stability": {'extent'}
    }

    to_compute, cols_to_return = _setup_colnames_to_compute(
        MINE_IMPLICATIONS_COLUMN, to_compute, dependencies, return_all_computed=return_every_computed_column)

    ############################
    # Compute the (unit) basis #
    ############################
    bitarrays, objects, attributes = to_named_bitarrays(data)
    attr_extents = transpose_context(bitarrays)
    bitarrays, attr_extents = list(map(fbarray, bitarrays)), list(map(fbarray, attr_extents))

    n_objects = len(objects)
    min_support, min_delta_stability = [to_absolute_number(v, n_objects) for v in [min_support, min_delta_stability]]
    compute_only_stable_concepts = min_delta_stability != 0 or n_stable_concepts is not None

    extents_ba, int_ext_map = None, None

    if not compute_only_stable_concepts:
        intents_ba = mec.list_intents_via_LCM(bitarrays, min_supp=min_support)
        intents_ba = topological_sorting(intents_ba)[0]
        keys_ba = mec.list_keys(intents_ba)
    else:  # min_delta_stability > 0, so searching for only some stable concepts
        stable_extents = list(mec.list_stable_extents_via_gsofia(
            attr_extents, n_objects, min_delta_stability,
            n_stable_extents=n_stable_concepts, min_supp=min_support, n_attributes=len(attr_extents)))
        intents_ba = [intention(extent, attr_extents) for extent in stable_extents]
        int_ext_map = dict(zip(intents_ba, stable_extents))
        intents_ba = topological_sorting(intents_ba)[0]
        extents_ba = [int_ext_map[intent] for intent in intents_ba]
        keys_ba = mec.list_keys_for_extents(extents_ba, attr_extents)

    ppremises_ba = ibases.iter_proper_premises_via_keys(
        intents_ba, keys_ba,
        all_frequent_keys_provided=not compute_only_stable_concepts)
    ppremises_ba = sorted(ppremises_ba, key=lambda prem_intent: (prem_intent[1], prem_intent[0].count()))
    basis = ppremises_ba
    if basis_name == 'Pseudo-Intent':
        basis = ibases.list_pseudo_intents_via_keys(ppremises_ba, intents_ba)

    # compute pseudo-closures to reduce the conclusions by attributes implied by other implications
    if 'conclusion' in to_compute:
        subset_implied = [ibases.subset_saturate(premise, basis[:premise_i], intents_ba)
                          for premise_i, (premise, _) in enumerate(basis)]
        basis = [(premise, intents_ba[intent_i] & ~implied, intent_i)
                 for (premise, intent_i), implied in zip(basis, subset_implied)]
    else:
        basis = [(premise, None, intent_i) for premise, intent_i in basis]

    if unit_base and 'conclusion' in to_compute:
        single_attrs = to_named_bitarrays([{i} for i in range(len(attributes))])[0]
        basis = [(premise, single_attrs[attr_i], intent_i)
                 for premise, conclusion, intent_i in basis for attr_i in conclusion.search(True)]
    premises, conclusions, intents_idxs = zip(*basis)

    ####################################
    # Prepare values for the dataframe #
    ####################################
    def verbalise_descriptions(bas):
        return [verbalise(ba, attributes) for ba in bas]

    if 'premise' in to_compute:
        column_premise = verbalise_descriptions(premises)
    if 'conclusion' in to_compute:
        column_conclusion = verbalise_descriptions(conclusions)
        if unit_base:
            column_conclusion = [list(conclusion)[0] for conclusion in column_conclusion]
    if 'conclusion_full' in to_compute:
        column_conclusion_full = verbalise_descriptions(map(intents_ba.__getitem__, intents_idxs))
    if "extent" in to_compute and extents_ba is None:
        extents_ba = [fbarray(extension(intent, attr_extents)) for intent in intents_ba]
        int_ext_map = dict(zip(intents_ba, extents_ba))
    if "extent" in cols_to_return:
        column_extent = [verbalise(int_ext_map[intents_ba[intent_i]], objects) for intent_i in intents_idxs]
    if "support" in cols_to_return:
        column_support = [int_ext_map[intents_ba[intent_i]].count() for intent_i in intents_idxs]
    if 'delta_stability' in cols_to_return:
        column_delta_stability = [
            idxs.delta_stability_by_description(intents_ba[intent_i], attr_extents, int_ext_map[intents_ba[intent_i]])
            for intent_i in intents_idxs
        ]

    locals_ = locals()
    return pd.DataFrame({f: locals_[f"column_{f}"] for f in cols_to_return}).rename_axis('implication_id')
