import heapq
from collections.abc import Iterable, Iterator, Callable
from operator import itemgetter
from typing import TypeAlias
from functools import wraps

from bitarray import bitarray


TPatternMeasure: TypeAlias = Callable[[Iterable[int], list[bitarray]], float]

def project_context(attribute_extents: list[bitarray], n_first_attributes: int) -> list[bitarray]:
    """Select the subcontext containing the first `n_first_attributes`"""
    return attribute_extents[:n_first_attributes]


def project_itemset(itemset: Iterable[int], n_first_attributes: int) -> Iterator[int]:
    """Restrict the values in `itemset` to the `n_first_attributes`

    That is, remove all attributes whose index is greater than or equal to `n_first_attributes`.
    """
    return (item for item in itemset if item < n_first_attributes)


def project_measure(measure: TPatternMeasure, n_first_attributes: int) -> TPatternMeasure:
    """Restrict the pattern measure to the `n_first_attributes`

    That is, discard all attributes whose index is greater than or equal to `n_first_attributes`.
    """
    @wraps(measure)
    def wrapper(itemset, attribute_extents):
        return measure(project_itemset(itemset, n_first_attributes), project_context(attribute_extents, n_first_attributes))
    return wrapper


def betaSofia(
        attribute_extents: list[bitarray], measure: TPatternMeasure, threshold: float, n_best_descriptions: int = None
) -> dict[tuple[int,...], float]:
    """Mine subsets of attributes whose `measure` value is above the `threshold`

    The output is sound and complete when `measure` is projection-antimonotone and `n_best_descriptions` is `None`.
    Otherwise, the result is still sound, but not necessarily complete:
        some good subsets of attributes might have been lost along the way.

    The `measure` function takes two arguments:
        1) an iterable of attribute indices, and
        2) list if bitarrays, representing binary columns in the data.
    IMPORTANT: the `measure` should always return some high value (an upper bound) when called upon
        empty description and empty list of attribute_extents: `measure_max = measure(tuple(), list())`.
    """
    try:
        upper_bound = measure(tuple(), [])
    except Exception as e:
        raise ValueError("The `measure` should output its upper bound for the empty input: `measure(tuple(), [])`. "
                         f"Current output for the empty input is: {e}")
    assert threshold <= upper_bound, ("The output of `measure(tuple(), [])` is above the `threshold`, "
                                      "but is supposed to be the upper bound for any other measure value.")

    descriptions: dict[tuple[int,...], float] = {tuple(): upper_bound}
    for projection_i in range(1, len(attribute_extents)):
        projected_context = project_context(attribute_extents, projection_i)

        for old_description in list(descriptions):
            value = measure(old_description, projected_context)
            if value >= threshold:
                descriptions[old_description] = value
            else:
                del descriptions[old_description]

            new_description = old_description + (projection_i,)
            value = measure(new_description, projected_context)
            if value >= threshold:
                descriptions[new_description] = value

        if n_best_descriptions is not None and len(descriptions) > n_best_descriptions:
            descriptions: dict[tuple[int,...], float] = dict(heapq.nlargest(n_best_descriptions, descriptions.items(), key=itemgetter(1)))

    return descriptions
