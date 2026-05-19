from typing import overload, Iterable
from bitarray import bitarray
from bitarray.util import count_and

from caspailleur.oop import TObject, TAttribute, FormalContext
from caspailleur.base_functions import extension


@overload
def support(description: Iterable[TAttribute], context: FormalContext) -> int: ...
@overload
def support(description: Iterable[int], context: list[bitarray]) -> int: ...
def support(description, context) -> int:
    if isinstance(context, FormalContext):
        attributes_order = list(context.attributes)
        description = map(attributes_order.index, description)
        context = context.to_attribute_extents_ba(attributes_order=attributes_order)
    return extension(description, context).count()


@overload
def frequency(description: Iterable[TAttribute], context: FormalContext) -> float: ...
@overload
def frequency(description: Iterable[int], context: list[bitarray]) -> float: ...
def frequency(description, context) -> float:
    n_objects = len(context.objects) if isinstance(context, FormalContext) else len(context[0])
    return support(description, context) / n_objects


@overload
def delta_stability(description: Iterable[TAttribute], context: FormalContext) -> int: ...
@overload
def delta_stability(description: Iterable[int], context: bitarray) -> int: ...
def delta_stability(description, context) -> int:
    description = set(description)
    if isinstance(context, FormalContext):
        other_attributes = context.attributes - description
        max_subsupport = max((support(description|{attr}, context) for attr in other_attributes), default=0)
        return support(description, context) - max_subsupport
    # if context is a list of columns represented with bitarrays
    extent = extension(description, context)
    n_attributes = len(context)
    if len(description) < n_attributes:
        other_attributes = (m_idx for m_idx in range(n_attributes) if m_idx not in description)
        max_subsupport = max(count_and(extent, context[m_idx]) for m_idx in other_attributes)
    else:
        max_subsupport = 0
    return extent.count() - max_subsupport


@overload
def true_positives(description: Iterable[TAttribute], context: FormalContext, target: set[TObject]) -> int: ...
@overload
def true_positives(description: Iterable[int], context: list[bitarray], target: bitarray) -> int: ...
def true_positives(description, context, target) -> int:
    if isinstance(context, FormalContext):
        objects_order = list(context.objects)
        attributes_order = list(context.attributes)
        target = bitarray([g in target for g in objects_order])
        description = map(attributes_order.index, description)
        context = context.to_attribute_extents_ba(objects_order=objects_order, attributes_order=attributes_order)
    return count_and(extension(description, context), target)


@overload
def false_positives(description: Iterable[TAttribute], context: FormalContext, target: set[TObject]) -> int: ...
@overload
def false_positives(description: Iterable[int], context: list[bitarray], target: bitarray) -> int: ...
def false_positives(description, context, target) -> int:
    neg_target = context.objects - target if isinstance(context, FormalContext) else ~target
    return true_positives(description, context, neg_target)


@overload
def false_negatives(description: Iterable[TAttribute], context: FormalContext, target: set[TObject]) -> int: ...
@overload
def false_negatives(description: Iterable[int], context: list[bitarray], target: bitarray) -> int: ...
def false_negatives(description, context, target) -> int:
    n_positives = len(target) if isinstance(context, FormalContext) else target.count()
    return n_positives - true_positives(description, context, target)


@overload
def true_negatives(description: Iterable[TAttribute], context: FormalContext, target: set[TObject]) -> int: ...
@overload
def true_negatives(description: Iterable[int], context: list[bitarray], target: bitarray) -> int: ...
def true_negatives(description, context, target) -> int:
    n_negatives = len(context.objects)-len(target) if isinstance(context, FormalContext) else len(target)-target.count()
    return n_negatives - false_positives(description, context, target)


@overload
def precision(description: Iterable[TAttribute], context: FormalContext, target: set[TObject]) -> float: ...
@overload
def precision(description: Iterable[int], context: list[bitarray], target: bitarray) -> float: ...
def precision(description, context, target) -> float:
    if isinstance(context, FormalContext):
        support_ = support(description, context)
        return true_positives(description, context, target)/support_ if support_ else 0
    # if context is a list of columns represented with bitarrays
    extent = extension(description, context)
    return count_and(extent, target)/extent.count() if extent.any() else 0


@overload
def recall(description: Iterable[TAttribute], context: FormalContext, target: set[TObject]) -> float: ...
@overload
def recall(description: Iterable[int], context: list[bitarray], target: bitarray) -> float: ...
def recall(description, context, target):
    if isinstance(context, FormalContext):
        return true_positives(description, context, target)/len(target) if target else 0
    # if context is a list of columns represented with bitarrays
    extent = extension(description, context)
    return count_and(extent, target)/target.count() if target.any() else 0


@overload
def f1_score(description: Iterable[TAttribute], context: FormalContext, target: set[TObject]) -> float: ...
@overload
def f1_score(description: Iterable[int], context: list[bitarray], target: bitarray) -> float: ...
def f1_score(description, context, target):
    if isinstance(context, FormalContext):
        tp = true_positives(description, context, target)
        fp = false_positives(description, context, target)
        fn = false_negatives(description, context, target)
        return 2*tp/(tp+fp+fn)
    # if context is a list of columns represented with bitarrays
    extent = extension(description, context)
    return 2*count_and(extent, target)/(extent.count()+target.count())


@overload
def wracc_score(description: Iterable[TAttribute], context: FormalContext, target: set[TObject]) -> float: ...
@overload
def wracc_score(description: Iterable[int], context: list[bitarray], target: bitarray) -> float: ...
def wracc_score(description, context, target):
    freq = frequency(description, context)
    relative_acc = precision(description, context, target) - precision([], context, target)
    return freq * relative_acc
