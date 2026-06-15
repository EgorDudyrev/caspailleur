from typing import overload, Iterable
from bitarray import bitarray
from bitarray.util import count_and, subset as basubset

from caspailleur.algorithms.base_functions import isets2bas, extension
from caspailleur.classes.formal_context import TObject, TAttribute, FormalContext


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
        positive_size = len(target)
    else: # if context is a list of columns represented with bitarrays
        extent = extension(description, context)
        tp = count_and(extent, target)
        fp = count_and(extent, ~target)
        positive_size = target.count()
    return 2 * tp / (tp + fp + positive_size)


@overload
def wracc_score(description: Iterable[TAttribute], context: FormalContext, target: set[TObject]) -> float: ...
@overload
def wracc_score(description: Iterable[int], context: list[bitarray], target: bitarray) -> float: ...
def wracc_score(description, context, target):
    freq = frequency(description, context)
    relative_acc = precision(description, context, target) - precision([], context, target)
    return freq * relative_acc


@overload
def false_positives_projection_lbound(description: Iterable[TAttribute], context: FormalContext, target: set[TObject]) -> float: ...
@overload
def false_positives_projection_lbound(description: Iterable[int], context: list[bitarray], target: bitarray) -> float: ...
def false_positives_projection_lbound(description, context, target):
    if isinstance(context, FormalContext):
        objects_order = list(context.objects)
        attributes_order = list(context.attributes)

        target = next(isets2bas([[objects_order.index(object_) for object_ in target]], len(context.objects)))
        description = {attributes_order.index(m) for m in description}
        context = context.to_attribute_extents_ba(objects_order=objects_order, attributes_order=attributes_order)

    description = set(description)
    extent = extension(description, context)
    subextents = (extent & context[attr] for attr in range(len(context)) if attr not in description)# and count_and(extent, context[attr]) < extent.count())
    greatest_subextents = sorted(subextents, key=lambda ext: ext.count(), reverse=True)
    if not greatest_subextents:
        return count_and(extent, ~target) #0
    i = 0
    while i < len(greatest_subextents):
        subextent = greatest_subextents[i]
        greatest_subextents[i+1:] = [other for other in greatest_subextents[i+1:] if not basubset(other, subextent)]
        i += 1
    return min(count_and(subextent, ~target) for subextent in greatest_subextents)


@overload
def true_positives_projection_lbound(description: Iterable[TAttribute], context: FormalContext, target: set[TObject]) -> float: ...
@overload
def true_positives_projection_lbound(description: Iterable[int], context: list[bitarray], target: bitarray) -> float: ...
def true_positives_projection_lbound(description, context, target):
    if isinstance(context, FormalContext):
        objects_order = list(context.objects)
        attributes_order = list(context.attributes)

        target = next(isets2bas([[objects_order.index(object_) for object_ in target]], len(context.objects)))
        description = {attributes_order.index(m) for m in description}
        context = context.to_attribute_extents_ba(objects_order=objects_order, attributes_order=attributes_order)

    description = set(description)
    extent = extension(description, context)
    subextents = (extent & context[attr] for attr in range(len(context)) if attr not in description)# and count_and(extent, context[attr]) < extent.count())
    greatest_subextents = sorted(subextents, key=lambda ext: ext.count(), reverse=True)
    if not greatest_subextents:
        return count_and(extent, target) #0
    i = 0
    while i < len(greatest_subextents):
        subextent = greatest_subextents[i]
        greatest_subextents[i+1:] = [other for other in greatest_subextents[i+1:] if not basubset(other, subextent)]
        i += 1
    return min(count_and(subextent, target) for subextent in greatest_subextents)


@overload
def f1_score_projection_ubound(description: Iterable[TAttribute], context: FormalContext, target: set[TObject]) -> float: ...
@overload
def f1_score_projection_ubound(description: Iterable[int], context: list[bitarray], target: bitarray) -> float: ...
def f1_score_projection_ubound(description, context, target):
    tp_ubound = true_positives(description, context, target)
    fp_lbound = false_positives_projection_lbound(description, context, target)
    pos_size = len(target) if isinstance(target, set) else target.count()
    return 2 * tp_ubound / (tp_ubound + fp_lbound + pos_size)


@overload
def f1_score_ubound_level(description: Iterable[TAttribute], context: FormalContext, target: set[TObject], threshold: float) -> float: ...
@overload
def f1_score_ubound_level(description: Iterable[int], context: list[bitarray], target: bitarray, threshold: float) -> float: ...
def f1_score_ubound_level(description, context, target, threshold):
    return f1_score_projection_ubound(description, context, target) >= threshold


@overload
def wracc_score_projection_ubound(description: Iterable[TAttribute], context: FormalContext, target: set[TObject]) -> float: ...
@overload
def wracc_score_projection_ubound(description: Iterable[int], context: list[bitarray], target: bitarray) -> float: ...
def wracc_score_projection_ubound(description, context, target):
    tp_ubound = true_positives(description, context, target)
    fp_lbound = false_positives_projection_lbound(description, context, target)
    pos_size = len(target) if isinstance(target, set) else target.count()
    context_size = len(context.objects) if isinstance(context, FormalContext) else len(target)

    support_bound = tp_ubound + fp_lbound
    freq_bound_ubound = support_bound / context_size
    relative_acc_ubound = tp_ubound/support_bound - pos_size/context_size
    return freq_bound_ubound * relative_acc_ubound


@overload
def wracc_score_ubound_level(description: Iterable[TAttribute], context: FormalContext, target: set[TObject], threshold: float) -> float: ...
@overload
def wracc_score_ubound_level(description: Iterable[int], context: list[bitarray], target: bitarray, threshold: float) -> float: ...
def wracc_score_ubound_level(description, context, target, threshold):
    return wracc_score_projection_ubound(description, context, target) >= threshold
