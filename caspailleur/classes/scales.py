from collections.abc import Hashable, Iterable
from numbers import Number
from typing import TypeVar, Self, Literal

from caspailleur.classes.formal_context import FormalContext
from caspailleur.classes.poset import Poset

TObject = TypeVar('TObject', bound=Hashable)
TValue = TypeVar('TValue', bound=Hashable)
TAttribute = TypeVar('TAttribute', bound=Hashable)


class Scale:
    def __init__(self, context: FormalContext = None):
        self.context = context

    @classmethod
    def from_values(cls, values, scale_name=None) -> Self:
        raise NotImplementedError

    def fit(self, data: dict[TObject, TValue], scale_name: str = None) -> None:
        self.from_values(set(data.values()), scale_name)

    def transform(self, data: dict[TObject, TValue]) -> FormalContext:
        return FormalContext.from_named_itemsets({g: self.context.intent({v}) for g, v in data.items()})

    def fit_transform(self, data: dict[TObject, TValue]) -> FormalContext:
        self.fit(data)
        return self.transform(data)

    @staticmethod
    def _form_name(v, scale_name = None):
        return (scale_name, v) if scale_name is not None else v


class InterordinalScale(Scale):
    @classmethod
    def from_values(cls, values: Iterable[float], scale_name: str = None) -> Self:
        scaling_dict = dict()
        values = sorted(set(values))
        for object_value in values:
            attributes = []

            for attribute_value in values:
                if object_value <= attribute_value:
                    attributes.append(cls._form_name(f"<= {attribute_value}", scale_name))
                if object_value >= attribute_value:
                    attributes.append(cls._form_name(f">= {attribute_value}", scale_name))
            scaling_dict[cls._form_name(object_value, scale_name)] = attributes
        return cls(FormalContext.from_named_itemsets(scaling_dict))


class NominalScale(Scale):
    @classmethod
    def from_values(cls, values: Iterable[TValue], scale_name: str = None) -> Self:
        scaling_dict = dict()
        values = sorted(set(values))
        for object_value in values:
            scaling_dict[cls._form_name(object_value, scale_name)] = {cls._form_name(object_value, scale_name)}
        return cls(FormalContext.from_named_itemsets(scaling_dict))


class OrdinalScale(Scale):
    @classmethod
    def from_values(cls, values: Iterable[float], scale_name: str = None) -> Self:
        values = sorted(set(values))
        scaling_dict = {cls._form_name(obj_value, scale_name):
                            {cls._form_name(f"<= {v}", scale_name) for v in values if obj_value <= v}
                        for obj_value in values}
        return cls(FormalContext.from_named_itemsets(scaling_dict))

    @classmethod
    def from_poset(cls, poset: Poset, operation: Literal['>=', '<='] = '<=') -> Self:
        context = {element: poset.successors(element,) if operation == '<=' else poset.predecessors(element)
                   for element in poset}
        return cls(FormalContext.from_named_itemsets(context))


class BiordinalScale(Scale):
    @classmethod
    def from_values(cls, values: Iterable[float], scale_name: str = None, splitting_point: float = None) -> Self:
        values = sorted(set(values))
        if splitting_point is None:
            splitting_point = values[len(values) // 2]

        scaling_dict = {cls._form_name(obj_value, scale_name):
                        {cls._form_name(f"<= {v}", scale_name) for v in values if obj_value <= v <= splitting_point} |
                        {cls._form_name(f">= {v}", scale_name) for v in values if splitting_point < v <= obj_value}
                        for obj_value in values}
        return cls(FormalContext.from_named_itemsets(scaling_dict))



class PolygonScale(Scale):
    @classmethod
    def from_values(cls, values: list[tuple[float, float]], scale_name: str = None, precision: int = 2):
        import numpy as np

        datapoints = list(set((round(x, precision), round(y, precision)) for x, y in values))
        xmin, xmax = min(x for x, _ in datapoints), max(x for x, _ in datapoints)
        ymin, ymax = min(y for _, y in datapoints), max(y for _, y in datapoints)

        lines = set()
        lines |= {(0, ymin), (0, ymax), (np.inf, xmin), (np.inf, xmax)}
        for i, (x1, y1) in enumerate(datapoints):
            for (x2, y2) in datapoints[i + 1:]:
                if y1 == y2:
                    k, b = 0, y1
                elif x1 == x2:
                    k, b = np.inf, x1
                else:
                    k = (y1 - y2) / (x1 - x2) if x1 != x2 else np.inf
                    b = y1 - k * x1
                lines.add((k, b))
        lines = sorted(lines, key=lambda kb: (abs(kb[0]), abs(kb[1])), reverse=True)

        crosspoints = set()
        for i, (k1, b1) in enumerate(lines):
            for k2, b2 in lines[i + 1:]:
                # knowing that k1>=k2, and if k1=k2 then b1>=b2
                if k1 == k2:  # parallel lines
                    continue
                # from now on k1 != k2
                if k1 == np.inf:  # implies k2!=np.inf
                    crossx = b1
                else:  # if k2<k1<np.inf
                    crossx = (b2 - b1) / (k1 - k2)

                crossy = k2 * crossx + b2
                if crossx < xmin or crossx > xmax or crossy < ymin or crossy > ymax:
                    continue
                crosspoints.add((round(crossx, precision), round(crossy, precision)))


        X = np.array(list(set(datapoints) | set(crosspoints)))
        attribute_extents = dict()
        for (k, b) in lines:
            m = (k, b, '>=')
            if k == np.inf:
                flg = X[:, 0] >= b
            else:
                flg = (X[:, 1] - (k * X[:, 0] + b)).round(precision) >= 0
            attribute_extents[m] = set(map(tuple, X[flg]))

            m = (k, b, '<=')
            if k == np.inf:
                flg = X[:, 0] <= b
            else:
                flg = (X[:, 1] - k * X[:, 0] - b).round(precision) <= 0
            attribute_extents[m] = set(map(tuple, X[flg]))

        attribute_extents = {cls._form_name(m): {cls._form_name(g) for g in gs} for m, gs in attribute_extents.items()}
        return cls(FormalContext.from_named_itemsets(attribute_extents).T)

    def draw(self, ax=None, datapoints=None, line_kwargs: dict = None, datapoints_kwargs: dict = None,
             crosspoints_kwargs: dict = None):
        import numpy as np
        import matplotlib.pyplot as plt

        ax = plt.gca() if ax is None else ax
        datapoints = list(self.context.objects) if datapoints is None else datapoints
        datapoints_kwargs = dict() if datapoints_kwargs is None else dict(datapoints_kwargs)
        for k, v in dict(c='blue', alpha=1, zorder=2).items():
            if k not in datapoints_kwargs:
                datapoints_kwargs[k] = v
        crosspoints_kwargs = dict() if crosspoints_kwargs is None else dict(crosspoints_kwargs)
        for k, v in dict(c='gray', alpha=1, zorder=1).items():
            if k not in crosspoints_kwargs:
                crosspoints_kwargs[k] = v
        line_kwargs = dict(linestyle='--', c='gray', zorder=0) if line_kwargs is None else line_kwargs

        xmin, xmax = np.array(datapoints)[:, 0].min(), np.array(datapoints)[:, 0].max()
        ymin, ymax = np.array(datapoints)[:, 1].min(), np.array(datapoints)[:, 1].max()

        xlim_min, xlim_max = xmin - (xmax - xmin) * 0.05, xmax + (xmax - xmin) * 0.05
        ylim_min, ylim_max = ymin - (ymax - ymin) * 0.05, ymax + (ymax - ymin) * 0.05

        points = [point[1] if not isinstance(point[1], Number) and len(point[1])==2 else point for point in self.context.objects]  # clean up the scale name
        points = [(x, y) for x, y in points if xlim_min <= x <= xlim_max and ylim_min <= y <= ylim_max]
        crosspoints = list(set(points) - set(datapoints))
        datapoints = list(datapoints)
        lines = list({(k, b) for k, b, _ in self.context.attributes})

        for k, b in lines:
            if k == np.inf:
                ax.axvline(b, **line_kwargs)
            else:
                ax.plot([xlim_min, xlim_max], [k * xlim_min + b, k * xlim_max + b], **line_kwargs)

        if crosspoints:
            ax.scatter(np.array(crosspoints)[:, 0], np.array(crosspoints)[:, 1], **crosspoints_kwargs)
        ax.scatter(np.array(datapoints)[:, 0], np.array(datapoints)[:, 1], **datapoints_kwargs)

        ax.set_xlim((xlim_min, xlim_max))
        ax.set_ylim((ylim_min, ylim_max))
