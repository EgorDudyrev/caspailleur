from abc import abstractmethod, ABCMeta
from collections.abc import Hashable
from typing import TypeVar, Self

from caspailleur.classes.formal_context import FormalContext


TObject = TypeVar('TObject', bound=Hashable)
TValue = TypeVar('TValue', bound=Hashable)
TAttribute = TypeVar('TAttribute', bound=Hashable)


class Scale(metaclass=ABCMeta):
    def __init__(self, scaling_dict: dict[TValue, set[TAttribute]] = None):
        self._scaling_dict = scaling_dict

    @classmethod
    @abstractmethod
    def from_values(cls, values, scale_name=None) -> Self:
        ...

    def fit(self, data: dict[TObject, TValue], scale_name: str = None) -> None:
        self.from_values(set(data.values()), scale_name)

    def transform(self, data: dict[TObject, TValue]) -> FormalContext:
        return FormalContext.from_named_itemsets({g: self._scaling_dict[v] for g, v in data.items()})

    def fit_transform(self, data: dict[TObject, TValue]) -> FormalContext:
        self.fit(data)
        return self.transform(data)

    @property
    def context(self) -> FormalContext:
        assert self._scaling_dict is not None
        return FormalContext.from_named_itemsets(self._scaling_dict)

    @staticmethod
    def _form_name(v, scale_name = None):
        return (scale_name, v) if scale_name is not None else v


class InterordinalScale(Scale):
    @classmethod
    def from_values(cls, values, scale_name: str = None) -> Self:
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

        return cls(scaling_dict=scaling_dict)


class NominalScale(Scale):
    @classmethod
    def from_values(cls, values, scale_name: str = None) -> Self:
        scaling_dict = dict()
        values = sorted(set(values))
        for object_value in values:
            scaling_dict[cls._form_name(object_value, scale_name)] = {cls._form_name(object_value, scale_name)}
        return cls(scaling_dict=scaling_dict)
