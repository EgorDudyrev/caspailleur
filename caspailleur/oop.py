from dataclasses import dataclass
from functools import reduce
from typing import TypeVar, Hashable, Self, TextIO, Union, Iterable

import pandas as pd
from bitarray import bitarray

import io

TObject = TypeVar("TObject", bound=Hashable)
TAttribute = TypeVar("TAttribute", bound=Hashable)


@dataclass
class FormalContext:
    objects: set[TObject]
    attributes: set[TAttribute]
    incidence: set[tuple[TObject, TAttribute]]

    @classmethod
    def from_pandas(cls, df: pd.DataFrame) -> Self:
        objects = set(df.index)
        attributes = set(df.columns)
        incidence = {(g, m) for g in objects for m in attributes if df.at[g, m]}
        return cls(objects, attributes, incidence)

    def to_pandas(self) -> pd.DataFrame:
        data = {g: {(g, m) in self.incidence for m in self.attributes} for g in self.objects}
        return pd.DataFrame(data)

    @classmethod
    def from_named_itemsets(cls, data: dict[TObject, Iterable[TAttribute]]) -> Self:
        objects = set(data)
        attributes = set(reduce(set.union, map(set, data.values()), set()))
        incidence = {(g, m) for g, itemset in data.items() for m in itemset}
        return cls(objects, attributes, incidence)

    def to_named_itemsets(self) -> dict[TObject, set[TAttribute]]:
        return {g: {m for m in self.attributes if (g, m) in self.incidence} for g in self.objects}

    @classmethod
    def from_itemsets(cls, data: Iterable[Iterable[TAttribute]]) -> Self:
        objects, attributes, incidence = set(), set(), set()
        for object_idx, itemset in enumerate(data):
            itemset = set(itemset)
            objects.add(object_idx)
            attributes |= itemset
            incidence |= {(object_idx, m) for m in itemset}
        return cls(objects, attributes, incidence)

    def to_itemsets(self) -> list[set[TAttribute]]:
        return [{m for m in self.attributes if (g, m) in self.incidence} for g in self.objects]

    @classmethod
    def from_itemsets_ba(cls, data: Iterable[bitarray]) -> Self:
        objects, attributes, incidence = set(), set(), set()
        for object_idx, itemset_ba in enumerate(data):
            itemset = set(itemset_ba.search(True))
            objects.add(object_idx)
            attributes |= itemset
            incidence |= {(object_idx, m) for m in itemset}
        return cls(objects, attributes, incidence)

    def to_itemsets_ba(self) -> list[bitarray]:
        attributes = list(self.attributes)
        bitarrays = []
        for g in self.objects:
            ba = bitarray([(g, m) in self.incidence for m in attributes])
            bitarrays.append(ba)
        return bitarrays

    @classmethod
    def from_attribute_extents_ba(cls, data: Iterable[bitarray]) -> Self:
        objects, attributes, incidence = None, set(), set()
        for attribute_idx, extent_ba in enumerate(data):
            objects = set(range(len(extent_ba))) if objects is None else objects
            attributes.add(attribute_idx)
            incidence |= {(g_idx, attribute_idx) for g_idx in extent_ba.search(True)}
        return cls(objects, attributes, incidence)

    def to_attribute_extents_ba(self) -> list[bitarray]:
        return [bitarray([(g, m) in self.incidence for g in self.objects]) for m in self.attributes]

    @classmethod
    def from_fca_repo(cls, context_name: str) -> Self:
        return cls.from_pandas(io.from_fca_repo(context_name)[0])

    @classmethod
    def read_csv(cls, filename: str, **kwargs) -> Self:
        """Read .csv file using Pandas read_csv function.

        Use `kwargs` to pass some parameters to this function if needed.
        """
        return cls.from_pandas(pd.DataFrame.read_csv(filename, **kwargs))

    def write_csv(self, file: Union[str, TextIO], **kwargs):
        """Write .csv file using Pandas to_csv function."""
        self.to_pandas().to_csv(file, **kwargs)

    @classmethod
    def read_cxt(cls, filename: str) -> Self:
        return cls.from_pandas(io.read_cxt(filename))

    def write_cxt(self, file: TextIO):
        return io.write_cxt(self.to_pandas(), file)





