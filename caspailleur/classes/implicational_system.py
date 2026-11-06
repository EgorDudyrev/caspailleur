from collections.abc import Iterable
from functools import reduce
from typing import Literal, Union, overload, Callable, Self, Optional

from tqdm.auto import tqdm

from caspailleur.algorithms.implication_bases import iter_proper_premises_via_keys, iter_proper_premises_for_closure
from caspailleur.algorithms.mine_equivalence_classes import list_intents_via_LCM, list_keys
from caspailleur.classes.formal_context import TAttribute, FormalContext
from caspailleur.classes.implicational_backends import ImplicationalSystemBackend
from caspailleur.registries import IMPLICATIONAL_BACKEND_REGISTRY, CLOSURE_ITERATOR_REGISTRY


class ImplicationalSystem:
    def __init__(
            self,
            implications: dict[frozenset[TAttribute], set[TAttribute]] = None,
            backend_class: Union[type[ImplicationalSystemBackend], Literal[tuple(IMPLICATIONAL_BACKEND_REGISTRY)]] = 'Naive',
            attributes_list: list[TAttribute] = None
    ):
        implications = implications if implications is not None else dict()
        self.backend = backend_class

        self._attributes_order, self._attribute_index_map = [], dict()
        attributes_order = list(attributes_list) if attributes_list is not None else []
        for attr in attributes_order:
            self._add_attribute(attr)

        for premise, conclusion in implications.items():
            self.add(premise, conclusion)

    @property
    def implications(self) -> dict[frozenset[TAttribute], set[TAttribute]]:
        return {frozenset(self._idxs2attrs(premise)): set(self._idxs2attrs(conclusion))
                for premise, conclusion in self.backend.implications.items()}

    @implications.setter
    def implications(self, value: dict[frozenset[TAttribute], set[TAttribute]]) -> None:
        self.clear()
        for premise, conclusion in value.items():
            self.add(premise, conclusion)

    @property
    def unit_implications(self) -> list[tuple[frozenset[TAttribute], TAttribute]]:
        unit_implications = []
        for premise_idxs, unit_conclusion_idx in self.backend.unit_implications:
            premise = frozenset(self._idxs2attrs(premise_idxs))
            unit_conclusion = next(iter(self._idxs2attrs([unit_conclusion_idx])))
            unit_implications.append((premise, unit_conclusion))
        return unit_implications

    @property
    def backend(self) -> ImplicationalSystemBackend:
        return self._backend

    @backend.setter
    @overload
    def backend(self, value: ImplicationalSystemBackend) -> None: ...

    @backend.setter
    @overload
    def backend(self, value: type[ImplicationalSystemBackend]) -> None: ...

    @backend.setter
    @overload
    def backend(self, value: Literal[tuple(IMPLICATIONAL_BACKEND_REGISTRY)]) -> None: ...

    @backend.setter
    def backend(self, value) -> None:
        existing_implications = self.backend.implications if hasattr(self, '_backend') else dict()

        if isinstance(value, ImplicationalSystemBackend):
            new_backend = value.__class__(existing_implications) if existing_implications else value
        elif isinstance(value, type) and issubclass(value, ImplicationalSystemBackend):
            new_backend = value(existing_implications)
        elif isinstance(value, str) and value in IMPLICATIONAL_BACKEND_REGISTRY:
            new_backend = IMPLICATIONAL_BACKEND_REGISTRY[value](existing_implications)
        else:
            raise ValueError(f"Implicational system backend {value} is not supported.")

        self._backend: ImplicationalSystemBackend = new_backend

    def reorder_attributes(self, attributes_order: list[TAttribute]) -> None:
        assert set(attributes_order) == set(self._attributes_order)
        implications = dict(self.implications)

        self.clear()
        for attr in attributes_order:
            self._add_attribute(attr)
        for premise, conclusion in implications.items():
            self.add(premise, conclusion)

    def saturate(self, description: set[TAttribute], single_pass: bool = False) -> set[TAttribute]:
        return self._idxs2attrs(self.backend.saturate(self._attrs2idxs(description), single_pass=single_pass))

    def __call__(self, description: set[TAttribute]) -> set[TAttribute]:
        return self._idxs2attrs(self.backend(self._attrs2idxs(description)))

    def __contains__(self, item: set[TAttribute]) -> bool:
        return self.backend.__contains__(self._attrs2idxs(item))

    def add(self, premise: Iterable[TAttribute], conclusion: Iterable[TAttribute]):
        premise, conclusion = set(premise), set(conclusion)
        for attr in premise | conclusion:
            self._add_attribute(attr)

        return self.backend.add(self._attrs2idxs(premise), self._attrs2idxs(conclusion))

    def remove(self, premise: Iterable[TAttribute], conclusion: Iterable[TAttribute]):
        premise, conclusion = set(premise), set(conclusion)
        for attr in premise | conclusion:
            self._add_attribute(attr)

        self.backend.remove(self._attrs2idxs(premise), self._attrs2idxs(conclusion))

        legacy_attributes = premise | conclusion
        for p, c in self.implications.items():
            legacy_attributes -= p | c
        for attr in legacy_attributes:
            self._remove_attribute(attr)

    def add_definition(self, premise: Iterable[TAttribute], conclusion: Iterable[TAttribute]):
        """Add two implications `premise` => `conclusion` and `conclusion` => `premise`"""
        self.add(premise, conclusion)
        self.add(conclusion, premise)

    def clear(self) -> None:
        self.backend.clear()
        self._attributes_order = []
        self._attribute_index_map = dict()

    def __len__(self) -> int:
        return self.backend.__len__()

    def size(self) -> int:
        return self.backend.size()

    @property
    def base_set(self) -> set[TAttribute]:
        return set(self._attributes_order)

    def iterate_closures(
            self,
            algorithm: Literal[tuple(CLOSURE_ITERATOR_REGISTRY)] = 'CbO-FW',
            antimonotone_constraint_func: Callable[[Iterable[TAttribute]], bool] = None,
            single_saturation_pass: bool = False,
    ) -> Iterable[set[TAttribute]]:
        if antimonotone_constraint_func is not None:
            indexed_antimonotone_constraint_func = lambda idxs: antimonotone_constraint_func(self._idxs2attrs(idxs))
        else:
            indexed_antimonotone_constraint_func = None
        closure_iterator = self.backend.iterate_closures(algorithm, antimonotone_constraint_func=indexed_antimonotone_constraint_func, single_saturation_pass=single_saturation_pass)
        return map(self._idxs2attrs, closure_iterator)

    def count_closures(
            self,
            use_tqdm: bool = False,
            iteration_algorithm: Literal[tuple(CLOSURE_ITERATOR_REGISTRY)] = 'CbO-FW',
            antimonotone_constraint_func: Callable[[Iterable[TAttribute]], bool] = None,
            single_saturation_pass: bool = False,
    ) -> int:
        return self.backend.count_closures(use_tqdm, iteration_algorithm, antimonotone_constraint_func=antimonotone_constraint_func, single_saturation_pass=single_saturation_pass)

    def _idxs2attrs(self, indices: Iterable[int]) -> set[TAttribute]:
        return {self._attributes_order[idx] for idx in indices}

    def _attrs2idxs(self, attributes: Iterable[TAttribute]) -> set[int]:
        return {self._attribute_index_map[attr] for attr in attributes}

    def _add_attribute(self, attr: TAttribute) -> None:
        if attr in self._attribute_index_map:
            return

        idx = len(self._attributes_order)
        self._attributes_order.append(attr)
        self._attribute_index_map[attr] = idx
        self.backend.add_attribute()

    def _remove_attribute(self, attr: TAttribute) -> None:
        if attr not in self._attribute_index_map:
            return

        idx = self._attribute_index_map[attr]
        self._attributes_order.pop(idx)
        self._attribute_index_map.pop(attr)
        self.backend.remove_attribute(idx)


    @classmethod
    def from_context(cls, context: FormalContext, min_support: Union[int, float] = 0) -> Self:
        min_support = int(min_support * len(context.objects)) if 0 < min_support < 1 else min_support
        obj_order, attrs_order = list(context.objects), list(context.attributes)
        intents_ba = list_intents_via_LCM(context.to_itemsets_ba(attributes_order=attrs_order), min_supp=min_support)
        keys_ba = list_keys(intents_ba)
        proper_premises_iterator = iter_proper_premises_via_keys(intents_ba, keys_ba)
        impsys = cls(implications={
            frozenset([attrs_order[m] for m in p.search(True)]):
                {attrs_order[m] for m in intents_ba[i].search(True)}
            for p, i in proper_premises_iterator}, attributes_list=attrs_order, backend_class='VerticalWild')
        return impsys

    def union(self, other: Self) -> Self:
        copy = self.copy()
        for p, c in other.implications.items():
            copy.add(p, c)
        return copy

    def __or__(self, other: Self) -> Self:
        return self.union(other)

    def minimal_generators(self, closure: set[TAttribute]) -> list[set[TAttribute]]:
        return [self._idxs2attrs(mingen) for mingen in self.backend.minimal_generators(self._attrs2idxs(closure))]

    def minimal_generator(self, closure: set[TAttribute]) -> Optional[set[TAttribute]]:
        mingen = None
        for mingen_idxs in self.backend.minimal_generators(self._attrs2idxs(closure)):
            if mingen is None:
                mingen = self._idxs2attrs(mingen_idxs)
            else:
                return None
        return mingen

    def shortest_generators(self, closure: set[TAttribute]) -> Iterable[set[TAttribute]]:
        return [self._idxs2attrs(shortgen) for shortgen in self.backend.shortest_generators(self._attrs2idxs(closure))]

    def shortest_generator(self, closure: set[TAttribute]) -> Optional[set[TAttribute]]:
        shortgen = None
        for shortgen_idxs in self.backend.shortest_generators(self._attrs2idxs(closure)):
            if shortgen is None:
                shortgen = self._idxs2attrs(shortgen_idxs)
            else:
                return None
        return shortgen

    def canonical_generators(self, closure: set[TAttribute]) -> Iterable[set[TAttribute]]:
        return [self._idxs2attrs(dgen) for dgen in self.backend.canonical_generators(self._attrs2idxs(closure))]

    def canonical_generator(self, closure: set[TAttribute]) -> Optional[set[TAttribute]]:
        canonical_generator = None
        for gen_idxs in self.backend.canonical_generators(self._attrs2idxs(closure)):
            if canonical_generator is None:
                canonical_generator = self._idxs2attrs(gen_idxs)
            else:
                return None
        return canonical_generator

    def shortest_canonical_generators(self, closure: set[int]) -> Iterable[set[int]]:
        return [self._idxs2attrs(dgen) for dgen in self.backend.shortest_canonical_generators(self._attrs2idxs(closure))]

    def shortest_canonical_generator(self, closure: set[TAttribute]) -> Optional[set[TAttribute]]:
        scgen = None
        for scgen_idxs in self.backend.shortest_canonical_generators(self._attrs2idxs(closure)):
            if scgen is None:
                scgen = self._idxs2attrs(scgen_idxs)
            else:
                return None
        return scgen

    def extremes(self, closure: set[TAttribute]) -> Iterable[TAttribute]:
        return {m for m in closure if self.saturate(closure-{m}) != closure}

    def minmax_generators(self, closure: set[TAttribute]) -> Iterable[set[TAttribute]]:
        return [self._idxs2attrs(gen) for gen in self.backend.minmax_generators(self._attrs2idxs(closure))]

    def minmax_generator(self, closure: set[TAttribute]) -> Optional[set[TAttribute]]:
        mmgen = None
        for mmgen_idxs in self.backend.minmax_generators(self._attrs2idxs(closure)):
            if mmgen is None:
                mmgen = self._idxs2attrs(mmgen_idxs)
            else:
                return None
        return mmgen

    def shortmax_generators(self, closure: set[TAttribute]) -> Iterable[set[TAttribute]]:
        return [self._idxs2attrs(gen) for gen in self.backend.shortmax_generators(self._attrs2idxs(closure))]

    def shortmax_generator(self, closure: set[TAttribute]) -> Optional[set[TAttribute]]:
        smgen = None
        for smgen_idxs in self.backend.shortmax_generators(self._attrs2idxs(closure)):
            if smgen is None:
                smgen = self._idxs2attrs(smgen_idxs)
            else:
                return None
        return smgen

    def concise_generators(self, closure: set[TAttribute]) -> Iterable[set[TAttribute]]:
        return self.shortmax_generators(closure)

    def concise_generator(self, closure: set[TAttribute]) -> Optional[set[TAttribute]]:
        return self.shortmax_generator(closure)

    def change_basis(
            self,
            basis_name: Literal['Proper Premise', 'ACDUB', 'Pseudo Intent', 'Proper Pattern Premise', 'Pattern ACDUB', 'Pseudo Pattern Intent'],
            background_system: "ConciseImplicationalSystem" = None) -> Self:
        if basis_name == 'Proper Premise':
            pp_iterator = iter_proper_premises_for_closure(set(self.base_set), self.saturate)
            return self.__class__(
                {p: self.saturate(p) - p for p in pp_iterator},
                backend_class=self.backend, attributes_list=list(self._attributes_order)
            )

        if basis_name == 'ACDUB':
            pp_basis = self.change_basis('Proper Premise')

            cduab_basis = ImplicationalSystem(attributes_list=list(pp_basis._attributes_order))
            for premise, conclusion in pp_basis.implications.items():
                subpremises = [premise - {m} for m in premise]
                for m in conclusion:
                    if not any(m in self.saturate(subpremise) for subpremise in subpremises):
                        cduab_basis.add(premise, {m})
            return cduab_basis

        if basis_name == 'Pseudo Intent':
            pseudo_intents = set()
            for premise, conclusion in self.implications.items():
                sub_system = ImplicationalSystem(
                    {p: c for p, c in self.implications.items() if c < conclusion},
                    attributes_list=list(self.base_set)
                )
                pi = frozenset(sub_system.saturate(premise))
                if pi != conclusion:
                    pseudo_intents.add(pi)
            return ImplicationalSystem({pi: self.saturate(pi) - pi for pi in pseudo_intents},
                                       attributes_list=list(self._attributes_order))

        if basis_name == 'Proper Pattern Premise':
            pp_basis = self.change_basis('Proper Premise')
            # assert background_system.is_concise_geometry()
            pattern_implications = dict()
            for premise, conclusion in tqdm(pp_basis.implications.items(), desc='filter background'):
                conclusion = conclusion | premise
                assert background_system.saturate(conclusion) == conclusion, \
                    f"{premise=} {conclusion=} {background_system.saturate(conclusion)-conclusion=}"
                premise_bsat = background_system.saturate(premise)
                subclosure = background_system.saturate(
                    reduce(set.union, [self.saturate(premise_bsat - {m}) for m in premise], premise_bsat))
                if subclosure == conclusion:
                    continue

                pattern_implications[frozenset(premise_bsat)] = conclusion

            return ImplicationalSystem(
                {frozenset(background_system.concise_generator(p)): background_system.concise_generator(c) - p
                 for p, c in tqdm(pattern_implications.items(), desc='turn concise')},
                attributes_list=list(self._attributes_order))

        if basis_name == 'Pattern ACDUB':
            cduab = self.change_basis('ACDUB')
            pattern_implications = dict()
            for premise, conclusion in tqdm(cduab.implications.items(), desc='filter background'):
                premise_bsat = background_system.saturate(premise)
                subclosure = background_system.saturate(
                    reduce(set.union, [self.saturate(premise_bsat - {m}) for m in premise], premise_bsat))
                conclusion_bits = set(conclusion) - set(subclosure)
                if conclusion_bits:
                    pattern_implications[frozenset(premise_bsat)] = background_system.saturate(conclusion_bits)

            return ImplicationalSystem(
                {frozenset(background_system.concise_generator(p)): background_system.concise_generator(c) - p
                 for p, c in tqdm(pattern_implications.items(), desc='turn concise')},
                attributes_list=list(self._attributes_order))

        if basis_name == 'Pseudo Pattern Intent':
            dg = self.change_basis('Pseudo Intent')
            pdg = ImplicationalSystem(attributes_list=list(self._attributes_order))
            pseudo_intents = sorted(dg.implications,
                                    key=lambda pi: (len(dg.saturate(pi)), len(background_system.saturate(pi))))
            for pintent in pseudo_intents:
                conclusion = dg.saturate(pintent)
                ppintent = pdg.saturate(background_system.saturate(pintent))
                if ppintent != conclusion:
                    pdg.add(ppintent, conclusion)

            return ImplicationalSystem(
                {frozenset(background_system.concise_generator(p)): background_system.concise_generator(c) - p
                 for p, c in tqdm(pdg.implications.items(), desc='turn concise')},
                attributes_list=list(self._attributes_order))

        raise ValueError('Unknown basis')
