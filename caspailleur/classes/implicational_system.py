from collections.abc import Iterable
from dataclasses import dataclass

from caspailleur.classes.formal_context import TAttribute
from caspailleur.algorithms.base_functions import powerset


@dataclass
class ImplicationalSystem:
    implications: dict[frozenset[TAttribute], set[frozenset[TAttribute]]]

    def saturate(self, description: set[TAttribute]) -> set[TAttribute]:
        closure = set(description)
        while True:
            closure_new = set(closure)
            for premise, conclusion in self.implications.items():
                if premise <= closure_new:
                    closure_new |= conclusion
            if closure == closure_new:
                break
            closure = closure_new
        return closure

    def __call__(self, description: set[TAttribute]) -> set[TAttribute]:
        return self.saturate(description)

    def __contains__(self, item: set[TAttribute]) -> bool:
        return self.saturate(item) == item

    def add(self, premise: Iterable[TAttribute], conclusion: Iterable[TAttribute]):
        premise = frozenset(premise)
        if premise not in self.implications:
            self.implications[premise] = set()
        self.implications[premise] |= set(conclusion)

    def __len__(self) -> int:
        return len(self.implications)

    def size(self) -> int:
        return sum(len(premise)+len(conclusion) for premise, conclusion in self.implications.items())

    @property
    def base_set(self) -> set[TAttribute]:
        return {attr for premise, conclusion in self.implications.items() for attr in premise | conclusion}

    def __iter__(self) -> Iterable[TAttribute]:
        return (description for description in powerset(self.base_set) if description in self)
