from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Literal, Union

from caspailleur.classes.formal_context import TAttribute
from caspailleur.algorithms.base_functions import powerset


IMPLICATIONAL_REGISTRY: dict[str, type['ImplicationalSystemBackend']] = dict()


def register_implicational_backend(key: str):
    def decorator(cls):
        assert key not in IMPLICATIONAL_REGISTRY
        IMPLICATIONAL_REGISTRY[key] = cls
        return cls
    return decorator


class ImplicationalSystemBackend(ABC):
    @property
    @abstractmethod
    def implications(self) -> dict[frozenset[TAttribute], set[TAttribute]]:
        pass

    @abstractmethod
    def saturate(self, description: set[TAttribute]) -> set[TAttribute]:
        pass

    @abstractmethod
    def add(self, premise: Iterable[TAttribute], conclusion: Iterable[TAttribute]):
        ...

    @abstractmethod
    def remove(self, premise: Iterable[TAttribute], conclusion: Iterable[TAttribute]):
        pass

    def __call__(self, description: set[TAttribute]) -> set[TAttribute]:
        return self.saturate(description)

    def __contains__(self, item: set[TAttribute]) -> bool:
        return self.saturate(item) == item

    def __len__(self) -> int:
        return len(self.implications)

    def size(self) -> int:
        return sum(len(premise)+len(conclusion) for premise, conclusion in self.implications.items())

    @property
    def base_set(self) -> set[TAttribute]:
        return {attr for premise, conclusion in self.implications.items() for attr in premise | conclusion}

    def __iter__(self) -> Iterable[TAttribute]:
        return (description for description in powerset(self.base_set) if description in self)


@register_implicational_backend('Naive')
class NaiveImplicationalSystemBackend(ImplicationalSystemBackend):
    def __init__(self, implications: dict[frozenset[TAttribute], set[TAttribute]]):
        self._implications = dict(implications)

    @property
    def implications(self) -> dict[frozenset[TAttribute], set[TAttribute]]:
        return dict(self._implications)

    def saturate(self, description: set[TAttribute]) -> set[TAttribute]:
        closure = set(description)
        while True:
            closure_new = set(closure)
            for premise, conclusion in self._implications.items():
                if premise <= closure_new:
                    closure_new |= conclusion
            if closure == closure_new:
                break
            closure = closure_new
        return closure

    def add(self, premise: Iterable[TAttribute], conclusion: Iterable[TAttribute]):
        premise = frozenset(premise)
        if premise not in self._implications:
            self._implications[premise] = set()
        self._implications[premise] |= set(conclusion)

    def remove(self, premise: Iterable[TAttribute], conclusion: Iterable[TAttribute]):
        premise = frozenset(premise)
        if premise not in self._implications:
            return
        self._implications[premise] -= set(conclusion)


class ImplicationalSystem:
    def __init__(self, implications: dict[frozenset[TAttribute], set[TAttribute]], backend_class: Union[type[ImplicationalSystemBackend], Literal[tuple(IMPLICATIONAL_REGISTRY)]] = 'Naive'):
        backend_class = IMPLICATIONAL_REGISTRY[backend_class] if not isinstance(backend_class, type) else backend_class
        assert issubclass(backend_class, ImplicationalSystemBackend)
        self._backend = backend_class(implications)

    @property
    def implications(self) -> dict[frozenset[TAttribute], set[TAttribute]]:
        return self.backend.implications

    @property
    def backend(self) -> ImplicationalSystemBackend:
        return self._backend

    def saturate(self, description: set[TAttribute]) -> set[TAttribute]:
        return self.backend.saturate(description)

    def __call__(self, description: set[TAttribute]) -> set[TAttribute]:
        return self.backend(description)

    def __contains__(self, item: set[TAttribute]) -> bool:
        return self.backend.__contains__(item)

    def add(self, premise: Iterable[TAttribute], conclusion: Iterable[TAttribute]):
        return self.backend.add(premise, conclusion)

    def __len__(self) -> int:
        return self.backend.__len__()

    def size(self) -> int:
        return self.backend.size()

    @property
    def base_set(self) -> set[TAttribute]:
        return self.backend.base_set

    def __iter__(self) -> Iterable[TAttribute]:
        return self.backend.__iter__()
