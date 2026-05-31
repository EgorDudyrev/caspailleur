from collections.abc import Hashable, Callable, Iterable, Iterator
from typing import TypeVar, Protocol


T = TypeVar('T', bound=Hashable)
class ClosureIteratorProtocol(Protocol):
    def __call__(
            self, elements: set[T], closure_func: Callable[[Iterable[T]], set[T]],
            /, antimonotone_constraint_func: Callable[[Iterable[T]], bool] = None,
            **kwargs
    ) -> Iterator[set[T]]:
        ...

CLOSURE_ITERATOR_REGISTRY: dict[str, ClosureIteratorProtocol] = dict()

def register_closure_iterator(key: str):
    def decorator(func):
        assert key not in CLOSURE_ITERATOR_REGISTRY
        CLOSURE_ITERATOR_REGISTRY[key] = func
        return func

    return decorator



IMPLICATIONAL_BACKEND_REGISTRY: dict[str, type['ImplicationalSystemBackend']] = dict()

def register_implicational_backend(key: str):
    def decorator(cls):
        assert key not in IMPLICATIONAL_BACKEND_REGISTRY
        IMPLICATIONAL_BACKEND_REGISTRY[key] = cls
        return cls

    return decorator


class LineLayoutProtocol(Protocol):
    def __call__(
            self, nodes: set[T], edges: set[tuple[T, T]],
            **kwargs
    ) -> dict[T, tuple[float, float]]:
        ...

LINE_LAYOUT_REGISTRY: dict[str, LineLayoutProtocol] = dict()

def register_line_layout(key: str):
    def decorator(func):
        assert key not in LINE_LAYOUT_REGISTRY
        LINE_LAYOUT_REGISTRY[key] = func
        return func

    return decorator
