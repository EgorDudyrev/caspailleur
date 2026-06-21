from collections.abc import Hashable, Callable, Iterable, Iterator
from typing import TypeVar, Protocol

from caspailleur.algorithms.implication_bases import CLOSURE_ITERATOR_REGISTRY, register_closure_iterator
from caspailleur.classes.implicational_backends import IMPLICATIONAL_BACKEND_REGISTRY, register_implicational_backend
from caspailleur.algorithms.layouts import LINE_LAYOUT_REGISTRY, register_line_layout
