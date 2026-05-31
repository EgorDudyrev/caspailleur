import warnings
from collections.abc import Callable
from inspect import signature
from typing import Any


def filter_kwargs(
        given_func: Callable, given_kwarg_start_index: int, given_vals: dict[str, Any], custom_kwargs: set,
        target_func: Callable, target_kwarg_start_index: int) -> tuple[dict[str, Any], set[str], set[str]]:
    # TODO: Improve the support of expected named kwargs and kwargs passed via **
    expected_kwargs = set(list(signature(given_func).parameters)[given_kwarg_start_index:])
    defined_kwargs = expected_kwargs | set(custom_kwargs)
    supported_kwargs = set(list(signature(target_func).parameters)[target_kwarg_start_index:])
    not_supported_kwargs = defined_kwargs - supported_kwargs
    if not_supported_kwargs:
        warnings.warn(f"Passed arguments {not_supported_kwargs} are not supported by function '{target_func.__name__}' "
                      f"and thus will not affect its run.")
    kwargs_to_pass = {p: given_vals[p] for p in supported_kwargs if p in defined_kwargs}
    return kwargs_to_pass, defined_kwargs, supported_kwargs
