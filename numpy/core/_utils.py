import warnings


def _raise_warning(attr: str) -> None:

    warnings.warn(
        f"`numpy.core` has been made officially private and renamed to "
        "`numpy._core`. Accessing `_core` directly is discouraged, as "
        "members of `_core` can change without any warning at any time. "
        "You should use a public module instead that exports the attribute "
        "in question. If you still would like to access an attribute "
        f"from it, please use `np._core.{attr}`.",
        DeprecationWarning, 
        stacklevel=3
    )
