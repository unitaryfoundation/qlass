"""Regression tests for issue #218: the user-facing API must be importable
from the top-level qlass namespace."""

import qlass


def test_top_level_exports_are_importable():
    for name in qlass.__all__:
        assert hasattr(qlass, name), f"qlass.__all__ lists '{name}' but it is not importable"


def test_headline_classes_at_top_level():
    from qlass import (  # noqa: F401
        VQE,
        M3Mitigator,
        PhotonicErrorModel,
        ZNEMitigator,
        compile_circuit,
        kerr_ansatz,
    )


def test_compile_circuit_is_alias_of_compile():
    from qlass import compile, compile_circuit

    assert compile_circuit is compile
