from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Any

import numpy as np
import perceval as pcvl

from qlass.compiler import HardwareConfig


def _validate_scaling_factor(scaling_factor: float) -> float:
    factor = float(scaling_factor)
    if not np.isfinite(factor):
        raise ValueError("Noise scaling factors must be finite real numbers.")
    if factor < 1:
        raise ValueError("Noise scaling factors must be greater than or equal to 1.")
    return factor


def _validate_global_folding_factor(scaling_factor: float) -> int:
    _validate_scaling_factor(scaling_factor)
    rounded_factor = round(scaling_factor)
    if not np.isclose(scaling_factor, rounded_factor) or rounded_factor % 2 == 0:
        raise ValueError("Global folding requires an odd integer scaling factor.")
    return int(rounded_factor)


def fold_global_interferometer(
    circuit: pcvl.Circuit,
    scaling_factor: float,
) -> pcvl.Circuit:
    """
    Fold a deterministic linear optical circuit as ``U (U† U)^n``.

    The folding factor must be an odd integer ``lambda = 2n + 1``. The original
    circuit is copied first, then global ``U†`` and ``U`` blocks are appended for
    each fold. This keeps the effective unitary unchanged while increasing the
    deterministic interferometer depth before any post-selection layer.

    Args:
        circuit: Perceval linear optical circuit to fold.
        scaling_factor: Odd integer noise scaling factor.

    Returns:
        Folded Perceval circuit with the same effective unitary.
    """
    if not isinstance(circuit, pcvl.Circuit):
        raise TypeError("fold_global_interferometer expects a Perceval Circuit.")

    odd_factor = _validate_global_folding_factor(scaling_factor)
    num_folds = (odd_factor - 1) // 2
    folded = circuit.copy()

    if num_folds == 0:
        return folded

    unitary = np.array(circuit.compute_unitary(), dtype=complex)
    unitary_dagger = unitary.conj().T

    for _ in range(num_folds):
        folded = folded // pcvl.Unitary(pcvl.Matrix(unitary_dagger), name="ZNE_Udg")
        folded = folded // pcvl.Unitary(pcvl.Matrix(unitary), name="ZNE_U")

    return folded


def scale_loss_config(config: HardwareConfig, scaling_factor: float) -> HardwareConfig:
    """
    Return a copy of ``config`` with photonic loss rates scaled in dB.

    Only the component loss and waveguide loss fields are scaled. Source efficiency,
    detector efficiency, visibility, and other hardware parameters are preserved.

    Args:
        config: Baseline hardware configuration.
        scaling_factor: Multiplicative loss scaling factor.

    Returns:
        New hardware configuration with scaled loss rates.
    """
    factor = _validate_scaling_factor(scaling_factor)
    return replace(
        config,
        photon_loss_component_db=config.photon_loss_component_db * factor,
        photon_loss_waveguide_db_per_cm=config.photon_loss_waveguide_db_per_cm * factor,
    )


class ZNEMitigator:
    """
    Zero Noise Extrapolation helper for expectation-value executors.

    The wrapped executor is called once per scaling factor. It must return an
    expectation value and accept the noise scale through the keyword configured by
    ``scale_keyword``.
    """

    def __init__(
        self,
        executor: Callable[..., float],
        scaling_factors: list[float],
        extrapolation_method: str = "polynomial",
        polynomial_degree: int = 2,
        scale_keyword: str = "noise_scale",
    ) -> None:
        """
        Initialize a ZNE mitigator.

        Args:
            executor: Callable returning an expectation value for a requested noise scale.
            scaling_factors: Noise scaling factors used for the extrapolation.
            extrapolation_method: One of ``"linear"``, ``"polynomial"``, or ``"exponential"``.
            polynomial_degree: Degree used by polynomial extrapolation.
            scale_keyword: Keyword used to pass each scaling factor to ``executor``.
        """
        if len(scaling_factors) < 2:
            raise ValueError("ZNE requires at least two scaling factors.")
        if len(set(scaling_factors)) != len(scaling_factors):
            raise ValueError("Scaling factors must be unique.")

        self.scaling_factors = [_validate_scaling_factor(factor) for factor in scaling_factors]
        self.executor = executor
        self.extrapolation_method = extrapolation_method.lower()
        self.polynomial_degree = polynomial_degree
        self.scale_keyword = scale_keyword

        if self.extrapolation_method not in {"linear", "polynomial", "exponential"}:
            raise ValueError(
                "Invalid extrapolation_method. Use 'linear', 'polynomial', or 'exponential'."
            )
        if polynomial_degree < 1:
            raise ValueError("polynomial_degree must be at least 1.")

    def scaled_expectation_values(
        self,
        params: np.ndarray,
        *executor_args: Any,
        **executor_kwargs: Any,
    ) -> list[float]:
        """
        Execute the wrapped expectation-value executor at each noise scale.

        Args:
            params: Variational parameters passed to the executor.
            *executor_args: Additional positional arguments forwarded to the executor.
            **executor_kwargs: Additional keyword arguments forwarded to the executor.

        Returns:
            Expectation values ordered like ``scaling_factors``.
        """
        values = []
        for scaling_factor in self.scaling_factors:
            call_kwargs = dict(executor_kwargs)
            call_kwargs[self.scale_keyword] = scaling_factor
            values.append(float(self.executor(params, *executor_args, **call_kwargs)))
        return values

    def extrapolate(self, expectation_values: Sequence[float]) -> float:
        """
        Extrapolate expectation values to the zero-noise limit.

        Args:
            expectation_values: Expectation values ordered like ``scaling_factors``.

        Returns:
            Extrapolated zero-noise expectation value.
        """
        if len(expectation_values) != len(self.scaling_factors):
            raise ValueError("Expectation values must match the number of scaling factors.")

        x = np.array(self.scaling_factors, dtype=float)
        y = np.array(expectation_values, dtype=float)
        if not np.all(np.isfinite(y)):
            raise ValueError("Expectation values must be finite real numbers.")

        if self.extrapolation_method == "linear":
            return float(np.polyval(np.polyfit(x, y, 1), 0.0))

        if self.extrapolation_method == "polynomial":
            degree = min(self.polynomial_degree, len(self.scaling_factors) - 1)
            return float(np.polyval(np.polyfit(x, y, degree), 0.0))

        return self._extrapolate_exponential(x, y)

    def mitigate(
        self,
        params: np.ndarray,
        *executor_args: Any,
        **executor_kwargs: Any,
    ) -> float:
        """
        Run scaled executions and return the zero-noise extrapolated expectation value.
        """
        values = self.scaled_expectation_values(params, *executor_args, **executor_kwargs)
        return self.extrapolate(values)

    @staticmethod
    def _extrapolate_exponential(
        scaling_factors: np.ndarray,
        expectation_values: np.ndarray,
    ) -> float:
        if np.any(np.isclose(expectation_values, 0.0)):
            raise ValueError("Exponential extrapolation requires nonzero expectation values.")

        sign = np.sign(expectation_values[0])
        if not np.all(np.sign(expectation_values) == sign):
            raise ValueError("Exponential extrapolation requires values with the same sign.")

        log_values = np.log(sign * expectation_values)
        coefficients = np.polyfit(scaling_factors, log_values, 1)
        return float(sign * np.exp(np.polyval(coefficients, 0.0)))
