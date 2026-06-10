import numpy as np
import perceval as pcvl
import pytest

from qlass.compiler import HardwareConfig
from qlass.mitigation import ZNEMitigator, fold_global_interferometer, scale_loss_config


def test_zne_linear_extrapolation_from_noisy_two_mode_executor():
    params = np.array([0.25])
    ideal_expectation = float(np.cos(params[0]))
    noise_slope = 0.04
    calls = []

    def noisy_two_mode_executor(params, noise_scale):
        calls.append(noise_scale)
        return ideal_expectation + noise_slope * noise_scale

    mitigator = ZNEMitigator(
        noisy_two_mode_executor,
        scaling_factors=[1.0, 3.0, 5.0],
        extrapolation_method="linear",
    )

    assert np.isclose(mitigator.mitigate(params), ideal_expectation)
    assert calls == [1.0, 3.0, 5.0]


def test_zne_polynomial_extrapolation_from_noisy_two_mode_executor():
    params = np.array([0.4])
    ideal_expectation = float(np.sin(params[0]))

    def noisy_two_mode_executor(params, noise_scale):
        return ideal_expectation + 0.03 * noise_scale + 0.01 * noise_scale**2

    mitigator = ZNEMitigator(
        noisy_two_mode_executor,
        scaling_factors=[1.0, 2.0, 3.0],
        extrapolation_method="polynomial",
    )

    assert np.isclose(mitigator.mitigate(params), ideal_expectation)


def test_zne_mitigates_noisy_perceval_linear_optical_energy():
    circuit = pcvl.Circuit(2).add((0, 1), pcvl.BS.H())
    input_state = pcvl.BasicState([1, 0])
    loss_db = 0.15

    def noisy_perceval_energy(params, noise_scale):
        transmittance = 10 ** (-(loss_db * noise_scale) / 10)
        processor = pcvl.Processor(
            "SLOS",
            circuit.copy(),
            noise=pcvl.NoiseModel(transmittance=transmittance),
        )
        processor.with_input(input_state)
        processor.min_detected_photons_filter(0)
        probabilities = processor.probs()["results"]
        return -sum(state[0] * probability for state, probability in probabilities.items())

    params = np.array([])
    ideal_energy = noisy_perceval_energy(params, noise_scale=0.0)
    unmitigated_energy = noisy_perceval_energy(params, noise_scale=1.0)
    mitigator = ZNEMitigator(
        noisy_perceval_energy,
        scaling_factors=[1.0, 2.0, 3.0],
        extrapolation_method="exponential",
    )

    mitigated_energy = mitigator.mitigate(params)

    assert abs(mitigated_energy - ideal_energy) < abs(unmitigated_energy - ideal_energy)
    assert np.isclose(mitigated_energy, ideal_energy)


def test_global_interferometer_folding_preserves_unitary():
    circuit = pcvl.Circuit(2) // pcvl.BS()

    folded = fold_global_interferometer(circuit, scaling_factor=3)

    assert len(list(folded)) == 3
    assert np.allclose(folded.compute_unitary(), circuit.compute_unitary())


def test_global_interferometer_folding_factor_one_returns_copy():
    circuit = pcvl.Circuit(2) // pcvl.BS()

    folded = fold_global_interferometer(circuit, scaling_factor=1)

    assert folded is not circuit
    assert len(list(folded)) == len(list(circuit))
    assert np.allclose(folded.compute_unitary(), circuit.compute_unitary())


def test_global_interferometer_folding_rejects_non_circuit():
    with pytest.raises(TypeError, match="Perceval Circuit"):
        fold_global_interferometer(object(), scaling_factor=3)


def test_global_interferometer_folding_rejects_even_scaling_factor():
    circuit = pcvl.Circuit(2) // pcvl.BS()

    with pytest.raises(ValueError, match="odd integer"):
        fold_global_interferometer(circuit, scaling_factor=2)


def test_scale_loss_config_scales_only_photon_loss_terms():
    config = HardwareConfig(
        photon_loss_component_db=0.05,
        photon_loss_waveguide_db_per_cm=0.3,
        source_efficiency=0.8,
        detector_efficiency=0.7,
    )

    scaled_config = scale_loss_config(config, scaling_factor=3.0)

    assert np.isclose(scaled_config.photon_loss_component_db, 0.15)
    assert np.isclose(scaled_config.photon_loss_waveguide_db_per_cm, 0.9)
    assert scaled_config.source_efficiency == config.source_efficiency
    assert scaled_config.detector_efficiency == config.detector_efficiency
    assert config.photon_loss_component_db == 0.05
    assert config.photon_loss_waveguide_db_per_cm == 0.3


def test_scale_loss_config_rejects_subunit_scaling_factor():
    with pytest.raises(ValueError, match="greater than or equal to 1"):
        scale_loss_config(HardwareConfig(), scaling_factor=0.5)


def test_zne_rejects_mismatched_expectation_values():
    def executor(params, noise_scale):
        return 1.0

    mitigator = ZNEMitigator(executor, scaling_factors=[1.0, 3.0])

    with pytest.raises(ValueError, match="match the number"):
        mitigator.extrapolate([1.0])


def test_zne_rejects_non_finite_expectation_values():
    def executor(params, noise_scale):
        return 1.0

    mitigator = ZNEMitigator(executor, scaling_factors=[1.0, 3.0])

    with pytest.raises(ValueError, match="finite real numbers"):
        mitigator.extrapolate([1.0, np.nan])


def test_zne_exponential_extrapolation():
    def executor(params, noise_scale):
        return 1.0

    mitigator = ZNEMitigator(
        executor,
        scaling_factors=[1.0, 2.0, 3.0],
        extrapolation_method="exponential",
    )
    ideal_expectation = 0.8
    expectation_values = [
        ideal_expectation * np.exp(-0.2 * scaling_factor)
        for scaling_factor in mitigator.scaling_factors
    ]

    assert np.isclose(mitigator.extrapolate(expectation_values), ideal_expectation)


def test_zne_exponential_extrapolation_rejects_invalid_values():
    def executor(params, noise_scale):
        return 1.0

    mitigator = ZNEMitigator(
        executor,
        scaling_factors=[1.0, 2.0, 3.0],
        extrapolation_method="exponential",
    )

    with pytest.raises(ValueError, match="nonzero"):
        mitigator.extrapolate([1.0, 0.0, 0.2])

    with pytest.raises(ValueError, match="same sign"):
        mitigator.extrapolate([1.0, -0.5, 0.2])


def test_zne_rejects_invalid_configuration():
    def executor(params, noise_scale):
        return 1.0

    with pytest.raises(ValueError, match="at least two"):
        ZNEMitigator(executor, scaling_factors=[1.0])

    with pytest.raises(ValueError, match="unique"):
        ZNEMitigator(executor, scaling_factors=[1.0, 1.0])

    with pytest.raises(ValueError, match="Invalid extrapolation_method"):
        ZNEMitigator(executor, scaling_factors=[1.0, 3.0], extrapolation_method="cubic")

    with pytest.raises(ValueError, match="finite real numbers"):
        ZNEMitigator(executor, scaling_factors=[1.0, np.inf])
