import numpy as np
import perceval as pcvl
import pytest
from qiskit import QuantumCircuit

from qlass.compiler import HardwareConfig, ResourceAwareCompiler, compile, generate_report


@pytest.fixture
def bell_circuit():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    return qc


@pytest.fixture
def chip_config():
    return HardwareConfig(
        photon_loss_component_db=0.05, fusion_success_prob=0.11, hom_visibility=0.95
    )


def test_compile_returns_processor_with_input_state(bell_circuit):
    processor = compile(bell_circuit)
    assert isinstance(processor, pcvl.Processor)
    assert processor.m >= 2 * bell_circuit.num_qubits  # dual-rail plus possible ancillas


def test_resource_aware_compiler_report_contents(bell_circuit, chip_config):
    """The analysis report must contain physically sensible estimates, not just exist.

    Regression test for issue #217: the single previous test only checked that
    the report attribute was a dict, so the loss and success-probability
    estimates could regress arbitrarily with CI staying green.
    """
    compiler = ResourceAwareCompiler(config=chip_config)
    processor = compiler.compile(bell_circuit)

    report = processor.analysis_report
    assert report is compiler.analysis_report  # also exposed on the compiler

    # Component counts: the compiled Bell circuit contains beam splitters and
    # mode permutations, and every counted category adds up to the total.
    counts = report["component_count"]
    assert counts["BeamSplitter"] > 0
    assert counts["Permutation"] > 0, "PERM components must be counted (issue #217)"
    assert counts["Total"] == (
        counts["PhaseShifter"] + counts["BeamSplitter"] + counts["Permutation"] + counts["Other"]
    )

    assert report["num_modes"] == processor.m

    # Loss estimation: strictly positive for a non-empty circuit, and the
    # total is the sum of its parts.
    loss = report["loss_estimation_db"]
    assert loss["total_circuit_loss"] > 0
    assert np.isclose(loss["total_circuit_loss"], loss["component_loss"] + loss["waveguide_loss"])
    # Survival probability must be consistent with the dB loss figure.
    probs = report["probability_estimation"]
    assert np.isclose(probs["photon_survival_prob"], 10 ** (-loss["total_circuit_loss"] / 10))

    # Probabilities are probabilities.
    assert 0 < probs["photon_survival_prob"] <= 1
    assert 0 < probs["overall_success_prob"] <= 1
    for p in probs["details"].values():
        assert 0 < p <= 1
    # The overall estimate is the product of its factors.
    d = probs["details"]
    assert np.isclose(
        probs["overall_success_prob"],
        d["source_success"]
        * probs["photon_survival_prob"]
        * d["fusion_success"]
        * d["detector_success"],
    )


def test_resource_aware_compiler_counts_loss_from_all_components(bell_circuit, chip_config):
    """Total loss scales with ALL physical components, PERM included."""
    compiler = ResourceAwareCompiler(config=chip_config)
    compiler.compile(bell_circuit)
    counts = compiler.analysis_report["component_count"]
    component_loss = compiler.analysis_report["loss_estimation_db"]["component_loss"]

    assert np.isclose(component_loss, counts["Total"] * chip_config.photon_loss_component_db)


def test_fusion_probability_depends_on_cnot_count(chip_config):
    """No CNOTs -> no fusion penalty; with CNOTs the penalty kicks in."""
    compiler = ResourceAwareCompiler(config=chip_config)

    single_qubit = QuantumCircuit(1)
    single_qubit.h(0)
    compiler.compile(single_qubit)
    assert compiler.analysis_report["probability_estimation"]["details"]["fusion_success"] == 1.0

    entangling = QuantumCircuit(2)
    entangling.h(0)
    entangling.cx(0, 1)
    compiler.compile(entangling)
    fusion = compiler.analysis_report["probability_estimation"]["details"]["fusion_success"]
    assert np.isclose(fusion, chip_config.fusion_success_prob * chip_config.hom_visibility)


def test_generate_report_prints_summary(bell_circuit, chip_config, capsys):
    compiler = ResourceAwareCompiler(config=chip_config)
    compiler.compile(bell_circuit)

    generate_report(compiler.analysis_report)

    out = capsys.readouterr().out
    assert "Component Count" in out
    assert "Estimated Circuit Loss" in out
    assert "Overall Success Probability" in out


def test_hardware_config_defaults_and_overrides():
    default = HardwareConfig()
    assert 0 < default.source_efficiency <= 1
    assert 0 < default.detector_efficiency <= 1
    assert default.photon_loss_component_db >= 0

    custom = HardwareConfig(brightness=0.09, transmittance=0.4, phase_error=0.02)
    assert custom.brightness == 0.09
    assert custom.transmittance == 0.4
    assert custom.phase_error == 0.02
    # Untouched fields keep their defaults
    assert custom.fusion_success_prob == default.fusion_success_prob
