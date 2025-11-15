from qlass.compiler import ResourceAwareCompiler, HardwareConfig
from qiskit import QuantumCircuit

def test_resource_aware_compiler():
    """
    Test the ResourceAwareCompiler with a simple circuit and a hypothetical hardware configuration.
    """

    # Define a simple quantum circuit
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)

    # Define a hypothetical hardware configuration
    chip_config = HardwareConfig(
        photon_loss_component_db=0.05,
        fusion_success_prob=0.11,
        hom_visibility=0.95
    )

    # Compile the circuit using the resource-aware compiler
    compiler = ResourceAwareCompiler(config=chip_config)
    processor = compiler.compile(qc)

    # Check if the analysis report is generated correctly
    assert hasattr(processor, 'analysis_report'), "Processor should have an analysis report."
    assert isinstance(processor.analysis_report, dict), "Analysis report should be a dictionary."
