from qiskit import QuantumCircuit
from qlass.compiler import ResourceAwareCompiler, HardwareConfig, generate_report

# 1. Define a quantum circuit
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
print("Compiling a 2-qubit Bell State circuit...")

# 2. Define a hardware configuration
# A hypothetical chip with very low photon loss
chip_config = HardwareConfig(
    photon_loss_component_db=0.05,
    fusion_success_prob=0.11,
    hom_visibility=0.95
)

# 3. Compile with the resource-aware compiler for the chip
print("\n=== Compiling for the configured chip ===")
compiler_chip_config = ResourceAwareCompiler(config=chip_config)
processor_chip_config = compiler_chip_config.compile(qc)

# You can now access the report directly from the processor object
report = processor_chip_config.analysis_report
print(report)

# Or generate a human-readable report
print("\n=== Print a human-readable report ===")
generate_report(report)