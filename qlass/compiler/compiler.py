from typing import Optional, Union
from perceval.converters import QiskitConverter
from perceval.utils import NoiseModel
import perceval as pcvl
from qiskit import QuantumCircuit
from qlass.compiler.hardware_config import HardwareConfig
from perceval.components.unitary_components import PS, BS

def compile(circuit: QuantumCircuit, 
            backend_name: str = "Naive", 
            use_postselection: bool = True, 
            input_state: Optional[Union[pcvl.StateVector, pcvl.BasicState]] = None, 
            noise_model: NoiseModel = None) -> pcvl.Processor:
    """
    Convert a Qiskit quantum circuit to a Perceval processor.
    
    Args:
        circuit (QuantumCircuit): The Qiskit quantum circuit to convert
        backend_name (str): The backend to use for the Perceval processor
                           Options are: "Naive", "SLOS"
        use_postselection (bool): Whether to use postselection for the processor
        input_state (Optional[Union[pcvl.StateVector, pcvl.BasicState]]): 
                    The input state for the processor. If None, the |0...0> state is used.
        noise_model (NoiseModel): A perceval NoiseModel object representing the noise model
                                 for the processor.
    
    Returns:
        pcvl.Processor: The quantum circuit as a Perceval processor
    """
    # Initialize the Qiskit converter
    qiskit_converter = QiskitConverter(backend_name=backend_name, noise_model=noise_model)
    
    # Convert the circuit to a Perceval processor
    processor = qiskit_converter.convert(circuit, use_postselection=use_postselection)

    # Set the input state if provided, otherwise use the |0...0> state
    if input_state is None:
        processor.with_input(pcvl.LogicalState([0] * circuit.num_qubits))
    else:
        processor.with_input(input_state)
    
    return processor


class ResourceAwareCompiler:
    """
    A compiler that analyzes a quantum circuit against a hardware configuration
    to estimate its real-world performance and resource requirements.
    """
    def __init__(self, config: HardwareConfig):
        self.config = config
        self.noise_model = NoiseModel(transmittance=self.config.transmittance, phase_imprecision=self.config.phase_imprecision)
        self.qiskit_converter = QiskitConverter(backend_name="Naive", noise_model=self.noise_model)
        self.analysis_report: Dict[str, Any] = {}

    def _analyze(self, processor: pcvl.Processor, num_cnots: int) -> None:
        """Inspects the compiled Perceval processor and populates the analysis report."""
        # Get the underlying circuit
        circuit = processor.linear_circuit()

        # 1. Count components by iterating through the circuit
        num_ps = sum(1 for _, component in circuit if isinstance(component, PS))
        num_bs = sum(1 for _, component in circuit if isinstance(component, BS))
    
        num_components = num_ps + num_bs # Simplified count
        num_modes = processor.m  # Total number of modes in the circuit

        # 2. Estimate Photon Loss
        component_loss_db = num_components * self.config.photon_loss_component_db
        path_loss_db = num_components * self.config.avg_path_length_per_component_cm * self.config.photon_loss_waveguide_db_per_cm
        total_loss_db = component_loss_db + path_loss_db
        # Convert dB loss to survival probability
        # T = 10^(-dB/10)
        photon_survival_prob = 10**(-total_loss_db / 10)
        effective_fusion_prob_per_gate = self.config.fusion_success_prob * self.config.hom_visibility

        # 3. Estimate Overall Success Probability
        # This is a chain of probabilities
        source_prob = self.config.source_efficiency ** num_modes
        fusion_prob = (effective_fusion_prob_per_gate ** num_cnots) if num_cnots > 0 else 1.0
        detector_prob = self.config.detector_efficiency ** num_modes

        overall_success_prob = source_prob * photon_survival_prob * fusion_prob * detector_prob

        # 4. Populate the report
        self.analysis_report = {
            "num_modes": num_modes,
            "component_count": {
                "PhaseShifter": num_ps,
                "BeamSplitter": num_bs,
                "Total": num_components
            },
            "loss_estimation_db": {
                "component_loss": component_loss_db,
                "waveguide_loss": path_loss_db,
                "total_circuit_loss": total_loss_db
            },
            "probability_estimation": {
                "photon_survival_prob": photon_survival_prob,
                "overall_success_prob": overall_success_prob,
                "details": {
                    "source_success": source_prob,
                    "fusion_success": fusion_prob,
                    "detector_success": detector_prob
                }
            }
        }

    def compile(self, circuit: QuantumCircuit) -> pcvl.Processor:
        """Compiles a Qiskit circuit and runs a resource analysis."""
        processor = self.qiskit_converter.convert(circuit, use_postselection=True)
        processor.with_input(pcvl.LogicalState([0] * circuit.num_qubits))

        num_cnots = circuit.count_ops().get('cx', 0)

        self._analyze(processor, num_cnots)
        # self.generate_report() # Automatically print the report after compilation

        # Attach the report to the processor object for programmatic access
        setattr(processor, 'analysis_report', self.analysis_report)
        return processor

def generate_report(analysis_report) -> None:
    """Prints the analysis report in a human-readable format."""
    print("\n--- qlass Resource-Aware Compiler Report ---")

    report = analysis_report
    print(f"  Circuit modes: {report['num_modes']}")
    print(f"  Component Count: {report['component_count']['Total']} "
            f"(PS: {report['component_count']['PhaseShifter']}, BS: {report['component_count']['BeamSplitter']})")
    print("\n[Performance Estimation]")
    loss_db = report['loss_estimation_db']['total_circuit_loss']
    survival_prob = report['probability_estimation']['photon_survival_prob']
    print(f"  Estimated Circuit Loss: {loss_db:.2f} dB")
    print(f"  Photon Survival Probability (due to loss): {survival_prob:.2%}")
    
    overall_prob = report['probability_estimation']['overall_success_prob']
    print(f"  >> Estimated Overall Success Probability: {overall_prob:.4%}")
    print("     (This is the probability of getting a valid, post-selected result from a single shot)")
    print("-------------------------------------------\n")