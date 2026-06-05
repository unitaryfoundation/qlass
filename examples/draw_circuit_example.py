"""
Example demonstrating circuit visualization with draw_circuit and VQE.draw_ansatz.
"""

import warnings
from pathlib import Path

import numpy as np
from perceval.algorithm import Sampler

from qlass.quantum_chemistry import LiH_hamiltonian
from qlass.utils import draw_circuit
from qlass.vqe import VQE, le_ansatz

warnings.simplefilter("ignore")
warnings.filterwarnings("ignore")


# Define an executor function that uses the linear entangled ansatz
def executor(params, pauli_string):
    """
    Executor function that creates a processor with the le_ansatz.
    """
    processor = le_ansatz(params, pauli_string)
    sampler = Sampler(processor)
    samples = sampler.samples(100)
    return samples


def main():
    # Number of qubits
    num_qubits = 2

    # Variational parameters and measurement basis
    params = np.array([0.1, 0.2, 0.3, 0.4])
    pauli_string = "II"

    # Output directory for saved figures
    output_dir = Path(__file__).resolve().parent / "circuit_outputs"
    output_dir.mkdir(exist_ok=True)

    # Build the linear entangled ansatz processor
    processor = le_ansatz(params, pauli_string)

    # Save circuit visualizations in multiple formats
    print("Saving circuit visualizations...")
    draw_circuit(
        processor,
        output_format="mpl",
        skin="phys",
        save_path=str(output_dir / "le_ansatz_phys.png"),
    )
    print(f"  {output_dir / 'le_ansatz_phys.png'}")

    draw_circuit(
        processor,
        output_format="html",
        skin="symb",
        save_path=str(output_dir / "le_ansatz_symb.html"),
    )
    print(f"  {output_dir / 'le_ansatz_symb.html'}")

    draw_circuit(
        processor,
        output_format="mpl",
        skin="debug",
        compact=True,
        save_path=str(output_dir / "le_ansatz_debug.png"),
    )
    print(f"  {output_dir / 'le_ansatz_debug.png'}")

    draw_circuit(
        processor,
        output_format="text",
        skin="symb",
        save_path=str(output_dir / "le_ansatz_symb.txt"),
    )
    print(f"  {output_dir / 'le_ansatz_symb.txt'}")

    # Display the circuit in the terminal
    print("\nText preview (symb skin):")
    draw_circuit(processor, output_format="text", skin="symb")

    # Generate a 2-qubit Hamiltonian for LiH
    hamiltonian = LiH_hamiltonian(num_electrons=2, num_orbitals=1)

    # Initialize the VQE solver with the custom executor
    vqe = VQE(
        hamiltonian=hamiltonian,
        executor=executor,
        num_params=2 * num_qubits,  # Number of parameters in the linear entangled ansatz
    )

    # Visualize the ansatz via VQE.draw_ansatz (pass ansatz_fn when executor returns samples)
    vqe_path = output_dir / "vqe_draw_ansatz.html"
    print(f"\nSaving VQE.draw_ansatz output to {vqe_path} ...")
    vqe.draw_ansatz(
        params,
        ansatz_fn=le_ansatz,
        output_format="html",
        skin="phys",
        save_path=str(vqe_path),
    )

    print(f"\nCircuit figures saved in {output_dir}")


if __name__ == "__main__":
    main()
