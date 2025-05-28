from perceval.converters import QiskitConverter
from qiskit import transpile
from qiskit.circuit.library import TwoLocal

import perceval as pcvl
import numpy as np
from perceval.components import Processor
from perceval.utils.matrix import Matrix as PercevalMatrix # For type hinting and conversion
from qlass.utils.utils import rotate_qubits

qiskit_converter = QiskitConverter(backend_name="Naive") #or SLOS

def le_ansatz(lp: np.ndarray, pauli_string: str) -> Processor:
    """
    Creates Perceval quantum processor for the Linear Entangled Ansatz.
    This ansatz consists of a layer of parametrized rotations, followed by
    a layer of CNOT gates, and finally another layer of parametrized rotations.

    Args:
        lp (np.ndarray): Array of parameter values
        pauli_string (str): Pauli string

    Returns:
        Processor: The quantum circuit as a Perceval processor
    """
    num_qubits = len(pauli_string)
    ansatz = TwoLocal(num_qubits, 'ry', 'cx', reps=1)

    ansatz_assigned = ansatz.assign_parameters(lp)
    ansatz_transpiled = transpile(ansatz_assigned, basis_gates=['u3', 'cx'], optimization_level=3)

    ansatz_rot = rotate_qubits(pauli_string, ansatz_transpiled.copy())
    processor = qiskit_converter.convert(ansatz_rot, use_postselection=True)
    processor.with_input(pcvl.LogicalState([0]*num_qubits))

    return processor


def custom_unitary_ansatz(*le_ansatz_args, U: np.ndarray, **le_ansatz_kwargs) -> Processor:
    """
    Implements a custom unitary ansatz using a provided unitary matrix.
    Accepts same arguments as le_ansatz, plus U. Returns a Perceval processor.
    """
    if not isinstance(U, np.ndarray):
        raise TypeError("Unitary matrix U must be a NumPy ndarray.")
    if U.ndim != 2 or U.shape[0] != U.shape[1]:
        raise ValueError("Unitary matrix U must be a square 2D array.")
    num_modes = U.shape[0]
    if 'num_modes' in le_ansatz_kwargs and le_ansatz_kwargs['num_modes'] != num_modes:
        raise ValueError(
            f"The 'num_modes' argument ({le_ansatz_kwargs['num_modes']}) provided "
            f"via le_ansatz_kwargs does not match the dimension of U ({num_modes})."
        )
    identity_matrix = np.eye(num_modes, dtype=complex if np.iscomplexobj(U) else float)
    if not np.allclose(U @ U.conj().T, identity_matrix, atol=1e-8) or \
       not np.allclose(U.conj().T @ U, identity_matrix, atol=1e-8):
        raise ValueError("The provided matrix U is not unitary.")

    try:
        perceval_U_obj = U if isinstance(U, PercevalMatrix) else PercevalMatrix(U)
    except Exception as e:
        print(f"Warning: Could not convert U to perceval.utils.matrix.Matrix: {e}. Using raw np.ndarray.")
        perceval_U_obj = U

    unitary_circuit_component = pcvl.Unitary(perceval_U_obj)
    backend_name = "SLOS"
    try:
        # When a single component is passed to Processor, it often gets wrapped in a list under 'components'
        processor = pcvl.Processor(backend_name, unitary_circuit_component)
    except Exception as e:
        raise RuntimeError(f"Failed to instantiate pcvl.Processor: {e}") from e
    return processor


def test_custom_unitary_ansatz():
    print("Demonstrating custom_unitary_ansatz function:")
    print("=" * 40)

    H_matrix = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
    print("\nExample 1: 2-mode Hadamard unitary")
    try:
        hadamard_processor = custom_unitary_ansatz(U=H_matrix)
        print(f"Successfully created a Perceval processor.")
        print(f"  Number of modes (processor.m): {hadamard_processor.m}")

        unitary_matrix_to_print = None
        print_method = "unknown"

        if hasattr(hadamard_processor, 'components') and \
           isinstance(hadamard_processor.components, list) and \
           len(hadamard_processor.components) > 0:

            first_component_in_processor = hadamard_processor.components[0]

            # In some Perceval versions, components might be (port_range, actual_component)
            actual_component_obj = None
            if isinstance(first_component_in_processor, tuple) and len(first_component_in_processor) == 2:
                actual_component_obj = first_component_in_processor[1]
            elif isinstance(first_component_in_processor, pcvl.ACircuit): # pcvl.Unitary is an ACircuit
                actual_component_obj = first_component_in_processor

            if actual_component_obj is not None and isinstance(actual_component_obj, pcvl.Unitary):
                if hasattr(actual_component_obj, 'U'):
                    matrix_data = actual_component_obj.U
                    if isinstance(matrix_data, PercevalMatrix):
                        unitary_matrix_to_print = np.array(matrix_data, dtype=complex)
                    elif isinstance(matrix_data, np.ndarray):
                        unitary_matrix_to_print = matrix_data

                    if unitary_matrix_to_print is not None:
                        if isinstance(first_component_in_processor, tuple):
                             print_method = "processor.components[0][1].U"
                        else:
                             print_method = "processor.components[0].U"

        if unitary_matrix_to_print is not None:
            print(f"  Processor's effective unitary matrix (via {print_method}): \n{np.round(unitary_matrix_to_print.astype(complex), decimals=5)}")
        else:
            print(f"  Could not retrieve the unitary matrix for display using the expected 'components' attribute structure.")

    except Exception as e:
        print(f"  An error occurred in Example 1: {e}")
        import traceback
        traceback.print_exc()
    print("-" * 40)

    # Example 2
    omega = np.exp(2 * np.pi * 1j / 3)
    DFT_matrix = (1 / np.sqrt(3)) * np.array([[1, 1, 1], [1, omega, omega**2], [1, omega**2, omega**4]], dtype=complex)
    print("\nExample 2: 3-mode DFT unitary")
    try:
        dft_processor = custom_unitary_ansatz(U=DFT_matrix, num_modes=3) # Pass num_modes for consistency check
        print(f"Successfully created a Perceval processor.")
        print(f"  Number of modes (processor.m): {dft_processor.m}")

        # Retrieve and print matrix for Example 2 (similar logic as Example 1)
        unitary_matrix_to_print_ex2 = None
        print_method_ex2 = "unknown"
        if hasattr(dft_processor, 'components') and \
           isinstance(dft_processor.components, list) and \
           len(dft_processor.components) > 0:

            first_component_in_processor_ex2 = dft_processor.components[0]
            actual_component_obj_ex2 = None
            if isinstance(first_component_in_processor_ex2, tuple) and len(first_component_in_processor_ex2) == 2:
                actual_component_obj_ex2 = first_component_in_processor_ex2[1]
            elif isinstance(first_component_in_processor_ex2, pcvl.ACircuit):
                actual_component_obj_ex2 = first_component_in_processor_ex2

            if actual_component_obj_ex2 is not None and isinstance(actual_component_obj_ex2, pcvl.Unitary):
                if hasattr(actual_component_obj_ex2, 'U'):
                    matrix_data_ex2 = actual_component_obj_ex2.U
                    if isinstance(matrix_data_ex2, PercevalMatrix):
                        unitary_matrix_to_print_ex2 = np.array(matrix_data_ex2, dtype=complex)
                    elif isinstance(matrix_data_ex2, np.ndarray):
                        unitary_matrix_to_print_ex2 = matrix_data_ex2

                    if unitary_matrix_to_print_ex2 is not None:
                        if isinstance(first_component_in_processor_ex2, tuple):
                            print_method_ex2 = "processor.components[0][1].U"
                        else:
                            print_method_ex2 = "processor.components[0].U"

        if unitary_matrix_to_print_ex2 is not None:
            print(f"  Processor's effective unitary matrix (via {print_method_ex2}): \n{np.round(unitary_matrix_to_print_ex2.astype(complex), decimals=5)}")
        else:
            print(f"  Could not retrieve the unitary matrix for display for Example 2.")

    except Exception as e: # Catch errors during dft_processor creation or initial matrix retrieval
        print(f"  An error occurred during initial setup of Example 2: {e}")

    # Test for inconsistent num_modes separately to report the expected ValueError clearly
    print("\n  Testing inconsistent 'num_modes' (expecting ValueError):")
    try:
        inconsistent_processor = custom_unitary_ansatz(U=DFT_matrix, num_modes=4)
    except ValueError as ve:
        print(f"  Caught expected ValueError: {ve}")
    except Exception as e: # Catch any other unexpected error during this specific test
        print(f"  An unexpected error occurred during inconsistent num_modes test: {e}")
    print("-" * 40)

    # Examples 3 and 4 remain the same
    print("\nExample 3: Non-unitary matrix (expecting ValueError)")
    non_unitary_matrix = np.array([[1, 2], [3, 4]], dtype=complex)
    try:
        processor_fail_unitary = custom_unitary_ansatz(U=non_unitary_matrix)
    except ValueError as ve:
        print(f"  Caught expected ValueError: {ve}")
    print("-" * 40)

    print("\nExample 4: Non-square matrix (expecting ValueError)")
    non_square_matrix = np.array([[1, 2, 3], [4, 5, 6]], dtype=complex)
    try:
        processor_fail_square = custom_unitary_ansatz(U=non_square_matrix)
    except ValueError as ve:
        print(f"  Caught expected ValueError: {ve}")
    print("=" * 40)
