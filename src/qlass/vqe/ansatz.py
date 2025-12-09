import numpy as np
import perceval as pcvl
from perceval.components import Processor
from perceval.utils import NoiseModel
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import n_local
from qiskit.quantum_info import Statevector

from qlass.compiler import compile
from qlass.utils import rotate_qubits


def le_ansatz(
    lp: np.ndarray, pauli_string: str, noise_model: NoiseModel | None = None
) -> Processor:
    """
    Creates Perceval quantum processor for the Linear Entangled Ansatz.
    This ansatz consists of a layer of parametrized rotations, followed by
    a layer of CNOT gates, and finally another layer of parametrized rotations.

    Args:
        lp (np.ndarray): Array of parameter values
        pauli_string (str): Pauli string
        noise_model (NoiseModel): A perceval NoiseModel object representing the noise model

    Returns:
        Processor: The quantum circuit as a Perceval processor
    """
    num_qubits = len(pauli_string)
    ansatz = n_local(num_qubits, "ry", "cx", reps=1, entanglement="linear")

    ansatz_assigned = ansatz.assign_parameters(lp)
    ansatz_transpiled = transpile(ansatz_assigned, basis_gates=["u3", "cx"], optimization_level=3)

    ansatz_rot = rotate_qubits(pauli_string, ansatz_transpiled.copy())
    processor = compile(
        ansatz_rot, input_state=pcvl.LogicalState([0] * num_qubits), noise_model=noise_model
    )

    return processor


def custom_unitary_ansatz(
    lp: np.ndarray, pauli_string: str, U: np.ndarray, noise_model: NoiseModel | None = None
) -> Processor:
    """
    Creates Perceval quantum processor that directly implements a given unitary matrix.
    This function serves as a custom ansatz that bypasses circuit construction and instead
    loads a full unitary matrix representing the quantum operation.

    The unitary is embedded in a Qiskit QuantumCircuit, converted to Perceval format,
    and returned as a Processor object with post-selection enabled.

    Args:
        lp (np.ndarray): Placeholder array of parameter values (unused, for compatibility).
        pauli_string (str): Pauli string used to determine the number of qubits.
        U (np.ndarray): A unitary matrix of shape (2^n, 2^n) where n = len(pauli_string).
        noise_model (NoiseModel): A perceval NoiseModel object representing the noise model.

    Returns:
        Processor: The quantum circuit as a Perceval processor implementing unitary U.
    """
    num_qubits = len(pauli_string)
    dim = 2**num_qubits

    if U.shape != (dim, dim):
        raise ValueError(f"Expected unitary of shape ({dim}, {dim}), got {U.shape}")
    if not np.allclose(U @ U.conj().T, np.eye(dim), atol=1e-10):
        raise ValueError("Matrix U is not unitary")

    qc = QuantumCircuit(num_qubits)
    qc.unitary(U, range(num_qubits))

    qc_rotated = rotate_qubits(pauli_string, qc.copy())
    processor = compile(
        qc_rotated, input_state=pcvl.LogicalState([0] * num_qubits), noise_model=noise_model
    )
    processor.with_input(pcvl.LogicalState([0] * num_qubits))

    return processor


def list_of_ones(computational_basis_state: int, n_qubits: int) -> list[int]:
    """
    Return the indices of ones in the binary expansion of an integer, in big-endian order.

    Parameters
    ----------
    computational_basis_state : int
        Integer representing the computational basis state.
    n_qubits : int
        Total number of qubits (length of the binary string).

    Returns
    -------
    list of int
        List of indices where the binary representation has ones.
        Indices are in big-endian order. For example, for `n_qubits=6` and
        `computational_basis_state=22` (binary `010110`), the result is `[1, 3, 4]`.

    Notes
    -----
    The output index ordering corresponds to the big-endian representation of the qubits.
    """

    bitstring = format(computational_basis_state, "b").zfill(n_qubits)

    return [abs(j - n_qubits + 1) for j in range(len(bitstring)) if bitstring[j] == "1"]


def CSF_initial_states(
    num_spatial_orbitals: int,
    num_electrons: tuple[int, int],
    initial_parameters: np.ndarray,
    pauli_string: str,
    singlet_excitation: bool = True,
    k_index: int | None = None,
    l_index: int | None = None,
    noise_model: NoiseModel | None = None,
) -> Processor | list[Processor]:
    """
    Generate a Hartree-Fock initial state quantum circuit and optionally apply a singlet excitation.

    This function constructs a Hartree-Fock initial state for a given number of electrons
    and spatial orbitals, applies a parameterized ansatz, and prepares the circuit for
    simulation on a quantum processor. Optionally, it can include a singlet excitation
    between specified orbitals.

    Parameters
    ----------
    num_spatial_orbitals : int
        Number of spatial orbitals in the system.
    num_electrons : list of int
        Number of alpha and beta electrons, given as [num_alpha, num_beta].
    initial_parameters : np.ndarray
        Initial values for the parameterized ansatz.
    pauli_string : str
        Pauli string used to rotate the qubits after the ansatz.
    singlet_excitation : bool, optional
        If True, apply a singlet excitation to the Hartree-Fock state using the specified
        orbitals `i` and `j`. Default is False.
    k_index : int or None, optional
        Index of the occupied orbital for singlet excitation. Required if `singlet_excitation=True`.
    l_index : int or None, optional
        Index of the unoccupied orbital for singlet excitation. Required if `singlet_excitation=True`.
    noise_model : NoiseModel or None, optional
        Optional noise model for simulating the quantum processor. Default is None.

    Returns
    -------
    Processor or list of Processor
        - If `singlet_excitation=False`, returns a single `Processor` object representing the
          Hartree-Fock initial state with the ansatz applied.
        - If `singlet_excitation=True`, returns a list `[Processor_HF, Processor_SC]` containing
          the Hartree-Fock processor and the singlet-excited processor.

    Raises
    ------
    ValueError
        If `singlet_excitation=True` but either `k` or `l` is not provided.

    Notes
    -----
    - The Hartree-Fock state is created by initializing qubits corresponding to spin orbitals.
    - Do not use tampered Hamiltonian on this function.
    - The ansatz used is a `n_local` circuit with 'ry' rotations and linear 'cx' entanglement.
    - The circuit is transpiled using basis gates ['u3', 'cx'] and optimized to level 3.
    - The `rotate_qubits` function is applied to align with the specified Pauli string.
    - This function integrates with Perceval's `LogicalState` and `compile` functions to
      produce a processor-ready quantum circuit.
    """
    # Generating Hartree Fock initial state
    num_spin_orbitals = sum(num_electrons) * 2
    bitstring_alpha = [0] * (num_spin_orbitals // 2)
    bitstring_beta = [0] * (num_spin_orbitals // 2)
    for i in range(num_electrons[0]):
        bitstring_alpha[i] = 1
    for j in range(num_electrons[1]):
        bitstring_beta[j] = 1
    bitstring = bitstring_alpha + bitstring_beta
    bitstring = bitstring[::-1]
    num_qubits = len(bitstring)
    hfc = QuantumCircuit(num_qubits)
    # Apply X gates to occupied orbitals
    for i, b in enumerate(bitstring):
        if b:
            hfc.x(i)

    ansatz = n_local(
        num_qubits=num_qubits,
        rotation_blocks="ry",
        entanglement_blocks="cx",
        entanglement="linear",
        reps=1,
        insert_barriers=False,
    )

    hfc.compose(ansatz, inplace=True)
    ansatz_assigned = hfc.assign_parameters(initial_parameters)
    photonic_circuit = pcvl.LogicalState(bitstring)
    ansatz_transpiled = transpile(ansatz_assigned, basis_gates=["u3", "cx"], optimization_level=3)
    ansatz_rot = rotate_qubits(pauli_string, ansatz_transpiled.copy())
    processor_HF = compile(ansatz_rot, input_state=photonic_circuit, noise_model=noise_model)

    # singlet state
    processor_sc = None
    if singlet_excitation:
        if k_index is None or l_index is None:
            raise ValueError(
                "Singlet excitation requested but missing required parameters k and l."
            )
        sbs = bitstring.copy()[::-1]
        sbs[l_index - 1] = 1
        sbs[k_index - 1] = 0
        sbs = sbs[::-1]
        sc = QuantumCircuit(num_qubits)
        # Apply X gates to occupied orbitals
        for i, b in enumerate(sbs):
            if b:
                sc.x(i)
        sc.compose(ansatz, inplace=True)
        sc_assigned_ansatz = sc.assign_parameters(initial_parameters)
        ph_sc = pcvl.LogicalState(sbs)
        sc_ansatz_transpiled = transpile(
            sc_assigned_ansatz, basis_gates=["u3", "cx"], optimization_level=3
        )
        sc_ansatz_rot = rotate_qubits(pauli_string, sc_ansatz_transpiled.copy())
        processor_sc = compile(sc_ansatz_rot, input_state=ph_sc, noise_model=noise_model)

    if singlet_excitation:
        return [processor_HF, processor_sc]
    else:
        return processor_HF


def Bitstring_initial_states(
    layers: int,
    n_states: int,
    lp: np.ndarray,
    pauli_string: str,
    cost: str = "VQE",
    noise_model: NoiseModel | None = None,
) -> Processor | list[Processor]:
    """
    Build a Bitstring-based variational ansatz using Qiskit's ``n_local`` circuit,
    combined with initial reference states and compiled into Perceval processors.

    Parameters
    ----------
    layers : int
        Number of circuit layers (repetitions) in the ansatz.
    n_states : int
        Number of states.
    lp : np.ndarray
        Array of parameter values for the ansatz circuit.
    pauli_string : str
        Pauli operator string defining the measurement basis.
    cost : str, optional
        Type of cost function to prepare. One of:
            - ``"VQE"`` : Return only the ground-state processor (default)
            - ``"e-VQE"`` : Return a list of processors for excited states
    noise_model : NoiseModel, optional
        A Perceval ``NoiseModel`` object representing the noise model to include in compilation.

    Returns
    -------
    Processor or list of Processor
        If ``cost="VQE"``, returns a single Perceval ``Processor`` instance.
        If ``cost="e-VQE"``, returns a list of processors.

    Raises
    ------
    ValueError
        If ``method`` or ``cost`` arguments are invalid.

    Notes
    -----
    - The ansatz is constructed by Bitstrings with a parameterized ``n_local`` circuit (Ry–CX entangling pattern).
    - See `https://scipost.org/SciPostPhys.14.3.055` for details on the DFT mapping.
    """
    initial_circuits = []
    num_qubits = len(lp) // 2
    for _i in range(n_states):
        initial_circuits += [QuantumCircuit(num_qubits)]
    """Initial states"""
    for state in range(n_states):  # binarystring representation of the integer
        for i in list_of_ones(state, num_qubits):
            initial_circuits[state].x(i)

    circuits = []
    for state in range(n_states):
        ansatz = n_local(
            num_qubits=num_qubits,
            rotation_blocks="ry",
            entanglement_blocks="cx",
            entanglement="linear",
            reps=layers,
            insert_barriers=False,
        )
        # Prepend the initial state
        circuit = QuantumCircuit(num_qubits)
        circuit.compose(initial_circuits[state], inplace=True)
        circuit.compose(ansatz, inplace=True)
        circuits.append(circuit)

    """Assign parameters"""
    circuits = [c.assign_parameters(lp) for c in circuits]

    intial_states = []
    for c in range(len(initial_circuits)):
        sv = Statevector.from_instruction(initial_circuits[c])
        index = list(sv.data).index(1 + 0j)  # position of the "1"
        bitstring = format(index, f"0{sv.num_qubits}b")
        bits_list = [int(b) for b in bitstring]
        intial_states.append(pcvl.LogicalState(bits_list))

    ansatz_transpiled = [
        transpile(c, basis_gates=["u3", "cx"], optimization_level=3) for c in circuits
    ]

    ansatz_rot = [rotate_qubits(pauli_string, at.copy()) for at in ansatz_transpiled]

    processors = [
        compile(ar, input_state=st, noise_model=noise_model)
        for ar, st in zip(ansatz_rot, intial_states, strict=False)
    ]
    if cost == "VQE":
        return processors[0]
    elif cost == "e-VQE":
        return processors
    else:
        raise ValueError("Invalid cost option. Use 'VQE' or 'e-VQE'.")
