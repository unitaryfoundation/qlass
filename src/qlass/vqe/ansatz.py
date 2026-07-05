from functools import lru_cache

import numpy as np
import perceval as pcvl
from perceval.components import Processor
from perceval.utils import NoiseModel
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import n_local

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
    singlet_excitation: bool = False,
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
        Number of spatial orbitals in the system. Currently unused: the qubit
        count is derived as ``2 * sum(num_electrons)``, i.e. the number of spin
        orbitals is assumed to be twice the number of electrons.
    num_electrons : tuple of int
        Number of alpha and beta electrons, given as (num_alpha, num_beta).
    initial_parameters : np.ndarray
        Initial values for the parameterized ansatz. Must have length
        ``2 * num_qubits`` (the ``n_local`` circuit with one repetition).
    pauli_string : str
        Pauli string used to rotate the qubits after the ansatz.
    singlet_excitation : bool, optional
        If True, apply a singlet excitation to the Hartree-Fock state using
        `k_index` and `l_index`. Default is False.
    k_index : int or None, optional
        One-based index of the occupied orbital for singlet excitation.
        Required if `singlet_excitation=True`.
    l_index : int or None, optional
        One-based index of the unoccupied orbital for singlet excitation.
        Required if `singlet_excitation=True`.
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
        If `singlet_excitation=True` but either `k_index` or `l_index` is not provided.

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
    ansatz_transpiled = transpile(ansatz_assigned, basis_gates=["u3", "cx"], optimization_level=3)
    ansatz_rot = rotate_qubits(pauli_string, ansatz_transpiled.copy())
    # The X gates above already prepare the Hartree-Fock state, so the photonic
    # input is the dual-rail |0...0> (issue #236: feeding the HF bitstring here
    # as well made the two preparations cancel).
    input_state = pcvl.LogicalState([0] * num_qubits)
    processor_HF = compile(ansatz_rot, input_state=input_state, noise_model=noise_model)

    # singlet state
    processor_sc = None
    if singlet_excitation:
        if k_index is None or l_index is None:
            raise ValueError(
                "Singlet excitation requested but missing required parameters k_index and l_index."
            )
        sbs = bitstring.copy()[::-1]
        # k_index and l_index are 1-based orbital indices
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
        sc_ansatz_transpiled = transpile(
            sc_assigned_ansatz, basis_gates=["u3", "cx"], optimization_level=3
        )
        sc_ansatz_rot = rotate_qubits(pauli_string, sc_ansatz_transpiled.copy())
        processor_sc = compile(sc_ansatz_rot, input_state=input_state, noise_model=noise_model)

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
    # n_local(..., reps=layers) has num_qubits * (layers + 1) parameters
    num_qubits = len(lp) // (layers + 1)
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

    ansatz_transpiled = [
        transpile(c, basis_gates=["u3", "cx"], optimization_level=3) for c in circuits
    ]

    ansatz_rot = [rotate_qubits(pauli_string, at.copy()) for at in ansatz_transpiled]

    # The X-gate state preparation is already part of each circuit, so every
    # processor takes the dual-rail-encoded |0...0> as photonic input.
    input_state = pcvl.LogicalState([0] * num_qubits)
    processors = [
        compile(ar, input_state=input_state, noise_model=noise_model) for ar in ansatz_rot
    ]
    if cost == "VQE":
        return processors[0]
    elif cost == "e-VQE":
        return processors
    else:
        raise ValueError("Invalid cost option. Use 'VQE' or 'e-VQE'.")


def _fock_kerr(kappa: float, n_max: int) -> np.ndarray:
    """
    Single-mode Kerr gate in the Fock basis.

    The Kerr gate applies a phase shift proportional to n^2:
        U_kerr(κ)|n⟩ = exp(i κ n²) |n⟩

    Args:
        kappa: Kerr nonlinearity parameter.
        n_max: Maximum photon number (Fock space truncation).
                The Hilbert space dimension is n_max + 1.

    Returns:
        np.ndarray: Diagonal unitary matrix of shape (n_max+1, n_max+1).
    """
    dim = n_max + 1
    ns = np.arange(dim)
    return np.diag(np.exp(1j * kappa * ns**2))


def _fock_phase_shift(phi: float, n_max: int) -> np.ndarray:
    """
    Single-mode phase shifter in the Fock basis.

    The phase shifter applies a phase shift proportional to n:
        U_ps(φ)|n⟩ = exp(i φ n) |n⟩

    Args:
        phi: Phase shift parameter.
        n_max: Maximum photon number (Fock space truncation).

    Returns:
        np.ndarray: Diagonal unitary matrix of shape (n_max+1, n_max+1).
    """
    dim = n_max + 1
    ns = np.arange(dim)
    return np.diag(np.exp(1j * phi * ns))


@lru_cache(maxsize=16)
def _cached_fock_bs_operators(n_max: int) -> tuple[np.ndarray, np.ndarray]:
    """Cache the fixed operator products a0† a1 and a0 a1† for the Beamsplitter Hamiltonian."""
    dim = n_max + 1

    # Build annihilation operator for a single mode in Fock basis
    # a|n⟩ = √n |n-1⟩
    a = np.zeros((dim, dim), dtype=complex)
    for n in range(1, dim):
        a[n - 1, n] = np.sqrt(n)

    # Identity for single mode
    I_single = np.eye(dim, dtype=complex)

    # Two-mode operators: a_0 = a ⊗ I, a_1 = I ⊗ a
    a0 = np.kron(a, I_single)
    a1 = np.kron(I_single, a)

    a0dag = a0.conj().T
    a1dag = a1.conj().T

    return a0dag @ a1, a0 @ a1dag


def _fock_beamsplitter(theta: float, phi: float, n_max: int) -> np.ndarray:
    """
    Two-mode beamsplitter in the Fock basis.

    The beamsplitter is generated by the Hamiltonian:
        H_BS = θ (e^{iφ} a†_0 a_1 + e^{-iφ} a_0 a†_1)

    so that U_BS = exp(-i H_BS), which is unitary by construction.

    Args:
        theta: Beamsplitter angle (reflectivity parameter).
        phi: Beamsplitter phase.
        n_max: Maximum photon number per mode.

    Returns:
        np.ndarray: Unitary matrix of shape ((n_max+1)^2, (n_max+1)^2)
                    acting on the two-mode Fock space.
    """
    op1, op2 = _cached_fock_bs_operators(n_max)

    # BS Hamiltonian: H = θ (e^{iφ} a†_0 a_1 + e^{-iφ} a_0 a†_1)
    H_bs = theta * (np.exp(1j * phi) * op1 + np.exp(-1j * phi) * op2)

    # Since H_bs is Hermitian, we can compute exp(-i H_bs) via eigen-decomposition
    # H_bs = V D V† with real eigenvalues D, so exp(-i H_bs) = V exp(-i D) V†
    eigenvalues, eigenvectors = np.linalg.eigh(H_bs)
    U = eigenvectors @ np.diag(np.exp(-1j * eigenvalues)) @ eigenvectors.conj().T
    return U  # type: ignore[no-any-return]


def kerr_ansatz(params: np.ndarray, num_kerr: int = 4, n_max: int = 4) -> np.ndarray:
    """
    Construct a 2-mode nonlinear photonic ansatz with Kerr gates and a beamsplitter.

    The circuit layout is:

        Mode 0: ── Gate_0 ──┐          ┌── Gate_2 ──
                             BS(θ, φ)
        Mode 1: ── Gate_1 ──┘          └── Gate_3 ──

    where each Gate slot is either a Kerr gate or a phase shifter, depending
    on ``num_kerr``. When ``num_kerr < 4``, the last slots (3, 2, 1, ...) are
    replaced by phase shifters.

    This ansatz operates directly in the truncated Fock basis (no dual-rail
    qubit encoding). It returns a unitary matrix of dimension (n_max+1)^2
    acting on the 2-mode Fock space, intended for use with a bosonic
    Hamiltonian in a Fock-space VQE setting.

    Args:
        params (np.ndarray): Array of 6 variational parameters:
            - params[0..3]: gate parameters (Kerr κ or phase-shift φ)
            - params[4]: beamsplitter angle θ
            - params[5]: beamsplitter phase φ
        num_kerr (int): Number of Kerr gates to use (1–4). The remaining
            gate slots use phase shifters. Default is 4.
        n_max (int): Maximum photon number per mode (Fock space truncation).
            The single-mode Hilbert space has dimension n_max + 1.
            Default is 4.

    Returns:
        np.ndarray: Unitary matrix of shape ((n_max+1)^2, (n_max+1)^2)
                    representing the full 2-mode circuit in Fock space.

    Raises:
        ValueError: If ``num_kerr`` is not in {1, 2, 3, 4} or ``params``
                    does not have exactly 6 elements.
    """
    if num_kerr < 1 or num_kerr > 4:
        raise ValueError(f"num_kerr must be between 1 and 4, got {num_kerr}")
    if len(params) != 6:
        raise ValueError(f"Expected 6 parameters, got {len(params)}")

    # Determine which slots are Kerr gates (True) or phase shifters (False).
    # Slots 0..num_kerr-1 are Kerr, the rest are phase shifters.
    is_kerr = [i < num_kerr for i in range(4)]

    # Build single-mode gate unitaries
    def _build_gate(param: float, kerr: bool) -> np.ndarray:
        return _fock_kerr(param, n_max) if kerr else _fock_phase_shift(param, n_max)

    # Layer 1: gates on mode 0 and mode 1 (slots 0, 1)
    gate0_l1 = _build_gate(params[0], is_kerr[0])
    gate1_l1 = _build_gate(params[1], is_kerr[1])
    layer1 = np.kron(gate0_l1, gate1_l1)

    # Beamsplitter
    bs = _fock_beamsplitter(params[4], params[5], n_max)

    # Layer 2: gates on mode 0 and mode 1 (slots 2, 3)
    gate0_l2 = _build_gate(params[2], is_kerr[2])
    gate1_l2 = _build_gate(params[3], is_kerr[3])
    layer2 = np.kron(gate0_l2, gate1_l2)

    # Full circuit: layer2 @ bs @ layer1
    return layer2 @ bs @ layer1  # type: ignore[no-any-return]
