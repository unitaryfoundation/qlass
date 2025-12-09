import itertools

import numpy as np

# OpenFermion imports - replacing qiskit_nature
from openfermion.chem import MolecularData

# Import QubitOperator para type hinting
from openfermion.ops import InteractionOperator, QubitOperator
from openfermion.transforms import (
    get_fermion_operator,
    jordan_wigner,
    symmetry_conserving_bravyi_kitaev,
)
from openfermionpyscf import run_pyscf


def sparsepauliop_dictionary(H: QubitOperator) -> dict[str, float]:
    """
    Converts an OpenFermion QubitOperator into a dictionary representation.

    This function translates the structure of an OpenFermion QubitOperator,
    which represents a sum of Pauli strings, into a Python dictionary.
    The keys of the dictionary are string representations of Pauli operators
    (e.g., "IXYZ"), and the values are their corresponding real coefficients.

    Args:
        H: The OpenFermion QubitOperator to be converted. Each term in this
           operator is a product of Pauli operators acting on specific qubits.

    Returns:
        A dictionary where each key is a Pauli string (e.g., "IZX")
        representing a term in the Hamiltonian, and its value is the
        real part of the coefficient for that term.
    """
    pauli_dict: dict[str, float] = {}

    # Determine the total number of qubits in the system.
    # This is found by identifying the highest qubit index acted upon by any Pauli operator.
    max_qubit_idx = -1  # Initialize to -1 to correctly handle 0-indexed qubits
    if H.terms:
        for pauli_term_key in H.terms:
            if pauli_term_key:  # Checks if the term is not the global identity `()`
                # pauli_term_key is a tuple of (qubit_index, Pauli_operator_char) tuples, e.g., ((0, 'X'), (1, 'Y'))
                current_max_for_term = max(idx for idx, _ in pauli_term_key)
                if current_max_for_term > max_qubit_idx:
                    max_qubit_idx = current_max_for_term
        num_qubits = max_qubit_idx + 1
    else:
        num_qubits = max_qubit_idx + 1  # If max_qubit_idx remains -1, num_qubits becomes 0.

    # Let's ensure num_qubits is at least 1 if an identity term is present and no other terms define size.
    if not H.terms and num_qubits == 0:  # Truly empty QubitOperator
        return {}  # An empty operator has no Pauli terms.
    if H.terms and not any(bool(term) for term in H.terms) and num_qubits == 0:
        # This means H.terms only contains {(): coeff}, e.g. QubitOperator('')
        # max_qubit_idx was -1, num_qubits became 0. For an identity string, we need at least 1 qubit.
        num_qubits = 1

    # A term consists of a Pauli product (pauli_string_openfermion) and its coefficient.
    for pauli_string_openfermion, coefficient in H.terms.items():
        if (
            not pauli_string_openfermion
        ):  # This is the global identity term, represented by an empty tuple `()`.
            # The Pauli key for the identity is a string of 'I's, one for each qubit.
            pauli_key = "I" * num_qubits
        else:
            # For non-identity terms, construct the Pauli string.
            # Initialize a list representing the Pauli operators on all qubits, default to 'I'.
            pauli_array = ["I"] * num_qubits

            # Populate the array with the specific Pauli operators (X, Y, Z) at their respective qubit indices.
            for qubit_idx, pauli_op_char in pauli_string_openfermion:
                if qubit_idx < num_qubits:  # Ensure index is within bounds
                    pauli_array[qubit_idx] = pauli_op_char
                else:
                    # For now, we assume num_qubits is correctly pre-calculated.
                    pass

            pauli_key = "".join(pauli_array)

        # Store the term in the dictionary, using only the real part of the coefficient.
        pauli_dict[pauli_key] = float(coefficient.real)

    return pauli_dict


def pauli_commute(p1: str, p2: str) -> bool:
    """
    Check if two Pauli strings commute.

    Two Pauli strings commute if and only if the number of positions where
    both have non-identity operators that are different is even.

    Args:
        p1 (str): First Pauli string
        p2 (str): Second Pauli string

    Returns:
        bool: True if the Pauli strings commute, False otherwise

    Raises:
        ValueError: If Pauli strings have different lengths
    """
    if len(p1) != len(p2):
        raise ValueError("Pauli strings must have the same length")

    diff_count = 0
    for i in range(len(p1)):
        # Count positions where both are non-identity and different
        if p1[i] != "I" and p2[i] != "I" and p1[i] != p2[i]:
            diff_count += 1

    return diff_count % 2 == 0


def group_commuting_pauli_terms(hamiltonian: dict[str, float]) -> list[dict[str, float]]:
    """
    Group commuting Pauli terms in a Hamiltonian.

    This function takes a Hamiltonian represented as a dictionary of Pauli strings
    and their coefficients, and returns a list of Hamiltonians where each group
    contains only mutually commuting Pauli terms. This grouping can be used to
    reduce the number of measurements needed in quantum algorithms like VQE.

    This function provides a more general approach than OpenFermion's
    group_into_tensor_product_basis_sets, which only groups terms that are
    diagonal in the same tensor product basis. Our function groups all
    mutually commuting terms regardless of the measurement basis required.

    Args:
        hamiltonian (Dict[str, float]): Hamiltonian dictionary with Pauli string
                                       keys and coefficient values

    Returns:
        List[Dict[str, float]]: List of grouped Hamiltonians, where each group
                               contains mutually commuting Pauli terms
    """
    if not hamiltonian:
        return []

    groups: list[dict[str, float]] = []

    for pauli_string, coefficient in hamiltonian.items():
        placed = False

        # Try to place this term in an existing group
        for group in groups:
            # Check if this term commutes with all terms in the group
            if all(pauli_commute(pauli_string, existing) for existing in group):
                group[pauli_string] = coefficient
                placed = True
                break

        # If it doesn't fit in any existing group, create a new one
        if not placed:
            groups.append({pauli_string: coefficient})

    return groups


def group_commuting_pauli_terms_openfermion_hybrid(
    hamiltonian: dict[str, float],
) -> list[dict[str, float]]:
    """
    Hybrid approach that tries to use OpenFermion's grouping when possible,
    fallback to our implementation otherwise.

    This function attempts to leverage OpenFermion's optimized grouping functions
    when they are applicable, while maintaining full compatibility with our
    general commuting term grouping for all other cases.

    Args:
        hamiltonian (Dict[str, float]): Hamiltonian dictionary with Pauli string
                                       keys and coefficient values

    Returns:
        List[Dict[str, float]]: List of grouped Hamiltonians, where each group
                               contains mutually commuting Pauli terms
    """
    if not hamiltonian:
        return []

    try:
        # Try to use OpenFermion's grouping for tensor product basis sets
        from openfermion.measurements import group_into_tensor_product_basis_sets
        from openfermion.ops import QubitOperator

        # Convert Dict to QubitOperator
        qubit_op = QubitOperator()
        for pauli_string, coeff in hamiltonian.items():
            # Convert our format to OpenFermion format
            of_term = []
            for i, pauli in enumerate(pauli_string):
                if pauli != "I":
                    of_term.append((i, pauli))

            if of_term:
                qubit_op += QubitOperator(tuple(of_term), coeff)
            else:
                # Identity term
                qubit_op += QubitOperator((), coeff)

        # Use OpenFermion's grouping
        of_groups = group_into_tensor_product_basis_sets(qubit_op)

        # Convert back to our format
        groups = []
        for of_group in of_groups:
            group_dict = sparsepauliop_dictionary(of_group)
            if group_dict:  # Only add non-empty groups
                groups.append(group_dict)

        return groups

    except (ImportError, Exception):
        # Fallback to our implementation if OpenFermion grouping fails
        return group_commuting_pauli_terms(hamiltonian)


def LiH_hamiltonian(
    R: float = 1.5, charge: int = 0, spin: int = 0, num_electrons: int = 2, num_orbitals: int = 2
) -> dict[str, float]:
    """
    Generate the qubit Hamiltonian for the LiH molecule at a given bond length.

    This function uses OpenFermion and PySCF to compute molecular integrals,
    applies active space transformation, and maps to qubits via Jordan-Wigner.

    Note: Nuclear repulsion energy is excluded to maintain compatibility
    with qiskit_nature behavior.

    Args:
        R (float): Bond length in Angstroms
        charge (int): Charge of the molecule
        spin (int): Spin of the molecule
        num_electrons (int): Number of electrons in active space
        num_orbitals (int): Number of molecular orbitals in active space

    Returns:
        Dict[str, float]: Hamiltonian dictionary with Pauli string keys
    """

    # Create geometry in OpenFermion format
    geometry = [("Li", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, R))]

    # Create molecular data object
    molecule = MolecularData(
        geometry=geometry,
        basis="sto-3g",
        multiplicity=spin + 1,  # OpenFermion uses multiplicity = 2S + 1
        charge=charge,
    )

    # Run PySCF calculation
    molecule = run_pyscf(molecule, run_scf=True, run_fci=True)

    # Apply active space transformation
    # Calculate core orbitals to freeze
    total_electrons = molecule.n_electrons
    n_core_orbitals = (total_electrons - num_electrons) // 2
    occupied_indices = list(range(n_core_orbitals))

    # Calculate active orbital indices
    active_indices = list(range(n_core_orbitals, n_core_orbitals + num_orbitals))

    # Get molecular Hamiltonian in active space
    molecular_hamiltonian = molecule.get_molecular_hamiltonian(
        occupied_indices=occupied_indices, active_indices=active_indices
    )

    molecular_hamiltonian_no_nuclear = InteractionOperator(
        constant=0.0,
        one_body_tensor=molecular_hamiltonian.one_body_tensor,
        two_body_tensor=molecular_hamiltonian.two_body_tensor,
    )

    # Convert to fermionic operator
    fermionic_op = get_fermion_operator(molecular_hamiltonian_no_nuclear)

    # Apply Jordan-Wigner transformation
    H_qubit = jordan_wigner(fermionic_op)

    # Convert to dictionary format
    return sparsepauliop_dictionary(H_qubit)


def generate_random_hamiltonian(num_qubits: int) -> dict[str, float]:
    """
    Generate a random Hamiltonian.

    Creates a random Pauli operator by generating all possible Pauli strings
    for the given number of qubits and assigning random coefficients.

    Args:
        num_qubits (int): Number of qubits

    Returns:
        Dict[str, float]: Hamiltonian dictionary with random coefficients
    """

    # Generate all possible Pauli strings consisting of 'X', 'Y', 'Z', 'I'
    bitstrings = ["".join(bits) for bits in itertools.product("XYZI", repeat=num_qubits)]

    # Create a dictionary with these bitstrings as keys and random numbers as values
    random_values = np.random.random(len(bitstrings)) - 0.5
    H = dict(zip(bitstrings, random_values, strict=False))

    return H


def LiH_hamiltonian_tapered(R: float) -> dict[str, float]:
    """
    Generate the Hamiltonian for the LiH molecule at a given bond length using tapering technique.

    This function applies active space reduction equivalent to qiskit_nature's
    tapering approach, which reduces the number of qubits by removing
    orbitals that don't contribute significantly to bonding.

    The implementation uses specific orbital selection to mimic the behavior
    of ActiveSpaceTransformer with active_orbitals=[1,2,5].

    Args:
        R (float): Bond length in Angstroms

    Returns:
        Dict[str, float]: Hamiltonian dictionary with reduced qubit count
    """

    # Create geometry
    geometry = [("Li", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, R))]

    # Create molecular data object
    molecule = MolecularData(
        geometry=geometry,
        basis="sto-3g",
        multiplicity=1,  # Singlet state
        charge=0,
    )

    # Run PySCF calculation
    molecule = run_pyscf(molecule, run_scf=True, run_fci=True)
    total_n_elec = molecule.n_electrons

    # Apply active space reduction equivalent to tapering
    # Freeze core orbital (1s of Li)
    n_core_orbitals = 1
    occupied_indices = list(range(n_core_orbitals))

    active_indices = [1, 2, 5]
    active_spin_orbitals = 2 * len(active_indices)

    active_n_elec = total_n_elec - 2 * n_core_orbitals

    molecular_hamiltonian = molecule.get_molecular_hamiltonian(
        occupied_indices=occupied_indices, active_indices=active_indices
    )

    # Remove nuclear repulsion energy
    molecular_hamiltonian_no_nuclear = InteractionOperator(
        constant=0.0,  # Set constant to 0 to match qiskit_nature
        one_body_tensor=molecular_hamiltonian.one_body_tensor,
        two_body_tensor=molecular_hamiltonian.two_body_tensor,
    )

    # Convert to fermionic operator
    fermionic_op = get_fermion_operator(molecular_hamiltonian_no_nuclear)

    # Apply Bravyi-Kitaev transformation
    qubit_op = symmetry_conserving_bravyi_kitaev(fermionic_op, active_spin_orbitals, active_n_elec)

    return sparsepauliop_dictionary(qubit_op)


def Hchain_KS_hamiltonian(
    n_hydrogens: int = 2, R: float = 1.2
) -> tuple[dict[str, float], np.ndarray, int]:
    """
    Generate the one-body Hamiltonian for a linear chain of hydrogen atoms at a given bond length.

    This function constructs a non-interacting one-body Hamiltonian for a chain of hydrogen atoms
    using Density Functional Theory (DFT) with a closed-shell Hartree-Fock (RHF) method.
    The hydrogen atoms are placed linearly along the z-axis with a uniform bond length ``R``.
    The resulting molecular orbitals are transformed into a qubit Hamiltonian representation.

    Parameters
    ----------
    n_hydrogens : int, optional
        Number of hydrogen atoms in the linear chain. Must be an even integer.
        Default is 2.
    R : float, optional
        Bond length between adjacent hydrogen atoms in angstroms. Default is 1.2.

    Returns
    -------
    H_qubit_dic : dict
        Dictionary representation of the qubit Hamiltonian in terms of Pauli operators.
        The keys correspond to Pauli strings and the values are their coefficients.
    mo_energy : list of float
        Molecular orbital energies computed from PySCF.
    n_molecular_orbital : int
        Number of molecular orbitals.

    Notes
    -----
    - The electronic structure is calculated using the minimal ``sto-3g`` basis set.
    - The function internally performs the following steps:
        1. Builds the molecular geometry.
        2. Runs RHF self-consistent field (SCF) calculation via PySCF.
        3. Constructs the Fock and overlap matrices in the atomic orbital (AO) basis.
        4. Transforms to an orthogonalized atomic orbital (OAO) basis.
        5. Maps the resulting Hamiltonian to a qubit representation.
    - The transformation to the qubit Hamiltonian uses a helper function
      ``transformation_Hmatrix_Hqubit`` and a dictionary builder ``sparsepauliop_dictionary``.

    """
    from pyscf import gto, scf

    geometry = []
    numberof_qubits = int(np.log2(n_hydrogens))

    for d in range(n_hydrogens // 2):
        geometry.append(("H", (0.0, 0.0, -(R / 2.0 + d * R))))
        geometry.append(("H", (0.0, 0.0, +(R / 2.0 + d * R))))

    molecule = gto.M(atom=geometry, basis="sto-3g")
    mf = scf.RHF(molecule)
    mf.scf()
    F_AO = mf.get_fock()
    S_AO = mf.get_ovlp()
    # Compute the inverse square root of the overlap matrix S
    S_eigval, S_eigvec = np.linalg.eigh(S_AO)
    S_sqrt_inv = S_eigvec @ np.diag((S_eigval) ** (-1.0 / 2.0)) @ S_eigvec.T
    F_OAO = S_sqrt_inv @ F_AO @ S_sqrt_inv
    H_qubit = transformation_Hmatrix_Hqubit(F_OAO, numberof_qubits)
    H_qubit_dic = sparsepauliop_dictionary(H_qubit)
    mo_energy = mf.mo_energy

    return H_qubit_dic, mo_energy, int(len(mo_energy))


def transformation_Hmatrix_Hqubit(Hmatrix: np.ndarray, nqubits: int) -> QubitOperator:
    """
        Transform a Hamiltonian matrix into an OpenFermion ``QubitOperator`` representation.

        This function converts a Hermitian matrix, expressed in the computational basis,
        into an equivalent Hamiltonian written as a sum of tensor products of Pauli
        operators (I, X, Y, Z). The result is an OpenFermion ``QubitOperator`` that can
        be used in quantum simulation frameworks.

        Parameters
        ----------
        Hmatrix : np.ndarray
            A complex Hermitian matrix of shape ``(2**nqubits, 2**nqubits)`` representing
            the Hamiltonian in the computational basis.
        nqubits : int
            Number of qubits in the system. Determines the dimension of ``Hmatrix``.

        Returns
        -------
        H_qubit : openfermion.QubitOperator
            The Hamiltonian expressed as a sum of Pauli operators with complex coefficients.

        Notes
        -----
        - Each matrix element ``Hmatrix[i, j]`` is decomposed into a sum of tensor products
          of single-qubit projectors expressed in the Pauli basis.
        - The transformation uses the following single-qubit projector identities:

            .. math::
            |0⟩⟨0| = (I + Z) / 2, \\
            |1⟩⟨1| = (I - Z) / 2, \\
            |0⟩⟨1| = (X + iY) / 2, \\
            |1⟩⟨0| = (X - iY) / 2
        - Terms with negligible coefficients (magnitude < 1e-12) are ignored to improve
          numerical stability.

        """
    H_qubit = QubitOperator()

    # Basis projectors |i><j| expressed in Pauli basis
    def single_qubit_projector(bi: str, bj: str) -> list[tuple[str, complex]]:
        # |0><0| = (I+Z)/2, |1><1| = (I−Z)/2
        # |0><1| = (X+iY)/2, |1><0| = (X−iY)/2
        if bi == "0" and bj == "0":
            return [("I", 0.5), ("Z", 0.5)]
        elif bi == "1" and bj == "1":
            return [("I", 0.5), ("Z", -0.5)]
        elif bi == "0" and bj == "1":
            return [("X", 0.5), ("Y", 0.5j)]
        elif bi == "1" and bj == "0":
            return [("X", 0.5), ("Y", -0.5j)]
        else:
            raise ValueError

    # Loop through all matrix elements
    dim = 2**nqubits
    for i in range(dim):
        for j in range(dim):
            if np.abs(Hmatrix[i, j]) < 1e-12:
                continue
            bit_i = format(i, f"0{nqubits}b")
            bit_j = format(j, f"0{nqubits}b")

            # Build the tensor product operator for |i><j|
            # We do this by expanding all combinations from single-qubit projectors
            term_ops: list[tuple[str, complex]] = [("", 1.0)]  # (pauli_string, coeff)
            for q in range(nqubits):
                proj = single_qubit_projector(bit_i[q], bit_j[q])
                new_terms: list[tuple[str, complex]] = []
                for ps, pc in term_ops:
                    for p, coeff in proj:
                        new_terms.append((ps + p, pc * coeff))
                term_ops = new_terms

            # Add contributions to the Hamiltonian
            for ps, coeff in term_ops:
                # compress consecutive 'I's, drop them
                pauli_term = tuple((q, p) for q, p in enumerate(ps) if p != "I")
                H_qubit += QubitOperator(pauli_term, Hmatrix[i, j] * coeff)

    return H_qubit


def Hchain_hamiltonian_WFT(
    n_hydrogens: int = 2,
    R: float = 0.8,
    charge: int = 0,
    spin: int = 0,
    num_electrons: int = 2,
    num_orbitals: int = 2,
    tampering: bool = False,
) -> dict[str, float]:
    """
    Construct the qubit Hamiltonian for a linear hydrogen chain (Hₙ) using
        wavefunction-based methods.

        This function builds the molecular geometry, performs a PySCF electronic
        structure calculation through OpenFermion, extracts an active-space molecular
        Hamiltonian, and maps it to a qubit Hamiltonian.

        Nuclear repulsion energy is included manually. The resulting Hamiltonian is
        returned as a dictionary mapping Pauli strings to coefficients.

        Parameters
        ----------
        n_hydrogens : int, optional
            Number of hydrogen atoms in the linear chain. Must be even, as atoms
            are paired symmetrically about the origin. Default is ``2``.
        R : float, optional
            Bond length between adjacent hydrogens in ångström. Default is ``0.8``.
        charge : int, optional
            Total molecular charge. Default is ``0``.
        spin : int, optional
            Spin multiplicity parameter such that multiplicity = ``2S + 1``.
            For example, ``spin=0`` corresponds to a singlet. Default is ``0``.
        num_electrons : int, optional
            Number of electrons.
            Default is ``2``.
        num_orbitals : int, optional
            Number of spatial molecular orbitals.
            Default is ``2``.
        tampering: bool, optional
            if True, symmetry-conserving Bravyi–Kitaev transformed Hamiltonian
            if False, Full Hamiltonian transformed using JW

        Returns
        -------
        Dict[str, float]
            A dictionary representing the qubit Hamiltonian, where keys are
            Pauli strings (e.g., ``"XIZY"``) and values are real coefficients.

        Notes
        -----
        - Geometry is generated as a symmetric linear chain along the z-axis.
        - PySCF is used to compute SCF and FCI energies through OpenFermion.
        - Active-space selection freezes no core orbitals.
        - If symmetry is True, the Hamiltonian is mapped to qubits using the symmetry-conserving
          Bravyi–Kitaev transformation else Full Hamiltonian using Jordan–Wigner mapping.
    """

    geometry = []
    for d in range(n_hydrogens // 2):
        geometry.append(("H", (0.0, 0.0, -(R / 2.0 + d * R))))
        geometry.append(("H", (0.0, 0.0, +(R / 2.0 + d * R))))

    # molecule = gto.M(atom=geometry, basis='sto-3g')

    # Create molecular data object
    molecule = MolecularData(
        geometry=geometry,
        basis="sto-3g",
        multiplicity=spin + 1,  # OpenFermion uses multiplicity = 2S + 1
        charge=charge,
    )

    # Run PySCF calculation
    molecule = run_pyscf(molecule, run_scf=True, run_fci=True)
    fermionic_op = get_fermion_operator(molecule.get_molecular_hamiltonian())
    if tampering:
        from openfermion.transforms import symmetry_conserving_bravyi_kitaev

        # H_qubit_full = jordan_wigner(fermionic_op)
        H_qubit = symmetry_conserving_bravyi_kitaev(
            fermionic_op, active_orbitals=num_electrons * 2, active_fermions=num_electrons
        )
    if not tampering:
        from openfermion.transforms import jordan_wigner

        H_qubit = jordan_wigner(fermionic_op)
    # Convert to dictionary format
    return sparsepauliop_dictionary(H_qubit)
