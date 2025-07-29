import itertools
import numpy as np
# OpenFermion imports - replacing qiskit_nature
from openfermion.chem import MolecularData
from openfermion.transforms import get_fermion_operator, jordan_wigner, symmetry_conserving_bravyi_kitaev
# Import QubitOperator para type hinting
from openfermion.ops import InteractionOperator, QubitOperator
from openfermionpyscf import run_pyscf

from typing import Dict, List

def sparsepauliop_dictionary(H: QubitOperator) -> Dict[str, float]:
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
    pauli_dict: Dict[str, float] = {}

    # Determine the total number of qubits in the system.
    # This is found by identifying the highest qubit index acted upon by any Pauli operator.
    max_qubit_idx = -1 # Initialize to -1 to correctly handle 0-indexed qubits
    if H.terms:
        for pauli_term_key in H.terms.keys():
            if pauli_term_key:  # Checks if the term is not the global identity `()`
                # pauli_term_key is a tuple of (qubit_index, Pauli_operator_char) tuples, e.g., ((0, 'X'), (1, 'Y'))
                current_max_for_term = max(idx for idx, _ in pauli_term_key)
                if current_max_for_term > max_qubit_idx:
                    max_qubit_idx = current_max_for_term
        num_qubits = max_qubit_idx + 1
    else:
        num_qubits = max_qubit_idx + 1 # If max_qubit_idx remains -1, num_qubits becomes 0.

    # Let's ensure num_qubits is at least 1 if an identity term is present and no other terms define size.
    if not H.terms and num_qubits == 0 : # Truly empty QubitOperator
        return {} # An empty operator has no Pauli terms.
    if H.terms and not any(bool(term) for term in H.terms.keys()) and num_qubits == 0:
        # This means H.terms only contains {(): coeff}, e.g. QubitOperator('')
        # max_qubit_idx was -1, num_qubits became 0. For an identity string, we need at least 1 qubit.
        num_qubits = 1
        
    # A term consists of a Pauli product (pauli_string_openfermion) and its coefficient.
    for pauli_string_openfermion, coefficient in H.terms.items():
        if not pauli_string_openfermion:  # This is the global identity term, represented by an empty tuple `()`.
            # The Pauli key for the identity is a string of 'I's, one for each qubit.
            pauli_key = 'I' * num_qubits
        else:
            # For non-identity terms, construct the Pauli string.
            # Initialize a list representing the Pauli operators on all qubits, default to 'I'.
            pauli_array = ['I'] * num_qubits
            
            # Populate the array with the specific Pauli operators (X, Y, Z) at their respective qubit indices.
            for qubit_idx, pauli_op_char in pauli_string_openfermion:
                if qubit_idx < num_qubits: # Ensure index is within bounds
                    pauli_array[qubit_idx] = pauli_op_char
                else:
                    # For now, we assume num_qubits is correctly pre-calculated.
                    pass 
            
            pauli_key = ''.join(pauli_array)
        
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
        if p1[i] != 'I' and p2[i] != 'I' and p1[i] != p2[i]:
            diff_count += 1
    
    return diff_count % 2 == 0

def group_commuting_pauli_terms(hamiltonian: Dict[str, float]) -> List[Dict[str, float]]:
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
    
    groups = []
    
    for pauli_string, coefficient in hamiltonian.items():
        placed = False
        
        # Try to place this term in an existing group
        for group in groups:
            # Check if this term commutes with all terms in the group
            if all(pauli_commute(pauli_string, existing) for existing in group.keys()):
                group[pauli_string] = coefficient
                placed = True
                break
        
        # If it doesn't fit in any existing group, create a new one
        if not placed:
            groups.append({pauli_string: coefficient})
    
    return groups

def group_commuting_pauli_terms_openfermion_hybrid(hamiltonian: Dict[str, float]) -> List[Dict[str, float]]:
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
                if pauli != 'I':
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

def LiH_hamiltonian(R=1.5, charge=0, spin=0, num_electrons=2, num_orbitals=2) -> Dict[str, float]:
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
    geometry = [('Li', (0.0, 0.0, 0.0)), ('H', (0.0, 0.0, R))]
    
    # Create molecular data object
    molecule = MolecularData(
        geometry=geometry,
        basis='sto-3g',
        multiplicity=spin + 1,  # OpenFermion uses multiplicity = 2S + 1
        charge=charge
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
        occupied_indices=occupied_indices,
        active_indices=active_indices
    )
    
    from openfermion.ops import InteractionOperator
    molecular_hamiltonian_no_nuclear = InteractionOperator(
        constant=0.0,  
        one_body_tensor=molecular_hamiltonian.one_body_tensor,
        two_body_tensor=molecular_hamiltonian.two_body_tensor
    )
    
    # Convert to fermionic operator
    fermionic_op = get_fermion_operator(molecular_hamiltonian_no_nuclear)
    
    # Apply Jordan-Wigner transformation
    H_qubit = jordan_wigner(fermionic_op)
    
    # Convert to dictionary format
    return sparsepauliop_dictionary(H_qubit)

def generate_random_hamiltonian(num_qubits: int) -> Dict[str, float]:
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
    bitstrings = [''.join(bits) for bits in itertools.product('XYZI', repeat=num_qubits)]

    # Create a dictionary with these bitstrings as keys and random numbers as values
    random_values = np.random.random(len(bitstrings)) - 0.5
    H = dict(zip(bitstrings, random_values))

    return H

def LiH_hamiltonian_tapered(R: float) -> Dict[str, float]:
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
    geometry = [('Li', (0.0, 0.0, 0.0)), ('H', (0.0, 0.0, R))]
    
    # Create molecular data object
    molecule = MolecularData(
        geometry=geometry,
        basis='sto-3g',
        multiplicity=1,  # Singlet state
        charge=0
    )
    
    # Run PySCF calculation
    molecule = run_pyscf(molecule, run_scf=True, run_fci=True)
    total_n_elec = molecule.n_electrons
    
    # Apply active space reduction equivalent to tapering
    # Freeze core orbital (1s of Li)
    n_core_orbitals = 1
    occupied_indices = list(range(n_core_orbitals))
    
    active_indices = [1, 2, 5]  
    active_spin_orbitals = 2*len(active_indices)

    active_n_elec = total_n_elec - 2*n_core_orbitals
    
    molecular_hamiltonian = molecule.get_molecular_hamiltonian(
        occupied_indices=occupied_indices,
        active_indices=active_indices
    )

    # Remove nuclear repulsion energy
    from openfermion.ops import InteractionOperator
    molecular_hamiltonian_no_nuclear = InteractionOperator(
        constant=0.0,  # Set constant to 0 to match qiskit_nature
        one_body_tensor=molecular_hamiltonian.one_body_tensor,
        two_body_tensor=molecular_hamiltonian.two_body_tensor
    )
    
    # Convert to fermionic operator
    fermionic_op = get_fermion_operator(molecular_hamiltonian_no_nuclear)
    
    # Apply Bravyi-Kitaev transformation
    qubit_op = symmetry_conserving_bravyi_kitaev(fermionic_op, active_spin_orbitals, active_n_elec)

    return sparsepauliop_dictionary(qubit_op)
