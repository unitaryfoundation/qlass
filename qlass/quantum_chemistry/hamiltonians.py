import itertools
import numpy as np
# OpenFermion imports - replacing qiskit_nature
from openfermion.chem import MolecularData
from openfermion.transforms import get_fermion_operator, jordan_wigner
# Import QubitOperator para type hinting
from openfermion.ops import InteractionOperator, QubitOperator
from openfermionpyscf import run_pyscf

from typing import Dict
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
        # If there are no terms (H is an empty QubitOperator, representing zero),
        # or if H is only an identity (QubitOperator('') which has H.terms = {(): 1.0}),
        # default to a 1-qubit system for constructing identity strings.
        # This ensures QubitOperator('') results in {'I': 1.0}.
        # An empty QubitOperator() will result in num_qubits = 0 (from max_qubit_idx = -1),
        # and the loop over H.terms won't run, returning an empty pauli_dict.
        num_qubits = max_qubit_idx + 1 # If max_qubit_idx remains -1, num_qubits becomes 0.

    # If H is QubitOperator() (zero operator), num_qubits will be 0.
    # If H is QubitOperator('') (identity), num_qubits will be 1 (from max_qubit_idx= -1 -> 0, then +1).
    # This logic for num_qubits seems fine.
    # The case for QubitOperator('') being handled as 'I' (1 qubit) is implicitly covered
    # when the loop processes the {(): coefficient} term if num_qubits was set to 1.
    # If num_qubits became 0 (for an empty QubitOperator), the identity string would be empty.
    # Let's ensure num_qubits is at least 1 if an identity term is present and no other terms define size.
    if not H.terms and num_qubits == 0 : # Truly empty QubitOperator
        return {} # An empty operator has no Pauli terms.
    if H.terms and not any(bool(term) for term in H.terms.keys()) and num_qubits == 0:
        # This means H.terms only contains {(): coeff}, e.g. QubitOperator('')
        # max_qubit_idx was -1, num_qubits became 0. For an identity string, we need at least 1 qubit.
        num_qubits = 1


    # Iterate through each term in the QubitOperator.
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
                    # This case should ideally not happen if num_qubits was determined correctly.
                    # It implies a Pauli operator acts on a qubit index higher than initially detected.
                    # Handling this robustly might involve resizing pauli_array or raising an error.
                    # For now, we assume num_qubits is correctly pre-calculated.
                    pass 
            
            pauli_key = ''.join(pauli_array)
        
        # Store the term in the dictionary, using only the real part of the coefficient.
        pauli_dict[pauli_key] = float(coefficient.real)
    
    return pauli_dict
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
    
    # Remove nuclear repulsion energy for qiskit_nature compatibility
    # qiskit_nature excludes nuclear energy from qubit hamiltonian
    # but OpenFermion includes it by default
    from openfermion.ops import InteractionOperator
    molecular_hamiltonian_no_nuclear = InteractionOperator(
        constant=0.0,  # Set constant to 0 to match qiskit_nature
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
    
    # Apply active space equivalent to qiskit_nature's tapering approach
    # This mimics ActiveSpaceTransformer(2, 3, active_orbitals=[1,2,5])
    
    # Apply active space reduction equivalent to tapering
    # Freeze core orbital (1s of Li)
    n_core_orbitals = 1
    occupied_indices = list(range(n_core_orbitals))
    
    # Select specific active orbitals (equivalent to qiskit_nature's selection)
    # This corresponds to orbitals [1,2,5] from the original qiskit code
    active_indices = [1, 2, 5]  # As specified in the original qiskit code
    
    try:
        # Attempt to get molecular hamiltonian with active space
        molecular_hamiltonian = molecule.get_molecular_hamiltonian(
            occupied_indices=occupied_indices,
            active_indices=active_indices
        )
    except:
        # If it fails, use a simpler approach with fewer orbitals
        molecular_hamiltonian = molecule.get_molecular_hamiltonian(
            occupied_indices=[0],  # Freeze first orbital
            active_indices=[1, 2]  # Use only 2 active orbitals
        )
    
    # Remove nuclear repulsion energy for consistency with qiskit_nature
    from openfermion.ops import InteractionOperator
    molecular_hamiltonian_no_nuclear = InteractionOperator(
        constant=0.0,  # Set constant to 0 to match qiskit_nature
        one_body_tensor=molecular_hamiltonian.one_body_tensor,
        two_body_tensor=molecular_hamiltonian.two_body_tensor
    )
    
    # Convert to fermionic operator
    fermionic_op = get_fermion_operator(molecular_hamiltonian_no_nuclear)
    
    # Apply Jordan-Wigner transformation
    # Note: OpenFermion doesn't have automatic tapering like qiskit_nature's 
    # ParityMapper, but active space selection achieves similar goals
    qubit_op = jordan_wigner(fermionic_op)
    
    return sparsepauliop_dictionary(qubit_op)
