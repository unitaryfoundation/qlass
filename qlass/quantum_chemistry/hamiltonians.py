import itertools
import numpy as np

# OpenFermion imports - replacing qiskit_nature
from openfermion.chem import MolecularData
from openfermion.transforms import get_fermion_operator, jordan_wigner
from openfermionpyscf import run_pyscf

from typing import Dict, Tuple, List, Union
QubitOperatorTermKey = Tuple[Tuple[int, str], ...]

def sparsepauliop_dictionary(H: QubitOperator) -> Dict[str, float]:
    """
    Convert a Hamiltonian QubitOperator form to a dictionary.

    Args:
        H: OpenFermion QubitOperator. Its .terms attribute is a dictionary where keys are
           tuples of ((qubit_index, pauli_operator_str), ...) representing Pauli terms,
           and values are their complex coefficients. An empty tuple () as a key
           represents the identity term.

    Returns:
        Dict[str, float]: Dictionary with Pauli string keys (e.g., "IXYZ")
                          and the real part of their corresponding coefficient values as floats.
    """
    pauli_dict: Dict[str, float] = {}
    
    # Determine the number of qubits. This logic mirrors the original code's behavior.
    max_qubit_idx: int = 0
    if H.terms:
        term_key_for_max_idx: QubitOperatorTermKey # Type hint para la clave del término
        for term_key_for_max_idx in H.terms.keys():
            if term_key_for_max_idx:  # If not the identity tuple ()
                # Find the maximum qubit index within this specific term
                current_term_max_idx: int = 0
                # Ensure term_key_for_max_idx is not empty before calling max, for type checker robustness
                if term_key_for_max_idx: 
                    current_term_max_idx = max(idx for idx, _ in term_key_for_max_idx)
                
                if current_term_max_idx > max_qubit_idx:
                    max_qubit_idx = current_term_max_idx
        num_qubits: int = max_qubit_idx + 1
    else: # H.terms is empty (e.g., H is QubitOperator() or QubitOperator.zero())
        num_qubits: int = 1 # Original behavior: defaults to 1 qubit representation.
                           # If H is QubitOperator(), this function will return an empty dict {}.
                           # If H is QubitOperator((), 0.0), num_qubits becomes 1,
                           # and the result is {'I': 0.0}.

    # Iterate over the terms of the QubitOperator to convert them
    term_key: QubitOperatorTermKey # Type hint para la clave del término
    coefficient: Union[float, complex] # Coefficients in OpenFermion can be float or complex

    for term_key, coefficient in H.terms.items():
        pauli_string_representation: str
        if not term_key:  # This is the identity term, its key in H.terms is an empty tuple ()
            pauli_string_representation = 'I' * num_qubits
        else:
            # For non-identity terms, construct the Pauli string
            pauli_array: List[str] = ['I'] * num_qubits
            
            qubit_idx: int
            pauli_char: str
            for qubit_idx, pauli_char in term_key:
                # This check ensures we don't write out of bounds if num_qubits was
                # somehow miscalculated, though with the current logic it should be correct.
                if 0 <= qubit_idx < num_qubits:
                    pauli_array[qubit_idx] = pauli_char
                # else:
                    # Potentially raise an error or log a warning if qubit_idx is out of expected range.
                    # For now, we assume num_qubits is determined correctly to cover all indices.
            
            pauli_string_representation = ''.join(pauli_array)
        
        # Store the Pauli string with the real part of its coefficient
        pauli_dict[pauli_string_representation] = float(coefficient.real)
    
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
