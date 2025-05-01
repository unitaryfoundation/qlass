import itertools
import numpy as np

from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper, ParityMapper
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer

from typing import Dict

from qiskit.quantum_info import SparsePauliOp

def sparsepauliop_dictionary(H: SparsePauliOp) -> Dict[str, float]:
    """
    Convert a Hamiltonian SparsePauliOp form to a dictionary.

    Args:
        H (SparsePauliOp): Hamiltonian

    Returns:
        Dict[str, float]: Dictionary with Pauli string keys and coefficient values
    """
    pauli_strings = list(map(str, H.paulis))
    coeffs = H.coeffs
    
    return dict(zip(pauli_strings, coeffs))

def LiH_hamiltonian(R=1.5, charge=0, spin=0, num_electrons=2, num_orbitals=2) -> Dict[str, float]:
    """
    Generate the qubit Hamiltonian for the LiH molecule at a given bond length.

    Args:
        R (float): Bond length
        charge (int): Charge of the molecule
        spin (int): Spin of the molecule    
        num_electrons (int): Number of electrons
        num_orbitals (int): Number of molecular orbitals

    Returns:
        Dict[str, float]: Hamiltonian dictionary
    """
    
    driver = PySCFDriver(
        atom="Li 0 0 0; H 0 0 {}".format(R),
        basis="sto3g",
        charge=charge,
        spin=spin,
        unit=DistanceUnit.ANGSTROM,
    )

    problem = driver.run()
    mapper = JordanWignerMapper()
    as_transformer = ActiveSpaceTransformer(num_electrons, num_orbitals)
    as_problem = as_transformer.transform(problem)
    fermionic_op = as_problem.second_q_ops()
    H_qubit = mapper.map(fermionic_op[0]) # this is the qubit hamiltonian

    return sparsepauliop_dictionary(H_qubit)

def generate_random_hamiltonian(num_qubits: int) -> Dict[str, float]:
    """
    Generate a random Hamiltonian.

    Args:
        num_qubits (int): Number of qubits

    Returns:
        Dict[str, float]: Hamiltonian dictionary
    """

    # Generate all possible 3-letter bitstrings consisting of 'X', 'Y', 'Z', 'I'
    bitstrings = [''.join(bits) for bits in itertools.product('XYZI', repeat=num_qubits)]

    # Create a dictionary with these bitstrings as keys and random numbers as values
    random_values = np.random.random(len(bitstrings)) - 0.5
    H = dict(zip(bitstrings, random_values))

    return H


def LiH_hamiltonian_tapered(R: float) -> Dict[str, float]:
    """
    Generate the Hamiltonian for the LiH molecule at a given bond length using tapering technique.

    Args:
        R (float): Bond length

    Returns:
        Dict[str, float]: Hamiltonian dictionary
    """

    driver = PySCFDriver(
        atom="Li 0 0 0; H 0 0 {}".format(R),
        basis="sto3g",
        charge=0,
        spin=0,
        unit=DistanceUnit.ANGSTROM,
    )
    
    full_problem = driver.run()
    hamiltonian = full_problem.hamiltonian
    #hamiltonian.nuclear_repulsion_energy  # NOT included in the second_q_op above
    #12 molecular spin orbitals, linear combination of the 1s, 2s and 2px, 2py, 2pz of Li, and 1s of H
    fermionic_op = hamiltonian.second_q_op()

    #Now let's remove the 2px and 2py orbitals from LiH, as they do not contribute to the bonding
    as_transformer = ActiveSpaceTransformer(2, 5) # removing the core, which is globally the 1s of LiH
    as_transformer = ActiveSpaceTransformer(2, 3, active_orbitals=[1,2,5]) # the active space we want
    as_problem = as_transformer.transform(full_problem)

    # assuming that your total system size is 4 electrons in 6 orbitals:
    as_transformer.prepare_active_space(4, 6)

    # after preparation, you can now transform only your Hamiltonian like so
    reduced_hamiltonian = as_transformer.transform_hamiltonian(hamiltonian)
    fermionic_op = reduced_hamiltonian.second_q_op()

    #Parity mapping using conserving number of particles to get 10 qubits instead of 12
    mapper = ParityMapper(num_particles=as_problem.num_particles)

    #Use tappering off qubits to reduce the number of qubit by 2
    tapered_mapper = as_problem.get_tapered_mapper(mapper)
    qubit_op = tapered_mapper.map(fermionic_op)

    return sparsepauliop_dictionary(qubit_op)