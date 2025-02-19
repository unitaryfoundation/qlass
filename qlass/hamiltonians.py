import itertools
import numpy as np

from qiskit_aer import QasmSimulator
from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper, ParityMapper
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer

from typing import List, Tuple, Dict, Union
# number of qubits = num_orbitals * 2

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

def get_qubit_hamiltonian(dist='1.5', charge=0, spin=0, num_electrons=2, num_orbitals=2):
    # backend = QasmSimulator(method='statevector')   
    driver = PySCFDriver(
        atom=f"Li 0 0 0; H 0 0 {dist}",
        basis="sto3g",
        charge=charge,
        spin=spin,
        unit=DistanceUnit.ANGSTROM,
    #    unit=DistanceUnit.BOHR,
    )
    problem = driver.run()
    mapper = JordanWignerMapper()
    as_transformer = ActiveSpaceTransformer(num_electrons, num_orbitals)
    as_problem = as_transformer.transform(problem)
    fermionic_op = as_problem.second_q_ops()
    H_qubit = mapper.map(fermionic_op[0]) # this is the qubit hamiltonian

    # convert the SparsePauliOp in dictionary form
    H = sparsepauliop_dictionary(H_qubit)

    return H

def generate_hamiltonian(num_qubits: int) -> Dict[str, float]:
    # Generate all possible 3-letter bitstrings consisting of 'X', 'Y', 'Z', 'I'
    bitstrings = [''.join(bits) for bits in itertools.product('XYZI', repeat=num_qubits)]

    # Create a dictionary with these bitstrings as keys and random numbers as values
    random_values = np.random.random(len(bitstrings)) - 0.5
    H = dict(zip(bitstrings, random_values))

    return H


def LiH_hamiltonian():
    list_R = [0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,2.0,3.0]
    orb1 = []
    orb2 = []
    orb3 = []
    orb4 = []
    orb5 = []
    orb6 = []
    for R in list_R:
        driver = PySCFDriver(
            atom="Li 0 0 0; H 0 0 {}".format(R),
            basis="sto3g",
            charge=0,
            spin=0,
            unit=DistanceUnit.ANGSTROM,
        )
        
        full_problem = driver.run()
        orb1 += [full_problem.orbital_energies[0]]
        orb2 += [full_problem.orbital_energies[1]]
        orb3 += [full_problem.orbital_energies[2]]
        orb4 += [full_problem.orbital_energies[3]]
        orb5 += [full_problem.orbital_energies[4]]
        orb6 += [full_problem.orbital_energies[5]]

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
    # qubit_op = mapper.map(fermionic_op)
    # print("symmetry preserving N_particle:",qubit_op)

    #Use tappering off qubits to reduce the number of qubit by 2
    tapered_mapper = as_problem.get_tapered_mapper(mapper)
    qubit_op = tapered_mapper.map(fermionic_op)
    # print("tappering off Z2 symmetry:",qubit_op)
    # print(type(qubit_op))

    return sparsepauliop_dictionary(qubit_op)