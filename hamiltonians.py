
from qiskit_aer import QasmSimulator
from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper
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