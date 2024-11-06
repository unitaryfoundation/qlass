
from qiskit_aer import QasmSimulator
from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer

# number of qubits = num_orbitals * 2

def get_qubit_hamiltonian(dist='1.5', charge=0, spin=0, num_electrons=2, num_orbitals=2):
    backend = QasmSimulator(method='statevector')   
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

    return H_qubit