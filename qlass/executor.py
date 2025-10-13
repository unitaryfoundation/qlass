from qiskit import QuantumCircuit
from qlass import compile

# Create a Qiskit circuit
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)

# Compile to Perceval processor
processor = compile(qc)

import perceval as pcvl

pcvl.pdisplay(processor)