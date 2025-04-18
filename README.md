[![Repository](https://img.shields.io/badge/GitHub-5C5C5C.svg?logo=github)](https://github.com/unitaryfund/qlass)
[![European Union](https://img.shields.io/badge/Supported%20By-%20The%20EU-004494.svg)]([https://wellcomeleap.org](https://cordis.europa.eu/project/id/101135876))
[![Discord Chat](https://img.shields.io/badge/dynamic/json?color=blue&label=Discord&query=approximate_presence_count&suffix=%20online.&url=https%3A%2F%2Fdiscord.com%2Fapi%2Finvites%2FJqVGmpkP96%3Fwith_counts%3Dtrue)](http://discord.unitary.fund)


# qlass
`qlass` is a package to compile quantum algorithms on photonic devices. Part of the Quantum Glass-based Photonic Integrated Circuits ([QLASS](https://www.qlass-project.eu/))
project funded by the European Union. 

## Installing `qlass`
The development install of `qlass` requirements can be done by setting the working directory to the top level of the repository and running `pip install -e .`. 
QLASS builds upon open-source scientific software packages in Python: `scipy` for numerical optimization, `pyscf` and `qiskit-nature` for quantum chemistry, `qiskit` for quantum computing, and `perceval` for quantum photonics compilation.

## Getting started
You can use [this notebook on the variational quantum eigensolver (VQE)](https://github.com/unitaryfund/qlass/blob/main/photonic_vqe.ipynb) to get started with the QLASS package, or check out the [example script](https://github.com/unitaryfoundation/qlass/blob/main/examples/vqe_example.py).

## Features

### Circuit Compilation

`qlass` provides a convenient function to compile Qiskit quantum circuits to Perceval processors:

```python
from qiskit import QuantumCircuit
from qlass import compile

# Create a Qiskit circuit
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)

# Compile to Perceval processor
processor = compile(qc)
```

## Documentation
The main functions of the package are commented using the Google style format and can be found [here](https://qlass.readthedocs.io/en/latest/).

## Contributing
`qlass` is developed by the [Unitary Foundation](https://unitary.foundation/), in collaboration with QLASS performers.

You can join the UF [Discord server](http://discord.unitary.fund) for community support.

## Funding
Funded by the European Union. Views and opinions expressed are however those of the authors only and do not necessarily reflect those of the European Union. Neither the European Union nor the granting authority can be held responsible for them.