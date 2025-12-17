[![Repository](https://img.shields.io/badge/GitHub-5C5C5C.svg?logo=github)](https://github.com/unitaryfund/qlass)
[![build](https://github.com/unitaryfoundation/qlass/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/unitaryfoundation/qlass/actions)
[![European Union](https://img.shields.io/badge/Supported%20By-%20The%20EU-004494.svg)]([https://wellcomeleap.org](https://cordis.europa.eu/project/id/101135876))
[![Unitary Foundation](https://img.shields.io/badge/Supported%20By-Unitary%20Foundation-FFFF00.svg)](https://unitary.foundation)
[![Documentation Status](https://readthedocs.org/projects/qlass/badge/?version=stable)](https://qlass.readthedocs.io/en/stable/)
[![codecov](https://codecov.io/gh/unitaryfoundation/qlass/branch/main/graph/badge.svg)](https://codecov.io/gh/unitaryfoundation/qlass)
[![PyPI version](https://badge.fury.io/py/qlass.svg)](https://badge.fury.io/py/qlass)
[![Downloads](https://static.pepy.tech/personalized-badge/qlass?period=total&units=international_system&left_color=black&right_color=green&left_text=Downloads)](https://www.pepy.tech/projects/qlass)
[![License](https://img.shields.io/github/license/unitaryfoundation/qlass)](https://github.com/unitaryfoundation/qlass/blob/main/LICENSE)
[![Discord Chat](https://img.shields.io/badge/dynamic/json?color=blue&label=Discord&query=approximate_presence_count&suffix=%20online.&url=https%3A%2F%2Fdiscord.com%2Fapi%2Finvites%2FJqVGmpkP96%3Fwith_counts%3Dtrue)](http://discord.unitary.fund)

# qlass
`qlass` is a package to compile quantum algorithms on photonic devices. Part of the Quantum Glass-based Photonic Integrated Circuits ([QLASS](https://www.qlass-project.eu/))
project funded by the European Union. 


## Installing `qlass`


### Stable release (PyPI)

To install the latest stable release of `qlass` from [PyPI](https://pypi.org/project/qlass/):

```bash
pip install qlass
```

### Development install

To install the development version, set the working directory to the top level of the repository and run:

```bash
uv sync --all-groups
```

If you don't have `uv` installed, you can checkout the installation instructions [here](https://docs.astral.sh/uv/getting-started/installation/).
Or, you can install it via pip:

```bash
pip install uv
```

`qlass` builds upon open-source scientific software packages in Python: `scipy` for numerical optimization, `pyscf` and `openfermion` for quantum chemistry, `qiskit` for quantum computing, and `perceval` for quantum photonics compilation. Optionally, one can also use the [`piquasso`](https://piquasso.readthedocs.io/) package for quantum optics simulations, which can offer improved performance compared to `perceval` in certain regimes.

## Getting started
You can use [this demo notebook on the variational quantum eigensolver (VQE)](https://github.com/unitaryfoundation/qlass/blob/main/notebooks/demo.ipynb) to get started with the `qlass` package, or check out the [example script](https://github.com/unitaryfoundation/qlass/blob/main/examples/vqe_example.py).

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

### Variational Quantum Eigensolver (VQE)

`qlass` includes tools for implementing the Variational Quantum Eigensolver on photonic quantum computers:

```python
from qlass.vqe import VQE, le_ansatz
from qlass.utils import loss_function
from qlass.quantum_chemistry import LiH_hamiltonian

from perceval.algorithm import Sampler

# Generate a Hamiltonian for the LiH molecule
hamiltonian = LiH_hamiltonian(num_electrons=2, num_orbitals=1)

# Define an executor function that uses the linear entangled ansatz
def executor(params, pauli_string):
    processor = le_ansatz(params, pauli_string)
    sampler = Sampler(processor)
    samples = sampler.samples(10_000)
    return samples

# Initialize the VQE solver
vqe = VQE(
    hamiltonian=hamiltonian,
    executor=executor,
    num_params=4, # Number of parameters in the linear entangled ansatz
)

# Run the VQE optimization
vqe_energy = vqe.run(
    max_iterations=10,
    verbose=True
)
```

### Quantum Chemistry

The package provides tools for working with quantum chemistry Hamiltonians:

```python
from qlass.quantum_chemistry import LiH_hamiltonian, brute_force_minimize

# Generate a Hamiltonian for the LiH molecule
hamiltonian = LiH_hamiltonian(num_electrons=2, num_orbitals=1)

# Calculate the exact ground state energy for comparison
exact_energy = brute_force_minimize(hamiltonian)
```

## Module Structure

The `qlass` package is organized into several modules:

- `qlass.compiler`: Functions for compiling quantum circuits to photonic processors
- `qlass.quantum_chemistry`: Tools for generating and manipulating Hamiltonians
- `qlass.vqe`: VQE ansatz implementations for photonic quantum computing
- `qlass.utils`: Utility functions for executing algorithms and processing results

## Documentation
The main functions of the package are commented using the Google style format and can be found [here](https://qlass.readthedocs.io/en/latest/).

## Contributing
`qlass` is developed by the [Unitary Foundation](https://unitary.foundation/), in collaboration with QLASS performers.

You can join the UF [Discord server](http://discord.unitary.fund) for community support.

For a guide to opening a PR, checkout the [contributing guide](https://github.com/unitaryfoundation/qlass/blob/contributing/CONTRIBUTING.md).

## Funding
Funded by the European Union. Views and opinions expressed are however those of the authors only and do not necessarily reflect those of the European Union. Neither the European Union nor the granting authority can be held responsible for them.
