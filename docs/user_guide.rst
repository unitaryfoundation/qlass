User Guide
==========

Introduction
------------

``qlass`` is a Python package designed to compile quantum algorithms on photonic devices. It is part of the Quantum Glass-based Photonic Integrated Circuits (`QLASS <https://www.qlass-project.eu/>`_) project funded by the European Union.

Installation
-----------

Development Install
~~~~~~~~~~~~~~~~~~

To install ``qlass`` for development, follow these steps:

1. Clone the repository::

    git clone https://github.com/unitaryfund/qlass.git
    cd qlass

2. Create a virtual environment (recommended)::

    python -m venv venv
    source venv/bin/activate  # On Windows, use: venv\Scripts\activate

3. Install in development mode::

    pip install -e .

Dependencies
~~~~~~~~~~~

QLASS builds upon several open-source scientific software packages in Python:

* ``scipy`` - For numerical optimization
* ``pyscf`` and ``qiskit-nature`` - For quantum chemistry
* ``qiskit`` - For quantum computing
* ``perceval`` - For quantum photonics compilation

Getting Started
-------------

Basic Usage
~~~~~~~~~~

Here's a simple example of using ``qlass`` to create and compile a quantum circuit::

    import qlass
    from qiskit import QuantumCircuit

    # Create a simple quantum circuit
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)

    # Compile for photonic hardware
    photonic_circuit = qlass.compile(qc)

For more advanced examples, explore the `variational quantum eigensolver (VQE) notebook <https://github.com/unitaryfund/qlass/blob/main/photonic_vqe.ipynb>`_.

Key Features
-----------

1. Quantum Algorithm Compilation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``qlass`` specializes in compiling quantum algorithms for photonic devices. The package provides tools for:

* Converting quantum circuits to photonic implementations
* Optimizing photonic quantum computations
* Interfacing with quantum chemistry calculations

Example usage for quantum chemistry calculations::

    from qlass.chemistry import MoleculeHandler
    from qlass.compiler import PhotonicCompiler

    # Set up a molecule
    molecule = MoleculeHandler("H2")
    
    # Generate quantum circuit for electronic structure
    circuit = molecule.get_vqe_circuit()
    
    # Compile for photonic hardware
    photonic_circuit = PhotonicCompiler().compile(circuit)

2. Photonic Device Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The package includes features for:

* Working with photonic integrated circuits
* Optimizing quantum operations for photonic hardware
* Simulating photonic quantum computations

Example of working with photonic devices::

    from qlass.devices import PhotonicDevice
    
    # Configure a photonic device
    device = PhotonicDevice(num_modes=4)
    
    # Add components
    device.add_beamsplitter(0, 1)
    device.add_phase_shifter(1)

Common Workflows
--------------

1. Quantum Chemistry Simulations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Step-by-step guide for running quantum chemistry simulations:

1. Define your molecular system
2. Generate the appropriate quantum circuits
3. Compile to photonic operations
4. Execute or simulate the results

2. Circuit Optimization
~~~~~~~~~~~~~~~~~~~~

Best practices for optimizing circuits:

1. Start with a simplified circuit
2. Apply the photonic compiler
3. Use optimization tools for efficiency
4. Validate results through simulation

Troubleshooting
-------------

Common Issues
~~~~~~~~~~~

1. Installation Problems
    * Ensure all dependencies are properly installed
    * Check Python version compatibility (Python 3.8+ recommended)
    * Verify your virtual environment is activated

2. Compilation Errors
    * Verify input circuit validity
    * Check for unsupported quantum operations
    * Ensure sufficient resources for compilation

3. Performance Issues
    * Consider circuit optimization techniques
    * Verify hardware specifications
    * Use appropriate simulation backends

Getting Help
~~~~~~~~~~

If you encounter issues:

1. Check the GitHub issues for similar problems
2. Join the Discord community for real-time help
3. Include relevant code and error messages when seeking help

Support and Community
------------------

* Join the `Unitary Fund Discord server <http://discord.unitary.fund>`_ for community support
* Visit the `GitHub repository <https://github.com/unitaryfund/qlass>`_ for the latest updates
* Report issues and contribute through the GitHub issue tracker

Development and Contributing
-------------------------

``qlass`` is developed by the `Unitary Foundation <https://unitary.foundation/>`_, in collaboration with QLASS performers. The project welcomes contributions from the community.

Contributing Guidelines:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

All code should be documented using the Google style format.

Funding and Attribution
--------------------

This project is funded by the European Union. While the views and opinions expressed are those of the authors, they do not necessarily reflect those of the European Union. Neither the European Union nor the granting authority can be held responsible for them. 