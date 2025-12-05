Installation
============

Prerequisites
------------

Before installing ``qlass``, ensure you have the following:

- Python 3.9 or newer
- pip (Python package manager)


Stable install from PyPI
-----------------------

To install the latest stable release of ``qlass`` from PyPI using `uv`:

.. code-block:: bash

   pip install qlass

This will install ``qlass`` and all its dependencies from the Python Package Index.

Installing from Source
---------------------

To install the development version from source:

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/unitaryfund/qlass.git

   # Navigate to the repository directory
   cd qlass

   # Install the package in development mode using uv
   uv sync --all-groups

This will install ``qlass`` and all its dependencies necessary for development and testing.

If you don't have `uv` installed, you can checkout the installation instructions at https://docs.astral.sh/uv/getting-started/installation/. 
Or, you can install it via pip:

.. code-block:: bash

   pip install uv

Dependencies
-----------

``qlass`` relies on the following packages, which will be automatically installed:

- qiskit (version 1.4.2)
- numpy
- perceval-quandela (version 0.12.1)
- qiskit_aer
- qiskit_nature
- pyscf
- tqdm
- numba

Verifying Installation
---------------------

To verify that ``qlass`` has been installed correctly, you can run a simple test:

.. code-block:: python

   import qlass
   print(qlass.__version__)

   # Try importing a few key modules
   from qlass import compile
   from qlass.quantum_chemistry import LiH_hamiltonian
   from qlass.vqe import le_ansatz
   
   print("qlass installed successfully!")
