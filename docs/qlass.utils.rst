qlass.utils package
================

The ``utils`` module provides utility functions for executing quantum algorithms, processing measurement results, and computing loss functions for VQE optimization.

Key Features
------------

**Loss Functions**

- ``loss_function``: Standard loss function for sampling-based VQE
- ``loss_function_matrix``: Loss function for qubit unitary-based VQE
- ``loss_function_photonic_unitary``: Loss function for photonic unitaries with post-selection
- ``e_vqe_loss_function``: Ensemble-VQE loss function supporting multiple states

**Utilities**

- ``linear_circuit_to_unitary``: Converts Perceval linear circuits to unitary matrices
- ``DataCollector``: Collects and stores energy data during ensemble-VQE optimization

  - ``loss_data``: List of cost function values
  - ``energy_data``: List of energy values for each state

Submodules
----------

.. automodule:: qlass.utils
   :members:
   :show-inheritance:
   :undoc-members: