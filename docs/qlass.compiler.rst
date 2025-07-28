qlass.compiler package
===================

The ``compiler`` module provides functionality to convert Qiskit quantum circuits to Perceval photonic processors and analyze their resource requirements.

Submodules
----------

.. automodule:: qlass.compiler
   :members: compile, ResourceAwareCompiler, generate_report
   :show-inheritance:
   :undoc-members:

Resource-Aware Compilation
-------------------------

The compiler module includes a ``ResourceAwareCompiler`` that analyzes quantum circuits against hardware configurations to estimate real-world performance. The ``HardwareConfig`` dataclass allows you to specify physical properties of your photonic hardware, including photon loss, detector efficiency, and gate success probabilities.

qlass.compiler.hardware_config module
-----------------------------------

.. automodule:: qlass.compiler
   :members: HardwareConfig
   :show-inheritance:
   :undoc-members: