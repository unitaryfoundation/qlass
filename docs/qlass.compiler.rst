qlass.compiler package
===================

The ``compiler`` module provides functionality to convert Qiskit quantum circuits to Perceval photonic processors and analyze their resource requirements.

Key Features
------------

**Circuit Compilation**

- ``compile``: Convert Qiskit circuits to Perceval processors with configurable backends and noise models utilizing Perceval

**Resource-Aware Analysis**

- ``ResourceAwareCompiler``: Compile circuits with hardware resource analysis
- ``HardwareConfig``: Define photonic hardware specifications including:
  
  - Photon loss per component (dB)
  - Fusion gate success probability
  - Hong-Ou-Mandel (HOM) visibility
  
- ``generate_report``: Display compilation analysis reports with:
  
  - Component counts (beam splitters, phase shifters, fusion gates)
  - Estimated photon loss
  - Overall success probability

Submodules
----------

.. automodule:: qlass.compiler
   :members: compile, ResourceAwareCompiler, HardwareConfig, generate_report
   :show-inheritance:
   :undoc-members: