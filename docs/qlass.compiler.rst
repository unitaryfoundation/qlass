qlass.compiler package
===================

The ``compiler`` module provides functionality to convert Qiskit quantum circuits to Perceval photonic processors and analyze their resource requirements. It bridges the gap between the gate-based model and the specific constraints of Linear Optical Quantum Computing (LOQC) :cite:p:`kok2007linear`.

Key Features
------------

**Circuit Compilation**

- ``compile``: Convert Qiskit circuits to Perceval processors. This relies on the backend conversions provided by the Perceval platform :cite:p:`heurtel2023perceval`.

**Resource-Aware Analysis**

In photonic hardware, ideal gates are often probabilistic or resource-intensive. The Resource Aware Compiler models physical imperfections unique to photonics, such as Hong-Ou-Mandel (HOM) interference visibility and fusion gate probabilities :cite:p:`kok2007linear`.

- ``ResourceAwareCompiler``: Compile circuits with hardware resource analysis
- ``HardwareConfig``: Define photonic hardware specifications including:
  
  - Photon loss per component (dB)
  - Fusion gate success probability (critical for probabilistic entangling gates)
  - Hong-Ou-Mandel (HOM) visibility (measure of photon indistinguishability)
  
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

.. bibliography:: refs.bib
   :filter: docname in docnames