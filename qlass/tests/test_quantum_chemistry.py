"""
Tests for the quantum_chemistry module.
"""

import pytest
import numpy as np
from qlass.quantum_chemistry.quantum_chemistry import MolecularHamiltonian


@pytest.fixture
def h2_molecule():
    """Fixture for H2 molecule geometry."""
    return [('H', (0, 0, 0)), ('H', (0, 0, 0.74))]


@pytest.fixture
def h2_hamiltonian(h2_molecule):
    """Fixture for H2 MolecularHamiltonian instance."""
    return MolecularHamiltonian(
        geometry=h2_molecule,
        basis='sto-3g',
        multiplicity=1,
        charge=0,
        description='H2 molecule'
    )


def test_initialization(h2_molecule):
    """Test proper initialization of MolecularHamiltonian."""
    ham = MolecularHamiltonian(h2_molecule)
    
    assert ham.geometry == h2_molecule
    assert ham.basis == 'sto-3g'  # default value
    assert ham.multiplicity == 1  # default value
    assert ham.charge == 0  # default value
    assert ham.fermion_hamiltonian is None
    assert ham.qubit_hamiltonian is None
    assert ham.sparse_hamiltonian is None
    assert ham.ground_state_energy is None


def test_compute_hamiltonian(h2_hamiltonian):
    """Test Hamiltonian computation."""
    h2_hamiltonian.compute_hamiltonian()
    
    assert h2_hamiltonian.fermion_hamiltonian is not None
    assert h2_hamiltonian.qubit_hamiltonian is not None
    assert h2_hamiltonian.sparse_hamiltonian is not None


def test_get_ground_state(h2_hamiltonian):
    """Test ground state computation."""
    energy, wavefunction = h2_hamiltonian.get_ground_state()
    
    assert isinstance(energy, float)
    assert isinstance(wavefunction, np.ndarray)
    assert h2_hamiltonian.ground_state_energy == energy
    assert h2_hamiltonian.sparse_hamiltonian is not None


def test_get_reduced_hamiltonian(h2_hamiltonian):
    """Test reduced Hamiltonian computation."""
    active_indices = [0, 1]
    reduced_ham = h2_hamiltonian.get_reduced_hamiltonian(active_indices)
    
    assert reduced_ham is not None
    # Check that the reduced Hamiltonian has the correct number of terms
    assert len(reduced_ham.terms) > 0


def test_get_quadratic_hamiltonian(h2_hamiltonian):
    """Test quadratic Hamiltonian computation."""
    quad_ham = h2_hamiltonian.get_quadratic_hamiltonian()
    
    assert quad_ham is not None
    # Check that we can access the quadratic terms
    assert hasattr(quad_ham, 'constant')
    assert hasattr(quad_ham, 'chemical_potential')


def test_get_molecular_properties(h2_hamiltonian):
    """Test molecular properties retrieval."""
    properties = h2_hamiltonian.get_molecular_properties()
    
    assert isinstance(properties, dict)
    assert 'n_electrons' in properties
    assert 'n_orbitals' in properties
    assert 'n_qubits' in properties
    assert 'nuclear_repulsion' in properties
    assert 'multiplicity' in properties
    assert 'charge' in properties
    
    # Check specific values for H2
    assert properties['n_electrons'] == 2
    assert properties['multiplicity'] == 1
    assert properties['charge'] == 0


def test_invalid_geometry():
    """Test initialization with invalid geometry."""
    invalid_geometry = [('H', (0, 0))]  # Missing z-coordinate
    
    with pytest.raises(Exception):
        MolecularHamiltonian(invalid_geometry)


def test_custom_parameters():
    """Test initialization with custom parameters."""
    geometry = [('H', (0, 0, 0)), ('H', (0, 0, 0.74))]
    ham = MolecularHamiltonian(
        geometry=geometry,
        basis='6-31g',
        multiplicity=3,
        charge=1,
        description='H2+ molecule'
    )
    
    assert ham.basis == '6-31g'
    assert ham.multiplicity == 3
    assert ham.charge == 1
    assert ham.description == 'H2+ molecule' 