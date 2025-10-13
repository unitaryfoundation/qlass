"""
Module for handling molecular Hamiltonians using OpenFermion.
"""

from typing import Optional, Tuple, List, Union
import numpy as np
from openfermion import (
    MolecularData,
    get_fermion_operator,
    get_sparse_operator,
    jordan_wigner,
    InteractionOperator,
    FermionOperator
)
from openfermion.chem import make_reduced_hamiltonian
from openfermion.transforms import get_quadratic_hamiltonian
from openfermion.linalg import get_ground_state


class MolecularHamiltonian:
    """
    A class for handling molecular Hamiltonians using OpenFermion.
    
    This class provides functionality for:
    - Creating molecular Hamiltonians from molecular data
    - Converting between different representations (FermionOperator, QubitOperator)
    - Computing ground state energies
    - Analyzing molecular properties
    """
    
    def __init__(
        self,
        geometry: List[Tuple[str, Tuple[float, float, float]]],
        basis: str = 'sto-3g',
        multiplicity: int = 1,
        charge: int = 0,
        description: str = ''
    ):
        """
        Initialize a MolecularHamiltonian instance.
        
        Args:
            geometry: List of tuples containing (atom, (x, y, z)) coordinates
            basis: The basis set to use (default: 'sto-3g')
            multiplicity: The spin multiplicity (2S + 1)
            charge: The molecular charge
            description: Optional description of the molecule
        """
        self.geometry = geometry
        self.basis = basis
        self.multiplicity = multiplicity
        self.charge = charge
        self.description = description
        
        # Initialize molecular data
        self.molecule = MolecularData(
            geometry=geometry,
            basis=basis,
            multiplicity=multiplicity,
            charge=charge,
            description=description
        )
        
        # Initialize Hamiltonian-related attributes
        self.fermion_hamiltonian: Optional[FermionOperator] = None
        self.qubit_hamiltonian = None
        self.sparse_hamiltonian = None
        self.ground_state_energy = None
        
    def compute_hamiltonian(self) -> None:
        """
        Compute the molecular Hamiltonian in various representations.
        """
        # Get the fermion Hamiltonian
        self.fermion_hamiltonian = get_fermion_operator(self.molecule.get_molecular_hamiltonian())
        
        # Convert to qubit Hamiltonian using Jordan-Wigner transform
        self.qubit_hamiltonian = jordan_wigner(self.fermion_hamiltonian)
        
        # Get sparse matrix representation
        self.sparse_hamiltonian = get_sparse_operator(self.qubit_hamiltonian)
        
    def get_ground_state(self) -> Tuple[float, np.ndarray]:
        """
        Compute the ground state energy and wavefunction.
        
        Returns:
            Tuple containing (ground_state_energy, ground_state_wavefunction)
        """
        if self.sparse_hamiltonian is None:
            self.compute_hamiltonian()
            
        energy, wavefunction = get_ground_state(self.sparse_hamiltonian)
        self.ground_state_energy = energy
        return energy, wavefunction
    
    def get_reduced_hamiltonian(
        self,
        active_indices: List[int],
        occupied_indices: Optional[List[int]] = None
    ) -> InteractionOperator:
        """
        Get the reduced Hamiltonian for a subset of orbitals.
        
        Args:
            active_indices: List of indices for active orbitals
            occupied_indices: List of indices for occupied orbitals
            
        Returns:
            Reduced Hamiltonian as an InteractionOperator
        """
        if self.fermion_hamiltonian is None:
            self.compute_hamiltonian()
            
        return make_reduced_hamiltonian(
            self.fermion_hamiltonian,
            active_indices,
            occupied_indices
        )
    
    def get_quadratic_hamiltonian(self) -> 'QuadraticHamiltonian':
        """
        Get the quadratic part of the Hamiltonian.
        
        Returns:
            QuadraticHamiltonian object
        """
        if self.fermion_hamiltonian is None:
            self.compute_hamiltonian()
            
        return get_quadratic_hamiltonian(self.fermion_hamiltonian)
    
    def get_molecular_properties(self) -> dict:
        """
        Get basic molecular properties.
        
        Returns:
            Dictionary containing molecular properties
        """
        return {
            'n_electrons': self.molecule.n_electrons,
            'n_orbitals': self.molecule.n_orbitals,
            'n_qubits': self.molecule.n_qubits,
            'nuclear_repulsion': self.molecule.nuclear_repulsion,
            'multiplicity': self.multiplicity,
            'charge': self.charge
        } 