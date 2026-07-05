import warnings
from functools import reduce

import numpy as np
import torch
from scipy.linalg import toeplitz
from scipy.linalg import sqrtm
from tqdm import tqdm
import matplotlib.pyplot as plt

from qlass.quantum_chemistry import (
    LiH_hamiltonian_tapered,
    brute_force_minimize,
)
from qlass.vqe import VQE

warnings.simplefilter("ignore")

supported_lattice_geometries = ["triangular", "squared"]
# supported_rotation_types = ['spiral', 'angle']
supported_rotation_types = ["angle"]
map_N_modes_to_shifter_lengths = {
    6: 2.5e-3,  # [m]
    32: 5.7e-3,  # [m]
}


def generate_coupling_matrix(
    N_wg_rows: int,
    N_wg_cols: int,
    a: float = 30e3,  # (m^-1) Amplitude of the coupling decay: k(x)=a*exp(-b*x)
    b: float = 0.6e6,  # (m^-1) Coupling decay constant: k(x)=a*exp(-b*x)
    pitch: float = 15e-6,  # 20e-6 # (m) Waveguides interdistance # 8e-6,
    lattice: str = "triangular",
):
    # Create row and column link tensors
    row_link = torch.cat((torch.tensor([0, 1]), torch.zeros(N_wg_cols - 2)))
    col_link = torch.cat((torch.tensor([0, 1]), torch.zeros(N_wg_rows - 2)))

    # Compute the Toeplitz-like connectivity matrices using PyTorch
    row_connectivity = torch.tensor(toeplitz(row_link.cpu()))  # Connectivity within one row
    col_connectivity = torch.tensor(toeplitz(col_link.cpu()))  # Connectivity within one column

    # Identity matrices
    Ir = torch.eye(N_wg_cols)
    Ic = torch.eye(N_wg_rows)

    # Compute the initial coupling matrix for a square lattice
    K_matrix = torch.kron(Ic, row_connectivity) + torch.kron(col_connectivity, Ir)

    assert lattice in supported_lattice_geometries, (
        f"lattice must be in {supported_lattice_geometries}"
    )

    if lattice == "triangular":
        # New diagonal link for triangular lattice
        f = torch.diag(torch.ones(N_wg_cols - 1), diagonal=-1)

        # Defines link location
        v1 = torch.arange(N_wg_rows - 1) % 2
        v2 = v1 ^ 1
        v3 = torch.diag(v1, diagonal=-1) + torch.diag(v2, diagonal=1)

        # Diagonal link matrix for triangular lattice
        NL = torch.kron(v3, f)
        NL = NL + NL.T

        K_matrix += NL

    # Apply the coupling decay
    K_matrix = a * torch.exp(-torch.tensor(b * pitch)) * K_matrix

    return K_matrix


def generate_grid_coordinates(
    N_wg_rows,
    N_wg_cols,
    pitch: float = 7e-6,  # [m]
    wg_depth: float = 20e-6,  # [m]
    lateral_wg_depth: float = 7e-6,  # [m]
    lattice: str = "triangular",
    lateral_shifters: bool = False,
):
    """
    Builds two tensors:
    1) shifters_coordinates: defines the location of the shifters for a circuit column.
        Dim: (N_wg_cols + lateral_shifters * N_wg_rows, 2)
        The number of shifters is (N_wg_cols + lateral_shifters * N_wg_rows),
        the second dimension is for their x,y coordinates.
    2) wg_coordinates: defines the location of the guides on the x,y plane.
        Dim: N x 2. The origin of coordinates is at the circuit center (only useful for angle rotation - can be changed
        if this type of rotation is not used)
    """

    # Create a meshgrid of indices using PyTorch
    indices = torch.stack(
        torch.meshgrid(torch.arange(N_wg_cols), torch.arange(N_wg_rows), indexing="ij"), -1
    )

    # Multiply by pitch and transpose
    wg_coordinates = (pitch * indices).permute(1, 0, 2)
    shifters_coordinates = torch.tensor([(i * pitch, -wg_depth) for i in torch.arange(N_wg_cols)])

    if lattice.lower() == "triangular":
        center = pitch * torch.tensor(
            [(N_wg_cols - 1 + 1 / 2) / 2, (N_wg_rows - 1) / 2 * torch.sqrt(torch.tensor(3)) / 2]
        )

        wg_coordinates[:, :, 1] *= (
            torch.sqrt(torch.tensor(3)) / 2
        )  # Set vertical spacing to np.sqrt(3)/2 unit

        wg_coordinates[1::2, :, 0] += (
            pitch * 0.5
        )  # Shift every second row by 0.5 units to the right

        # shifters_coordinates = np.array([(i*pp/2, -dd) for i in np.arange(2*num_columns)], dtype=float) #This is to
        # have twice as many shifters on the top per column

        if lateral_shifters:
            lateral_shifters_coordinates = [
                (
                    (N_wg_cols - 1 + 1 / 2) * pitch + lateral_wg_depth,
                    i * pitch * torch.sqrt(torch.tensor(3)) / 2,
                )
                for i in torch.arange(N_wg_rows)
            ]
            shifters_coordinates = torch.cat(
                (shifters_coordinates, torch.tensor(lateral_shifters_coordinates))
            )

    elif lattice.lower() == "squared":
        center = pitch * torch.tensor([(N_wg_cols - 1) / 2, (N_wg_rows - 1) / 2])

        if lateral_shifters:
            lateral_shifters_coordinates = [
                ((N_wg_cols - 1) * pitch + lateral_wg_depth, i * pitch)
                for i in torch.arange(N_wg_rows)
            ]
            shifters_coordinates = torch.cat(
                (shifters_coordinates, torch.tensor(lateral_shifters_coordinates))
            )

    else:
        raise ValueError("Not a valid configuration")

    wg_coordinates = (wg_coordinates - center).reshape(-1, 2)
    shifters_coordinates -= center

    return shifters_coordinates, wg_coordinates


def apply_angle_rotation(wg_coordinates, n_rotations, rotation_angle):
    if n_rotations <= 0:
        raise ValueError("n_rotations must be greater than 0")

    # Build the 3D circuit as n_rotations x N x 2 tensor 'rotated_lattice'. Each slice is given by a clockwise rotation
    # of the 2D lattice by Angle.
    rotated_lattice = []

    for theta in rotation_angle * torch.arange(n_rotations):
        c, s = torch.cos(theta), torch.sin(theta)
        rot_mat = torch.tensor([[c, -s], [s, c]])
        rotated_lattice.append(
            torch.matmul(rot_mat, wg_coordinates.T).T.reshape(wg_coordinates.shape)
        )

    rotated_lattice = torch.stack(rotated_lattice)
    return rotated_lattice


# TODO: implement this, which seems very complicated, if we need a spiral rotation
def apply_spiral_rotation():
    pass


def compute_log_distances_shifters_wg(
    shifters_coordinates: torch.Tensor,
    wg_coordinates: torch.Tensor,
    lateral_shifters: bool = False,
    n_shifters_cols: int = 1,
    n_rotations: int = 1,  # Number of waveguide rotations in a single shifter
    rotation_width: int = 1,  # Number of lattice steps per spiral rotation
    rotation_angle: float = torch.pi / 6,
    rotation_type: str = "angle",  # spiral or angle: spiral rotation with relabeling or rotation by given angle.
    shifters_len: float = None,
    total_thickness: float = 1e-3,  # [m]
    simulation_steps_per_resistor: int = 1,
):
    N_modes = wg_coordinates.shape[0]
    if shifters_len is None:
        assert N_modes in map_N_modes_to_shifter_lengths, (
            "Length of shifters (shifters_len) is required."
        )
        shifters_len = map_N_modes_to_shifter_lengths[N_modes]

    # z increment
    dz = shifters_len / max(n_rotations, 1) / simulation_steps_per_resistor
    simulation_steps = n_shifters_cols * simulation_steps_per_resistor

    if n_rotations == 0:
        wg_coordinates_3D = wg_coordinates.repeat(simulation_steps, 1, 1)
    else:
        assert rotation_type.lower() in supported_rotation_types, (
            f"Rotation_type must be in {supported_rotation_types}"
        )

        tot_rotations = n_rotations * n_shifters_cols

        if rotation_type == "angle":
            if torch.norm(wg_coordinates[0]) > torch.abs(shifters_coordinates[0][1]):
                raise ValueError(
                    "The waveguides are too close to the surface to allow for rotations! Change wg_depth"
                )
            elif lateral_shifters and torch.norm(wg_coordinates[0]) > torch.abs(
                shifters_coordinates[-1][0]
            ):
                raise ValueError(
                    "They waveguides are too close to the surface to allow for rotations! Change lateral_wg_depth"
                )
            wg_coordinates_3D = apply_angle_rotation(wg_coordinates, tot_rotations, rotation_angle)

    log_distances = (
        -torch.log(
            torch.norm(
                wg_coordinates_3D[:, :, None, :] - shifters_coordinates[None, None, :, :], dim=3
            )
            / total_thickness
        )
    ).reshape(simulation_steps, -1, shifters_coordinates.shape[0])

    log_distances = torch.flip(log_distances, dims=[0]).reshape(
        simulation_steps, -1, shifters_coordinates.shape[0]
    )

    # Dim: n_shifters_cols x (N_modes*num_rotations) x n_shifters_rows. Flip is needed to multiply the exponentials from 0 to s (and not
    # from last to first). This makes the compute_matrix function a bit faster. Flip DT accordingly when printing

    return log_distances, dz


def compute_propagation_constants(
    resistors_temperatures: torch.tensor,
    log_distances: torch.tensor,
    wavelength: float,
    thermo_optic_coefficient: float,
    N_modes: int,
    return_temperatures=False,
):
    # The temperatures in the waveguides depend on the resistors temperature and on their relative log distances
    waveguides_temperatures = (
        (log_distances @ resistors_temperatures.unsqueeze(-1)).squeeze().reshape(-1, N_modes)
    )
    # NOTE: the last reshape is needed if we split a single column of shifters in more segments, due to rotations
    propagation_constants = (
        2 * torch.pi / wavelength * thermo_optic_coefficient * waveguides_temperatures
    )
    # Output shape: (n_shifters_cols * n_rotations, N_modes)
    # NOTE: both the 'waveguide_temperature' and 'propagation_constants' are not absolute values, they are 'Deltas'
    if return_temperatures:
        return propagation_constants, waveguides_temperatures
    else:
        return propagation_constants


def compute_unitary_matrix(
    propagation_constants: torch.tensor, coupling_matrix: torch.tensor, dz: float
):
    # propagation_matrices = [np.diag(diag) for diag in propagation_constants]
    # unitaries = np.array([expm(-1j * dz * (coupling_matrix + prop_mat)) for prop_mat in propagation_matrices])
    # return reduce(np.matmul, unitaries)

    # Create batched diagonal matrices from the propagation constants
    propagation_matrices = torch.diag_embed(propagation_constants)

    # Broadcast coupling_matrix to match the batch size of propagation_matrices
    coupling_matrix_batched = coupling_matrix.unsqueeze(0).expand_as(propagation_matrices)

    # Compute the unitaries using torch operations
    unitaries = torch.matrix_exp(-1j * dz * (coupling_matrix_batched + propagation_matrices))

    # Reduce using matrix multiplication
    result = reduce(torch.matmul, unitaries)

    return result


class ContinuouslyCoupledSimulator:
    """
    A simulator for a continuously coupled bundle of waveguides, i.e. a "cc_chips" chip.
    """

    _N_wg_cols: int

    def __init__(
        self,
        # chip architecture
        N_wg_rows: int,
        N_wg_cols: int,
        n_shifters_cols: int,
        lateral_shifters: bool = False,
        lattice: str = "triangular",
        # physical environment parameters
        Imax: float | int = 24e-3,  # [A], maximum current deliverable to a resistor
        wavelength: float = 900e-9,  # [m]
        thermo_optic_coefficient: float = 7e-6,  # [K^-1]
        thermal_conductivity: float = 1.0,  # W/K glass thermal conductivity
        # physical chip parameters
        resistances: float | int | torch.Tensor = 500,  # [Ohm]
        shifters_len: float = None,
        pitch: float = 7e-6,  # [m] Waveguides interdistance
        a: float = 30e3,  # (m^-1) Amplitude of the coupling decay: k(x)=a*exp(-b*x)
        b: float = 0.6e6,  # (m^-1) Coupling decay constant: k(x)=a*exp(-b*x)
        wg_depth: float = 20e-6,  # [m]
        lateral_wg_depth: float = 7e-6,  # [m]
        simulation_steps_per_resistor: int = 1,
        # rotation parameters
        rotation_type: str = "angle",
        n_rotations: int = 0,  # Number of waveguide rotations in a single shifter
        rotation_width: int = 1,  # Number of lattice steps per spiral rotation
        rotation_angle: float = torch.pi / 6,
        # spiral or angle: spiral rotation with relabeling or rotation by given angle.
        total_thickness: float = 1e-3,  # [m]
        # debug option
        return_temperatures: bool = False,
    ):
        """
        Note that in the variable names, N is used to indicate a number of optical modes (e.g. waveguides), n is used to
        indicate a number of phase shifters (e.g. resistors).
        Also, the row/col denomination is intended for different views of the chip:
         - For the modes (waveguides) we are looking at a *section* of the chip
         - For the shifters (resistors) we are looking at a *top view* of the chip
        """
        self._N_wg_rows = N_wg_rows
        self._N_wg_cols = N_wg_cols
        self._n_shifters_cols = n_shifters_cols
        self._lateral_shifters = lateral_shifters
        self._n_shifters_rows = self._N_wg_cols + lateral_shifters * self._N_wg_rows

        # public attributes
        self.N_modes = self._N_wg_rows * self._N_wg_cols
        self.n_shifters = self._n_shifters_cols * self._n_shifters_rows

        # lattice initialization
        assert lattice in supported_lattice_geometries, (
            f"lattice must be in {supported_lattice_geometries}"
        )
        self._lattice = lattice

        # physical environment parameters
        self._Imax = Imax
        self._wavelength = wavelength
        self._thermo_optic_coefficient = thermo_optic_coefficient
        self._thermal_conductivity = thermal_conductivity

        # resistances initialization
        self._resistances = None
        self.resistances = resistances

        # shifters initialization
        if shifters_len is None:
            assert self.N_modes in map_N_modes_to_shifter_lengths, (
                "Length of shifters (shifters_len) is required."
            )
            shifters_len = map_N_modes_to_shifter_lengths[self.N_modes]

        # physical chip parameters
        self._shifters_len = shifters_len
        self._pitch = pitch
        self._a_coupling = a
        self._b_coupling = b
        self._wg_depth = wg_depth
        self._lateral_wg_depth = lateral_wg_depth

        self._simulation_steps_per_resistor = simulation_steps_per_resistor
        self._dz_step_len = self._shifters_len / simulation_steps_per_resistor

        # rotation parameters
        self._rotation_type = rotation_type
        self._n_rotations = n_rotations
        self._rotation_width = rotation_width
        self._rotation_angle = rotation_angle
        self._total_thickness = total_thickness

        # coupling matrix
        self.coupling_matrix = generate_coupling_matrix(
            N_wg_rows=self._N_wg_rows,
            N_wg_cols=self._N_wg_cols,
            pitch=self._pitch,
            a=self._a_coupling,
            b=self._b_coupling,
            lattice=self._lattice,
        )

        self._log_distances = None
        self.set_log_distances()

        self.return_temperatures = return_temperatures

    @property
    def resistances(self):
        return self._resistances

    @resistances.setter
    def resistances(self, resistances):
        """
        The resistors must be organized in a (n_shifters_cols, n_shifters_rows) array. This is because the 'rows' of
        phase shifters act 'in parallel' on all the waveguides at the same time. Instead, we model the 'cols' of phase
        shifters as acting 'in series' one after the other.
        """
        validated_resistances_array = torch.zeros((self._n_shifters_cols, self._n_shifters_rows))
        if isinstance(resistances, (float, int)):
            validated_resistances_array = torch.full(
                (self._n_shifters_cols, self._n_shifters_rows), resistances
            )
        elif isinstance(resistances, torch.Tensor):
            if resistances.shape == validated_resistances_array.shape:
                validated_resistances_array = resistances
            else:
                raise ValueError(
                    f"Resistors shape: {resistances.shape}, must be (n_shifters_cols x n_shifters_rows): "
                    f"{validated_resistances_array.shape}."
                )
        else:
            raise TypeError("Resistors must be either a float, an int, or a numpy.ndarray")

        self._resistances = validated_resistances_array

    def set_log_distances(self):
        shifters_coordinates, wg_coordinates = generate_grid_coordinates(
            N_wg_rows=self._N_wg_rows,
            N_wg_cols=self._N_wg_cols,
            pitch=self._pitch,
            wg_depth=self._wg_depth,
            lateral_wg_depth=self._lateral_wg_depth,
            lattice=self._lattice,
            lateral_shifters=self._lateral_shifters,
        )
        log_distances, dz = compute_log_distances_shifters_wg(
            shifters_coordinates=shifters_coordinates,
            wg_coordinates=wg_coordinates,
            lateral_shifters=self._lateral_shifters,
            n_shifters_cols=self._n_shifters_cols,
            n_rotations=self._n_rotations,
            rotation_width=self._rotation_width,
            rotation_angle=self._rotation_angle,
            rotation_type=self._rotation_type,
            shifters_len=self._shifters_len,
            total_thickness=self._total_thickness,
            simulation_steps_per_resistor=self._simulation_steps_per_resistor,
        )
        self._dz_step_len = dz
        self._log_distances = log_distances

    def validate_currents_array(self, currents: None | torch.Tensor):
        validated_currents = None
        if currents is None:
            # Generate random currents using PyTorch
            random_tensor = torch.rand((self._n_shifters_cols, self._n_shifters_rows))
            validated_currents = torch.sqrt(random_tensor * self._Imax**2)
        elif isinstance(currents, torch.Tensor):
            if currents.shape == self._resistances.shape:
                validated_currents = currents.clone()  # Use clone to copy in PyTorch
            else:
                raise ValueError(
                    f"`currents` tensor shape does not match `resistors` shape {self._resistances.shape}"
                )
        else:
            raise TypeError("`currents` must be a torch tensor or None")
        return validated_currents

    def simulate(self, currents: torch.Tensor = None):
        # Validate currents using the updated method
        validated_currents = self.validate_currents_array(currents)

        # Compute powers using PyTorch square operation
        powers = torch.square(validated_currents) * self._resistances

        # Compute temperatures using PyTorch operations
        temperatures = powers / (torch.pi * self._shifters_len * self._thermal_conductivity)

        # Repeat the temperatures along a specified axis using PyTorch's repeat function
        temperatures_array = temperatures.repeat(self._simulation_steps_per_resistor, 1)

        if self.return_temperatures:
            # Compute propagation constants and waveguide temperatures using the updated compute function
            propagation_constants, wg_temperatures = compute_propagation_constants(
                resistors_temperatures=temperatures_array,
                log_distances=self._log_distances,
                wavelength=self._wavelength,
                thermo_optic_coefficient=self._thermo_optic_coefficient,
                N_modes=self.N_modes,
                return_temperatures=self.return_temperatures,
            )
            unitary_matrix = compute_unitary_matrix(
                propagation_constants=propagation_constants,
                coupling_matrix=self.coupling_matrix,
                dz=self._dz_step_len,
            )
            return validated_currents, unitary_matrix, wg_temperatures
        else:
            propagation_constants = compute_propagation_constants(
                resistors_temperatures=temperatures_array,
                log_distances=self._log_distances,
                wavelength=self._wavelength,
                thermo_optic_coefficient=self._thermo_optic_coefficient,
                N_modes=self.N_modes,
                return_temperatures=self.return_temperatures,
            )
            unitary_matrix = compute_unitary_matrix(
                propagation_constants=propagation_constants,
                coupling_matrix=self.coupling_matrix,
                dz=self._dz_step_len,
            )
            if currents is None:
                return validated_currents, unitary_matrix
            else:
                return unitary_matrix


def iterative_unitary(A, max_iter=10, tolerance=1e-10):
    """Löwdin iterative orthogonalization"""
    U = A  # Start with the input matrix

    for _ in range(max_iter):
        # Compute U†U
        UHU = U.conj().T @ U

        # Compute the error
        error = np.max(np.abs(UHU - np.eye(U.shape[0], dtype=U.dtype)))

        if error < tolerance:
            break

        # Update U using Löwdin's method
        U = U @ np.linalg.inv(sqrtm(UHU))

    return U


def check_unitarity(U, tolerance=1e-10):
    """Check if a matrix is unitary within given tolerance"""
    identity = np.eye(U.shape[0], dtype=U.dtype)
    UHU = U.conj().T @ U
    max_diff = np.max(np.abs(UHU - identity))
    return max_diff, max_diff < tolerance


# if __name__ == "__main__":
# import matplotlib.pyplot as plt

# N_wg_rows = 4
# N_wg_cols = 8
# n_shifters_cols = 1
# n_shifters_rows = N_wg_cols

# simulator = ContinuouslyCoupledSimulator(
#     N_wg_rows=N_wg_rows,
#     N_wg_cols=N_wg_cols,
#     n_shifters_cols=n_shifters_cols,
#     simulation_steps_per_resistor=1,
#     pitch=7e-6,
# )
# # currents = np.zeros((n_shifters_cols, n_shifters_rows))
# currents, unitary = simulator.simulate()
# print(currents)
# # print(unitary.shape)
# # print(unitary@np.conjugate(unitary.T))
# print(unitary)
# plt.imshow(torch.abs(unitary))
# plt.show()

# simulator = ContinuouslyCoupledSimulator(
#     N_wg_rows=N_wg_rows,
#     N_wg_cols=N_wg_cols,
#     n_shifters_cols=n_shifters_cols,
#     simulation_steps_per_resistor=6543,
# )
# currents, unitary = simulator.simulate(currents)
# plt.imshow(np.abs(unitary))
# plt.show()

N_wg_rows = 2  # 2 or 4
N_wg_cols = 4  # as high as you want
n_shifters_cols = 20
n_shifters_rows = N_wg_cols

simulator = ContinuouslyCoupledSimulator(
    N_wg_rows=N_wg_rows,
    N_wg_cols=N_wg_cols,
    n_shifters_cols=n_shifters_cols,
    simulation_steps_per_resistor=1,
    shifters_len=0.1,
    pitch=7e-6,
)


def unitary_executor(params):
    """
    Example unitary executor: Creates a unitary using parameterized generators.

    For a 2-qubit system, this creates U = exp(-i * sum(params[i] * generator[i]))
    """
    currents = torch.Tensor(np.reshape(params, (n_shifters_cols, n_shifters_rows)))
    U = simulator.simulate(currents=currents)
    U = np.array(U)
    Up = iterative_unitary(U)

    return Up


def main():
    # Generate Hamiltonian
    # hamiltonian = LiH_hamiltonian(num_electrons=2, num_orbitals=1)
    # hamiltonian = LiH_hamiltonian_tapered(R=0.1)
    # exact_energy = brute_force_minimize(hamiltonian)

    # print(f"Exact ground state energy: {exact_energy:.6f}\n")
    # num_params = n_shifters_cols * n_shifters_rows

    # # Initialize VQE with unitary executor
    # vqe = VQE(
    #     hamiltonian=hamiltonian,
    #     executor=unitary_executor,
    #     num_params=num_params,
    #     executor_type="photonic_unitary",  # Explicitly specify type
    # )

    # # Run optimization
    # vqe_energy = vqe.run(max_iterations=1000, verbose=True)

    # 1. Define the range of bond radii to simulate
    radii = np.linspace(0.5, 2.5, 15)
    exact_energies = []
    vqe_energies = []

    # The Hchain_KS Hamiltonian has 2 qubits
    # The 3d device has number of parameters equal to the number of shifters_rows * shifters_cols
    num_params = n_shifters_cols * n_shifters_rows

    # 2. Loop through each radius, run VQE, and store the results
    print("Running VQE simulations for different bond radii...")
    for r in tqdm(radii, desc="Simulating Radii"):
        # Generate the tapered Hamiltonian for the current radius
        # hamiltonian, _, _ = Hchain_KS_hamiltonian(n_hydrogens=4, R=r)
        hamiltonian = LiH_hamiltonian_tapered(R=r)
        # print(hamiltonian)

        # Calculate the exact ground state energy for the theoretical curve
        exact_energy = brute_force_minimize(hamiltonian)
        exact_energies.append(exact_energy)

        # Initialize the VQE solver
        vqe = VQE(
            hamiltonian=hamiltonian,
            executor=unitary_executor,
            num_params=num_params,
            executor_type="photonic_unitary",
            # ancillary_modes=list(range(8, 8+6)),
        )

        # Run the VQE optimization
        vqe_energy = vqe.run(
            max_iterations=1000,  # More iterations for better convergence
            verbose=False,  # Turn off verbose output for the loop
        )
        vqe_energies.append(vqe_energy)

    errors = np.array(vqe_energies) - np.array(exact_energies)
    l2_error = np.linalg.norm(errors)
    avg_error = np.mean(np.abs(errors))
    min_error = np.min(np.abs(errors))
    max_error = np.max(np.abs(errors))

    print(f"L2 error: {l2_error:.6f}")
    print(f"Average error: {avg_error:.6f}")
    print(f"Min error: {min_error:.6f}")
    print(f"Max error: {max_error:.6f}")
    # 3. Plot the results
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.figure(figsize=(12, 7))

    # Plot theoretical (exact) energy curve
    plt.plot(radii, exact_energies, "bo", label="Exact Theoretical Energy", linewidth=2)

    # Plot noisy VQE simulation results
    plt.plot(radii, vqe_energies, "ro", label="Noiseless VQE Simulation", markersize=8)

    plt.xlabel("Internuclear Distance (Å)", fontsize=14)
    plt.ylabel("Energy (Hartree)", fontsize=14)
    plt.title("Noiseless VQE Simulation of tapered LiH hamiltonian with 3D device", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.savefig("LiH_4q_noiseless_3d.png")
    print("\nSimulation complete. Displaying plot.")
    plt.show()

    # Compare results
    # comparison = vqe.compare_with_exact(exact_energy)
    # print("\n--- Results ---")
    # print(f"VQE Energy: {vqe_energy:.6f}")
    # print(f"Exact Energy: {exact_energy:.6f}")
    # print(f"Absolute Error: {comparison['absolute_error']:.6f}")
    # print(f"Relative Error: {comparison['relative_error']:.2%}")

    # Plot convergence
    # vqe.plot_convergence(exact_energy=exact_energy)


if __name__ == "__main__":
    main()
