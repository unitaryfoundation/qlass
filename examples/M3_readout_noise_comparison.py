
import warnings

import exqalibur
import numpy as np
from perceval.algorithm import Sampler

from qlass import LiH_hamiltonian
from qlass.mitigation import M3Mitigator, PhotonicErrorModel
from qlass.quantum_chemistry.classical_solution import brute_force_minimize
from qlass.vqe import VQE
from qlass.vqe.ansatz import le_ansatz

warnings.simplefilter('ignore')
warnings.filterwarnings('ignore')

# 1. SETUP NOISE MODEL
def create_readout_noise_model(num_modes, error_rate=0.05):
    """
    Creates a PhotonicErrorModel where each mode has a probability 
    of reporting the wrong photon count (readout error).
    Supports up to 2 photons per mode to handle potential bunching/dark counts.
    """
    max_photons = 2
    model = PhotonicErrorModel(num_modes, max_photons_per_mode=max_photons)

    # Define 3x3 probability matrix: P(measured | ideal)
    # Columns: Ideal 0, Ideal 1, Ideal 2
    # Rows: Measured 0, Measured 1, Measured 2

    p = error_rate

    # Model Logic:
    # - Ideal 0: Mostly 0, small chance of 1 (dark count)
    # - Ideal 1: Mostly 1, chance of 0 (loss) or 2 (dark count)
    # - Ideal 2: Mostly 2, chance of 1 (loss)

    # Matrix construction (must ensure columns sum to 1)
    matrix = np.zeros((3, 3))

    # Col 0 (Ideal=0)
    matrix[0, 0] = 1.0 - p      # Meas 0
    matrix[1, 0] = p            # Meas 1 (Dark count)
    matrix[2, 0] = 0.0          # Meas 2 (Double dark count ignored)

    # Col 1 (Ideal=1)
    matrix[0, 1] = p            # Meas 0 (Loss)
    matrix[1, 1] = 1.0 - 2*p    # Meas 1
    matrix[2, 1] = p            # Meas 2 (Dark count / bunching fake)

    # Col 2 (Ideal=2)
    matrix[0, 2] = 0.0          # Meas 0 (Double loss unlikely)
    matrix[1, 2] = p            # Meas 1 (Loss)
    matrix[2, 2] = 1.0 - p      # Meas 2

    # Sanity check
    if matrix[1, 1] < 0:
        raise ValueError(f"Error rate {error_rate} is too high, probability became negative.")

    for i in range(num_modes):
        model.set_mode_calibration(i, matrix)

    return model

# 2. DEFINE EXECUTOR WITH NOISE INJECTION
def make_noisy_executor(noise_model, num_shots=1000):

    def executor(params, pauli_string):
        # 1. Get the ideal processor for the given ansatz and parameters
        processor = le_ansatz(params, pauli_string)

        # 2. Sample from the ideal circuit
        sampler = Sampler(processor)
        # We need raw samples to inject noise
        # Sampler.samples returns a list of FockStates or similar
        # le_ansatz sets input state internally usually

        try:
            samples = sampler.samples(num_shots)["results"]
        except Exception:
             # Fallback if structure varies
             samples = sampler.samples(num_shots)
             if isinstance(samples, dict) and 'results' in samples:
                 samples = samples['results']

        # 3. Inject Readout Noise
        noisy_samples = []
        for sample in samples:
            # sample is typically a FockState object, e.g. |0, 1>
            # We need to convert to list of ints
            ideal_state = list(sample)
            measured_state = list(ideal_state)

            for i in range(noise_model.num_modes):
                 # Get probability column for the ideal photon count
                 ideal_n = ideal_state[i]

                 # Cap at max calibrated photons to avoid index error
                 if ideal_n > noise_model.max_photons:
                     ideal_n = noise_model.max_photons

                 probs = noise_model.calibration_data[i][:, ideal_n]

                 # Sample measured value
                 measured_val = np.random.choice(len(probs), p=probs)
                 measured_state[i] = measured_val

            # Wrap in FockState so utils.py treats it as photonic data
            noisy_samples.append(exqalibur.FockState(tuple(measured_state)))

        return noisy_samples

    return executor

# 3. MAIN BENCHMARK
def run_benchmark():
    print("--- M3 Mitigation Benchmark ---")

    # Setup problem
    num_qubits = 2
    hamiltonian = LiH_hamiltonian(R=0.1, num_electrons=2, num_orbitals=1)
    print(hamiltonian)

    num_modes = 2 * num_qubits
    readout_error = 0.1
    print(f"Readout Error Rate: {readout_error}")

    noise_model = create_readout_noise_model(num_modes, readout_error)
    mitigator = M3Mitigator(noise_model)

    # Create executors
    noisy_exec = make_noisy_executor(noise_model, num_shots=10000)

    # n_local(reps=1) on 2 qubits with Ry gates.
    # Structure: Ry layer - CX layer - Ry layer.
    # 2 params (1st layer) + 0 params (entanglement) + 2 params (2nd layer) = 4 params.
    num_params_guess = 4

    # --- Run Unmitigated ---
    print("\nRunning Unmitigated VQE...")
    vqe_unmitigated = VQE(
        hamiltonian,
        noisy_exec,
        num_params=num_params_guess,
        optimizer="COBYLA",
        executor_type="sampling"
    )
    energy_unmitigated = vqe_unmitigated.run(max_iterations=30)
    print(f"Unmitigated Energy: {energy_unmitigated:.4f}")

    # --- Run Mitigated ---
    print("\nRunning Mitigated VQE (M3)...")
    vqe_mitigated = VQE(
        hamiltonian,
        noisy_exec,
        num_params=num_params_guess,
        optimizer="COBYLA",
        executor_type="sampling",
        mitigator=mitigator
    )
    energy_mitigated = vqe_mitigated.run(max_iterations=30)
    print(f"Mitigated Energy: {energy_mitigated:.4f}")

    # Calculate Exact Energy
    exact_energy = brute_force_minimize(hamiltonian)
    print(f"Exact Energy:        {exact_energy:.4f}")

    print("\n--- Results ---")
    print(f"Exact:       {exact_energy:.4f}")
    print(f"Unmitigated: {energy_unmitigated:.4f}")
    print(f"Mitigated:   {energy_mitigated:.4f}")
    print(f"Improvement: {abs(energy_unmitigated - energy_mitigated):.4f}")
    print(f"Error (Unmit): {abs(energy_unmitigated - exact_energy):.4f}")
    print(f"Error (Mit):   {abs(energy_mitigated - exact_energy):.4f}")

    if abs(energy_mitigated - exact_energy) < abs(energy_unmitigated - exact_energy):
        print("PASS: Mitigation result is closer to exact energy.")
    else:
        print("Note: Mitigation did not reduce absolute error to exact.")

if __name__ == "__main__":
    run_benchmark()
