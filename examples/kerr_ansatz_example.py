from collections import Counter

import numpy as np

from qlass.vqe.ansatz import kerr_ansatz


def kerr_sampling_executor(
    params: np.ndarray, n_max: int = 4, num_kerr: int = 2, num_samples: int = 1000
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    An executor that computes the outcome probability distribution
    from the kerr_ansatz unitary and draws samples from it.
    """
    dim = n_max + 1

    # 1. Define the unitary using kerr_ansatz
    U = kerr_ansatz(params, num_kerr=num_kerr, n_max=n_max)

    # 2. Define the initial state
    # We use a superposition state 1/√2 (|2, 0> + |0, 2>) to manifest nonlinear interference
    state = np.zeros(dim**2, dtype=complex)
    state[2 * dim + 0] = 1.0 / np.sqrt(2)
    state[0 * dim + 2] = 1.0 / np.sqrt(2)

    # 3. Evolve the state through the circuit
    final_state = U @ state

    # 4. Compute the exact probability distribution
    probs = np.abs(final_state) ** 2

    # Ensure probabilities sum to 1 (to mitigate any minor floating point artifacts)
    probs = probs / np.sum(probs)

    # 5. Draw samples based on the computed probability distribution
    indices = np.arange(dim**2)
    samples = np.random.choice(indices, size=num_samples, p=probs)

    return samples, probs, dim


def main() -> None:
    n_max = 4
    num_samples = 5000

    # params: [gate0, gate1, gate2, gate3, bs_theta, bs_phi]
    kappa_phi = 1.0
    bs_theta = 0.5
    params = np.array([kappa_phi, 0.0, 0.0, 0.0, bs_theta, 0.0])

    print("=== Kerr Ansatz Sampling Executor Example ===\n")
    print(f"Drawing {num_samples} samples from the nonlinear kerr_ansatz circuit...\n")

    # Create the sample execution
    samples, exact_probs, dim = kerr_sampling_executor(
        params, n_max=n_max, num_kerr=2, num_samples=num_samples
    )

    # Tally the frequencies of the drawn samples
    counts = Counter(samples)

    print("Outcome | Exact Prob | Sampled Freq | Difference")
    print("-" * 52)

    # Sort states by their exact probability
    top_indices = np.argsort(exact_probs)[::-1]

    for idx in top_indices:
        prob = exact_probs[idx]
        if prob > 1e-4:  # Only show states with meaningful measurement probability
            n0, n1 = idx // dim, idx % dim

            sampled_freq = counts.get(idx, 0) / num_samples
            diff = abs(prob - sampled_freq)

            print(
                f" |{n0}, {n1}>  | {prob * 100:8.2f}% | {sampled_freq * 100:11.2f}% | {diff * 100:8.2f}%"
            )


if __name__ == "__main__":
    main()
