
import numpy as np
import pytest
from collections import Counter
from qlass.mitigation.m3 import PhotonicErrorModel, M3Mitigator

def simulate_noisy_experiment(ideal_dist, error_model, num_shots):
    """
    Simulates a noisy experiment by applying the error model to an ideal distribution.
    (Copied from M3-rem.py for verification)
    """
    noisy_counts = Counter()
    ideal_states = list(ideal_dist.keys())
    ideal_probs = np.array(list(ideal_dist.values()))
    shots_for_states = np.random.choice(len(ideal_states), size=num_shots, p=ideal_probs)

    for state_idx in shots_for_states:
        ideal_state = ideal_states[state_idx]
        measured_state = list(ideal_state)
        for i in range(error_model.num_modes):
            probs = error_model.calibration_data[i][:, ideal_state[i]]
            measured_state[i] = np.random.choice(len(probs), p=probs)
        noisy_counts[tuple(measured_state)] += 1

    return dict(noisy_counts)

def calculate_tvd(dist1, dist2):
    """Calculates Total Variation Distance."""
    all_states = set(dist1.keys()) | set(dist2.keys())
    tvd = 0.0
    for state in all_states:
        p1 = dist1.get(state, 0.0)
        p2 = dist2.get(state, 0.0)
        tvd += abs(p1 - p2)
    return 0.5 * tvd

def test_m3_mitigation():
    # 1. SETUP
    NUM_MODES = 3
    MAX_PHOTONS = 2
    NUM_SHOTS = 10000

    ideal_distribution = {
        (1, 0, 1): 0.80,
        (1, 1, 0): 0.15,
        (0, 0, 1): 0.05,
    }

    error_model = PhotonicErrorModel(NUM_MODES, MAX_PHOTONS)

    # Mode 0
    mode0_errors = np.array([
        [0.98, 0.15, 0.30],
        [0.02, 0.84, 0.15],
        [0.00, 0.01, 0.55],
    ])
    error_model.set_mode_calibration(0, mode0_errors)

    # Mode 1
    mode1_errors = np.array([
        [0.95, 0.05, 0.10],
        [0.05, 0.94, 0.05],
        [0.00, 0.01, 0.85],
    ])
    error_model.set_mode_calibration(1, mode1_errors)

    # Mode 2
    mode2_errors = np.array([
        [0.97, 0.10, 0.20],
        [0.03, 0.89, 0.10],
        [0.00, 0.01, 0.70],
    ])
    error_model.set_mode_calibration(2, mode2_errors)

    # 2. RUN
    noisy_counts = simulate_noisy_experiment(ideal_distribution, error_model, NUM_SHOTS)
    noisy_distribution = {k: v / NUM_SHOTS for k, v in noisy_counts.items()}

    mitigator = M3Mitigator(error_model)
    mitigated_distribution = mitigator.mitigate(noisy_counts)

    # 3. VERIFY
    tvd_noisy = calculate_tvd(ideal_distribution, noisy_distribution)
    tvd_mitigated = calculate_tvd(ideal_distribution, mitigated_distribution)
    
    print(f"TVD Noisy: {tvd_noisy}")
    print(f"TVD Mitigated: {tvd_mitigated}")

    # Mitigation should improve the result significantly (lower TVD)
    # Allow some slack for random fluctuations, but generally mitigated should be better
    # or very close if noise is low. Here noise is significant.
    assert tvd_mitigated < tvd_noisy, "Mitigation failed to reduce error."
    assert tvd_mitigated < 0.05, "Mitigated result is not close enough to ideal."

if __name__ == "__main__":
    test_m3_mitigation()
