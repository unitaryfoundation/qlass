from dataclasses import dataclass

@dataclass
class HardwareConfig:
    """
    A data structure to hold the physical properties of a photonic hardware backend.
    """
    # System-wide transmittance, e.g., 0.9 means 10% photon loss
    transmittance: float = 0.9
    # Small random phase shifts (in radians)
    phase_imprecision: float = 0.01  

    # Photon Loss Parameters
    photon_loss_component_db: float = 0.05  # Loss per optical component in dB
    photon_loss_waveguide_db_per_cm: float = 0.3 # Loss per cm of waveguide in dB

    # Source and Detector Efficiency
    source_efficiency: float = 0.9    # Probability of successfully generating a photon
    detector_efficiency: float = 0.95 # Probability of successfully detecting a photon

    # Gate and Interference Parameters
    fusion_success_prob: float = 0.11  # Prob. of a successful 2-qubit gate fusion event
    hom_visibility: float = 0.97       # Hong-Ou-Mandel visibility for 2-photon interference

    # Average physical properties
    avg_path_length_per_component_cm: float = 0.1 # Estimated path length per component