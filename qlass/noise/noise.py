from abc import ABC , abstractmethod
import perceval as pcvl

class PhotonicNoiseModel(ABC):
    """ Abstract base class for all photonic noise models in qlass.
    """
    @abstractmethod
    def to_perceval_model(self) -> pcvl.NoiseModel:
        """Converts the high-level model into a Perceval.NoiseModel instance."""
        pass

class IdealNoiseModel(PhotonicNoiseModel):
    """Represents a perfect, noiseless device."""
    def to_perceval_model(self) -> pcvl.NoiseModel:
        return pcvl.NoiseModel()  # Returns default, noiseless model

class UniformNoiseModel(PhotonicNoiseModel):
    """A generic model with uniform noise parameters."""
    def __init__(self, loss_db: float = 0.0, indistinguishability: float = 1.0,
                    g2: float = 0.0) -> None:
        self.loss_db = loss_db
        self.indistinguishability = indistinguishability
        self.g2 = g2

    def to_perceval_model(self) -> pcvl.NoiseModel:
        transmittance = 10**(-self.loss_db / 10)
        return pcvl.NoiseModel(transmittance=transmittance,
                                indistinguishability=self.indistinguishability,
                                g2=self.g2)