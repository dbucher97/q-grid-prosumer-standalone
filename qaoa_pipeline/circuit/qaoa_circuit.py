from abc import ABC, abstractmethod
import logging

from qaoa_pipeline.utils.layout import QubitLayout
from qaoa_pipeline.circuit.qaoa_parameters import QaoaParameters

import numpy as np

logger = logging.getLogger(__name__)


class QAOACircuit(ABC):
    def __init__(self, layout: QubitLayout):
        self.layout = layout

    @abstractmethod
    def expectation_value(self, qaoa_parameters: QaoaParameters):
        pass

    @abstractmethod
    def sample(self, qaoa_parameters: QaoaParameters):
        pass

    @abstractmethod
    def get_probs_with_sample(
        self, qaoa_parameters: QaoaParameters, prob_threshold: float
    ) -> list[tuple[dict, float]]:
        """
        Generates a list of tuples containing sample states and their corresponding probabilities.

        This method calculates the probabilities of sample states using the provided `betas` and `gammas`
        and returns them alongside the sample states in a list of tuples.

        Args:
            betas (np.ndarray | list): A list or numpy array representing the beta parameters.
            gammas (np.ndarray | list): A list or numpy array representing the gamma parameters.

        Returns:
            list[tuple[dict, float]]: A list of tuples where each tuple consists of:
                - dict: The sample state, represented as a dictionary.
                - float: The probability of the sample state.

        Example:
            Given `betas` and `gammas`, this method would return the corresponding sample states and
            their probabilities as a list of tuples.

            Example output:
            [
                ({"x1": 0, "x2": 1}, 0.12),
                ({"x1": 1, "x2": 0}, 0.08)
            ]
        """
        pass

    def state(self, qaoa_parameters: QaoaParameters):
        """Returns the quantum state if possible"""
        raise NotImplementedError("The method `state` has not been implemented")

    @abstractmethod
    def draw(self, **kwargs):
        """Draw the circuit"""
        pass

    @abstractmethod
    def get_energies(self, idxs: list[int] | np.ndarray | None = None) -> np.ndarray:
        """Returns enegies of selected samples, accessed by index"""
        pass

    @abstractmethod
    def get_circuit_depth(self, num_layers: int) -> float:
        """Returns circuit depth of the constructed circuit"""
        pass

    def get_threshold_probability(
        self, probabilities: list[float], relative_threshold: float, atol: float = 1e-3
    ) -> float:
        """Get the total probability of sampling an sample with energy within the
        relative range to the optimal solution defined by `relative_threshold`.

        Args:
            probabilities: The output of `sample()`
            relative_threshold: The relative threshold, e.g. 0.05 for 5%
            atol: Absolute tolerance (Default: 1e-3)

        Returns:
            The aggregated probability
        """

        energies = self.get_energies()
        min_energy = np.min(energies)
        threshold = min_energy * (1 + relative_threshold)
        return np.sum(probabilities[energies <= threshold + atol])


class GradientQAOACircuit(QAOACircuit):
    @abstractmethod
    def expectation_and_gradient(self, qaoa_parameters: QaoaParameters):
        pass
