import numpy as np
from dataclasses import dataclass


@dataclass
class QaoaParameters:
    num_layers: int
    deltas: tuple | None

    betas: np.ndarray
    gammas: np.ndarray

    def __post_init__(self):
        self._check_optimizer_setup()

    @classmethod
    def from_deltas(cls, num_layers: int, delta_beta: float, delta_gamma: float):
        """
        generate_init_params(num_layers, delta_beta=None, delta_gamma=None)

        Generates initial parameters for quantum layers, including beta and gamma values.

        Parameters
        ----------
        num_layers : int
            The number of layers for which to generate parameters.
        delta_beta : float, optional
            Scaling factor for beta values (default is None, which means no scaling).
        delta_gamma : float, optional
            Scaling factor for gamma values (default is None, which means no scaling).

        Returns
        -------
        dict
            A dictionary containing the generated beta and gamma values packed with keys 'betas' and 'gammas' respectively.
        """

        betas, gammas = deltas_to_params(
            num_layers=num_layers, delta_beta=delta_beta, delta_gamma=delta_gamma
        )

        return cls(
            num_layers=num_layers,
            deltas=(delta_beta, delta_gamma),
            betas=betas,
            gammas=gammas,
        )

    @classmethod
    def from_deltas_sin2(cls, num_layers: int, delta_beta: float, delta_gamma: float):
        betas, _ = deltas_to_params(num_layers=num_layers, delta_beta=1, delta_gamma=1)
        betas = np.sin(np.pi / 2 * np.sin(np.pi / 2 * betas) ** 2) ** 2
        gammas = 1 - betas

        return cls(
            num_layers=num_layers,
            deltas=(delta_beta, delta_gamma),
            betas=delta_beta * betas,
            gammas=delta_gamma * gammas,
        )

    @classmethod
    def from_beta_gamma(cls, betas, gammas):
        return cls(num_layers=len(betas), deltas=None, betas=betas, gammas=gammas)

    @classmethod
    def from_defined_params(cls, defined_params: np.ndarray):
        if not isinstance(defined_params, np.ndarray):
            raise ValueError(
                "Optimizer requires parameters to be in numpy array format!"
            )
        # print(defined_params)
        betas, gammas = unpack_params(params=defined_params)
        # print(betas)
        # print(gammas)
        return cls(num_layers=len(betas), deltas=None, betas=betas, gammas=gammas)

    @property
    def params(self):
        return np.concatenate([self.betas, self.gammas])

    @property
    def betas_gammas(self):
        return self.betas, self.gammas

    def update(self, type: str, update_array: np.ndarray):
        if type == "params":
            self.update_params(new_params=update_array)
        elif type == "deltas":
            self.update_deltas(deltas=update_array)
        else:
            raise ValueError(
                "Optimizer update type must be either 'params' or 'deltas'."
            )

    def update_params(self, new_params: np.ndarray):
        self.betas, self.gammas = unpack_params(params=new_params)

    def update_deltas(self, deltas: tuple | np.ndarray):
        assert len(deltas) == 2, "deltas must be a tuple of length 2"
        self.deltas = tuple(deltas)

        # recalculate deltas
        self.update_betas_gammas()

    def update_betas_gammas(self):
        delta_beta, delta_gamma = self.deltas
        self.betas, self.gammas = deltas_to_params(
            num_layers=self.num_layers, delta_beta=delta_beta, delta_gamma=delta_gamma
        )

    def interpolate_deltas(self, step_size: int):
        self.num_layers += step_size
        self.update_betas_gammas()

    def interpolate_params(self, step_size: int):
        """
        Interpolates the given parameters of a QAOA circuit.

        This function increases the number of layers (parameters) by the provided step-size,
        and then interpolates the old parameters to fit the new number of layers.

        Args:
            params (np.array): The current beta and gamma parameters of the QAOA circuit.
            step_size (int): The number of new layers to add during interpolation.

        Returns:
            np.array: An array containing the new interpolated beta and gamma parameters packed together.
        """

        def interpolate_array(param_array, step_size):
            p_old = len(param_array)
            p_new = p_old + step_size
            interpolated_params = (
                p_old
                / p_new
                * np.interp(
                    np.linspace(0, 1, p_new), np.linspace(0, 1, p_old), param_array
                )
            )

            return interpolated_params

        self.betas = interpolate_array(self.betas, step_size=step_size)
        self.gammas = interpolate_array(self.gammas, step_size=step_size)
        self.num_layers += step_size

    def _check_optimizer_setup(self):
        """
        Checks the optimizer setup to ensure the parameters (`betas`) match the expected
        number of layers (`num_layers`).

        This method validates if the length of the `betas` attribute corresponds to
        `num_layers`. If they do not match, it raises a ValueError.

        Raises
        ------
        ValueError
            If the length of `betas` does not match `num_layers`.
        """
        # Check if params are of correct length
        if not len(self.betas) == self.num_layers:
            raise ValueError(
                "Number of parameters does not match the number of starting layers"
            )


def unpack_params(params: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Unpacks the combined beta and gamma parameters into separate arrays.

    Args:
        params (np.array): The combined array of beta and gamma parameters.

    Returns:
        tuple: Two arrays, the first containing the beta parameters and the second containing the gamma parameters.
    """
    betas, gammas = params[: len(params) // 2], params[len(params) // 2 :]
    return betas, gammas


def deltas_to_params(num_layers: int, delta_beta: float, delta_gamma: float):
    """
    Transform deltas into betas and gammas.
    """

    gammas = np.linspace(1 / (2 * num_layers), 1 - 1 / (2 * num_layers), num_layers)
    gammas = delta_gamma * gammas

    betas = np.linspace(1 - 1 / (2 * num_layers), 1 / (2 * num_layers), num_layers)
    betas = delta_beta * betas

    return betas, gammas
