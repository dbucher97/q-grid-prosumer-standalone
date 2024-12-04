from copy import deepcopy
from dimod import ConstrainedQuadraticModel
import dimod

from qaoa_pipeline.circuit.qaoa_generator import QAOACircuit
from qaoa_pipeline.circuit.qiskit_generator import QiskitCircuit
from qaoa_pipeline.circuit.simulator_generator import Simulator
import numpy as np
from qaoa_pipeline.ir.passes.presolve import (
    ClassicalPresolveContext,
    ClassicalPresolvePass,
)
from qaoa_pipeline.circuit.qaoa_parameters import unpack_params, QaoaParameters
from qaoa_pipeline.simulator.diagonalize import diagonalize_ir


class EvaluationMetrics:
    cqm: ConstrainedQuadraticModel
    optimal_energy: float

    def __init__(
        self,
        base_cqm: ConstrainedQuadraticModel,
        optimal_energy: float,
        prob_rtol: float = 0.0,
    ):
        """
        Initializes the ArchiveRecord instance with the given ConstrainedQuadraticModel and optimal energy.

        Parameters:
        base_cqm (ConstrainedQuadraticModel): The base constrained quadratic model.
        optimal_energy (float): The optimal energy value.
        """
        self.base_cqm = base_cqm
        self.optimal_energy = optimal_energy
        self.prob_rtol = prob_rtol

        self._feasibility_cache = []

    def evaluate_circuit_performance_qiskit(
        self,
        sampleset: dimod.SampleSet,
        optimized_params: QaoaParameters,
        circuit: QiskitCircuit,
    ):
        expected_energy = np.average(
            sampleset.record.energy, weights=sampleset.record.num_occurrences
        )
        rel_energy_error = self.get_relative_energy_error(
            expected_energy=expected_energy, optimal_energy=self.optimal_energy
        )

        feasibility_ratio = np.average(
            sampleset.record.is_feasible, weights=sampleset.record.num_occurrences
        )

        tot = sampleset.record.num_occurrences.sum()
        where_opt = np.where(
            sampleset.record.energy <= self.optimal_energy * (1 + self.prob_rtol)
        )[0]
        p_opt = sampleset.record.num_occurrences[where_opt].sum() / tot

        tts = np.inf
        if p_opt > 1e-16:
            depth = circuit.get_circuit_depth(optimized_params.num_layers)
            if p_opt == 1:
                return depth
            tts = depth * np.ceil(np.log(0.01) / np.log(1 - p_opt))

        return {
            "rel_energy_error": rel_energy_error,
            "tts": tts,
            "p_opt": p_opt,
            "feasibility_ratio": feasibility_ratio,
            "opt_params": optimized_params.params,
        }

    def evaluate_circuit_performance(
        self, optimized_params: QaoaParameters, circuit: QAOACircuit
    ):
        """
        Evaluates the performance of a QAOA (Quantum Approximate Optimization Algorithm) circuit based on the results obtained from an optimizer.

        Parameters:
         results (OptimizerResult): The results obtained from optimizing the QAOA parameters.
         circuit (QAOACircuit): The QAOA circuit to be evaluated.

        Returns:
         dict: A dictionary containing the evaluation metrics:
            - "relative_energy_error": The relative error between the expected energy and the optimal energy.
            - "time_to_solution": The time required to reach a solution.
            - "feasibility_ratio": The ratio indicating how feasible the solution is, given the constraints and accuracy.
        """

        if isinstance(circuit, QiskitCircuit):
            sampleset = circuit.sample(optimized_params)
            return self.evaluate_circuit_performance_qiskit(
                sampleset=sampleset, optimized_params=optimized_params, circuit=circuit
            )

        # determine probs per idxs
        energies = circuit.get_energies()

        # determine energy per idxs
        probs = circuit.sample(qaoa_parameters=optimized_params)

        # detemrine expected energy
        expected_energy = probs.dot(energies)

        evaluation = {}
        evaluation["rel_energy_error"] = float(
            self.get_relative_energy_error(
                expected_energy=expected_energy, optimal_energy=self.optimal_energy
            )
        )
        evaluation["tts"], evaluation["p_opt"] = self.get_time_to_solution(
            circuit=circuit,
            num_layers=optimized_params.num_layers,
            energies=energies,
            probs=probs,
        )
        evaluation["tts"] = float(evaluation["tts"])
        evaluation["p_opt"] = float(evaluation["p_opt"])

        if isinstance(circuit, Simulator):
            evaluation["feasibility_ratio"] = float(
                self.get_feasibility_ratio(probs=probs, circuit=circuit)
            )
        else:
            evaluation["feasibility_ratio"] = float(
                self.get_feasibility_ratio_legacy(
                    parameters=optimized_params.params, circuit=circuit
                )
            )
        evaluation["opt_params"] = optimized_params.params

        evaluation["invalid_infeasible_states"] = float(
            (energies < self.optimal_energy - 1e-4).sum()
        )

        return evaluation

    def get_relative_energy_error(self, expected_energy: float, optimal_energy: float):
        """
        The relative energy error is given by calculating the absolut difference between the optimal solution and
        the found solution, divided by the optimal solution.
        """
        abs_diff = np.abs(expected_energy - optimal_energy)
        rel_error = abs_diff / optimal_energy
        return rel_error

    def get_time_to_solution(
        self,
        circuit: QAOACircuit,
        num_layers: int,
        energies: np.ndarray,
        probs: np.ndarray,
        tol: float = 1e-4,
    ):
        """
        Calculates the time-to-solution for a given QAOA circuit.

        Parameters:
            circuit (QAOACircuit): The quantum approximate optimization algorithm (QAOA) circuit.
            num_layers (int): The number of layers in the QAOA circuit.
            energies (np.ndarray): The array of energy values.
            probs (np.ndarray): The array of probabilities corresponding to the energy values.

        Returns:
            float: The calculated time-to-solution metric.
        """
        # get time to sample circuit
        ts = circuit.get_circuit_depth(num_layers=num_layers)

        # get idxs with optimal energy level
        opt_position = np.where(
            energies <= self.optimal_energy * (1 + self.prob_rtol) + tol
        )

        # determine summed prob to get optimal results
        prob_opt = np.sum(probs[opt_position])

        if prob_opt == 0:
            print("Optimal result was not found!")
            return np.inf, 0

        # calculate time-to-solution metric
        inner_division = np.log(0.01) / np.log(1 - prob_opt)
        tts = ts * np.ceil(inner_division)

        return tts, prob_opt

    def get_feasibility_ratio(self, probs: np.ndarray, circuit: Simulator):
        """
        Calculate the feasibility ratio fast and accurate based for only Simulator circuits.

        This method computes the ratio of feasible solution probabilities within by
        brute forcing the IR without the initial optimization objective

        Parameters:
        probs (np.ndarray): The array of QAOA probabilities.
        circuit (Simulator): The Simulator circuit to be executed with the given parameters.

        Returns:
        float: The sum of probabilities for all feasible states.
        """

        # retrieve feasibility vector for specific IR from cache
        for k, v in self._feasibility_cache:
            if k == circuit.ir:
                if v is None:
                    return 1.0
                else:
                    return v.dot(probs)

        # clone IR
        ir = deepcopy(circuit.ir)

        # get the objective of the original CQM, apply pre-solve if has been applied
        cqm = dimod.ConstrainedQuadraticModel()

        if ir.has(ClassicalPresolveContext):
            base_model2, _ = ClassicalPresolvePass().apply_pass(circuit.ir.base_model)
            obj = base_model2.objective
        else:
            obj = circuit.ir.base_model.objective

        qm = dimod.QuadraticModel() + obj

        cqm.set_objective(qm)
        bqm, _ = dimod.cqm_to_bqm(cqm)

        # subtract the objective from the IR objective -> only penalty terms should
        # persist
        ir.objective -= bqm

        # check if objective has entries
        no_linear = all(x == 0 for x in ir.objective.linear.values())
        no_quadratic = all(x == 0 for x in ir.objective.quadratic.values())

        if no_linear and no_quadratic:
            # if not, only use the feasibility from the simulator object, which should
            # be enough. This means that no penalty terms are in the final objective
            feasible = circuit.sim._feasibility
        else:
            # diagonalize the objective with only penalty terms
            dg, indicator_feas = diagonalize_ir(ir)

            # feasible if entry is zero
            feasible = np.isclose(dg, 0)
            # and if feasibility from indicator functions is given
            if indicator_feas is not None:
                feasible &= indicator_feas

        # store the feasibility for later re-use
        self._feasibility_cache.append((circuit.ir, feasible))

        if feasible is None:
            return 1

        return feasible.dot(probs)

    def get_feasibility_ratio_legacy(
        self, parameters: np.ndarray, circuit: QAOACircuit
    ):
        """
        Calculate the feasibility ratio for the given quantum approximate optimization algorithm (QAOA) parameters on a specified quantum circuit.

        This method computes the ratio of feasible solution probabilities within a threshold from a QAOA circuit and checks their feasibility using a constrained quadratic model (CQM).

        Parameters:
        parameters (np.ndarray): The array of QAOA parameters to be used in the quantum circuit.
        circuit (QAOACircuit): The QAOA circuit to be executed with the given parameters.
        accuracy (int): The number of decimal places to round the feasibility probability. Default is 4.

        Returns:
        float: The sum of probabilities for all feasible states, rounded to the specified accuracy.
        """
        betas, gammas = unpack_params(parameters)
        # get all solution states and their probs
        sample_n_probs = circuit.get_probs_with_sample(
            betas=betas, gammas=gammas, prob_threshold=1e-3
        )

        # translate into dimod sample
        cqm = self.base_cqm

        # check if feasible using dimod.ConstrainedQuadaraticProgramm.check_feasible
        prob_feasible = 0
        for sample, prob in sample_n_probs:
            feasible = cqm.check_feasible(sample)
            if feasible:
                prob_feasible += prob

        # sum prob for all feasible states
        return prob_feasible
