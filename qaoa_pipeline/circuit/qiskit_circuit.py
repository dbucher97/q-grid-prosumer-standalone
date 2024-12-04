from __future__ import annotations

from dataclasses import asdict
import logging
import numpy as np
import numpy.lib.recfunctions as rfn

from qiskit import ClassicalRegister, QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.providers.backend import BackendV2
from qiskit.passmanager import BasePassManager
from qiskit.quantum_info import Statevector
from qiskit_ibm_runtime import SamplerV2
from qiskit import qasm3
import dimod

from qaoa_pipeline.circuit.qaoa_circuit import QAOACircuit
from qaoa_pipeline.utils.layout import QubitLayout
from qaoa_pipeline.circuit.qaoa_parameters import QaoaParameters

logger = logging.getLogger(__name__)


class QiskitCircuit(QAOACircuit):
    def __init__(
        self,
        layout: QubitLayout,
        init_circuit: QuantumCircuit,
        layer_circuit: QuantumCircuit,
        objective: dimod.BinaryQuadraticModel,
        backend: BackendV2 | None = None,
        basemodel: dimod.ConstrainedQuadraticModel | None = None,
        penalties: list[float] | None = None,
        constr_idxs: list[int] | None = None,
    ):
        super().__init__(layout)
        self.layout = layout
        self.init_circuit = init_circuit
        self.layer_circuit = layer_circuit

        self._objective = objective
        self._basemodel = basemodel

        self._sampler = None
        if backend is not None:
            self._sampler = SamplerV2(mode=backend)

        self._penalties = penalties
        self._constr_idxs = constr_idxs

        self._circuit_store = {}

    @property
    def basemodel(self):
        return self.basemodel

    @basemodel.setter
    def basemodel(self, basemodel: dimod.ConstrainedQuadraticModel):
        self.basemodel = basemodel

    def _build_circuit(self, qaoa_parameters: QaoaParameters):
        qc = QuantumCircuit(*self.layer_circuit.qregs)
        qc &= self.init_circuit
        betas = [Parameter(f"beta_{i}") for i in range(qaoa_parameters.num_layers)]
        gammas = [Parameter(f"gamma_{i}") for i in range(qaoa_parameters.num_layers)]
        for b, g in zip(betas, gammas):
            qc &= self.layer_circuit.assign_parameters({"beta": b, "gamma": g})

        if self._sampler is not None:
            creg = ClassicalRegister(len(self.layer_circuit.qregs[0]), name="res")
            qc.add_register(creg)
            qc.measure(qc.qregs[0], creg)
        return qc

    def get_circuit(self, qaoa_parameters: QaoaParameters):
        if qaoa_parameters.num_layers not in self._circuit_store:
            self._circuit_store[qaoa_parameters.num_layers] = self._build_circuit(
                qaoa_parameters
            )
        mapping = {f"beta_{i}": v for i, v in enumerate(qaoa_parameters.betas)}
        mapping.update({f"gamma_{i}": v for i, v in enumerate(qaoa_parameters.gammas)})
        return self._circuit_store[qaoa_parameters.num_layers].assign_parameters(
            mapping
        )

    def sampleset_from_counts(self, counts: dict):
        num_occ = list(counts.values())
        samples = [
            {v: int(i) for v, i in zip(self.layout.variables, reversed(k))}
            for k in counts.keys()
        ]

        bqm_sampleset = dimod.SampleSet.from_samples_bqm(samples, self._objective)

        if self._basemodel is None:
            raise RuntimeError("Base CQM needs to be set for circuit.")

        sampleset = dimod.SampleSet.from_samples_cqm(
            samples, self._basemodel, num_occurrences=num_occ
        )

        sampleset._record = rfn.append_fields(
            sampleset.record,
            "cqm_energy",
            sampleset.record.energy,
            usemask=False,
            asrecarray=True,
        )

        sampleset.record.energy = bqm_sampleset.record.energy

        if self._penalties is not None:
            satisfied = sampleset.record.is_satisfied
            penalties = np.logical_not(satisfied[:, self._constr_idxs]).dot(
                self._penalties
            )

            sampleset.record.energy += penalties

        return sampleset

    def _get_sampleset(self, qaoa_parameters: QaoaParameters):
        qc = self.get_circuit(qaoa_parameters)

        if self._sampler is None:
            sv = Statevector(qc)
            counts = sv.probabilities_dict(list(range(len(qc.qregs[0]))), decimals=14)
        else:
            result = self._sampler.run([qc]).result()
            counts = result[0].data.res.get_counts()
        return self.sampleset_from_counts(counts)

    def expectation_value(self, qaoa_parameters: QaoaParameters):
        sampleset = self._get_sampleset(qaoa_parameters)
        return np.average(
            sampleset.record.energy, weights=sampleset.record.num_occurrences
        )

    def sample(self, qaoa_parameters: QaoaParameters):
        return self._get_sampleset(qaoa_parameters)

    def draw(self, **_):
        return self.get_circuit(QaoaParameters.from_deltas(1, 2, 2)).draw()

    def get_circuit_depth(self, num_layers: int) -> float:
        return num_layers * self.layer_circuit.depth() + self.init_circuit.depth()

    def get_probs_with_sample(
        self, _: QaoaParameters, __: float
    ) -> list[tuple[dict, float]]:
        raise NotImplementedError(
            "The method `get_probs_with_sample` has not been implemented"
        )

    def get_energies(self, _: list[int]):
        raise NotImplementedError(
            "The method `get_probs_with_sample` has not been implemented"
        )

    def transpile(
        self, passmanager: BasePassManager, backend: BackendV2 | None
    ) -> QiskitCircuit:
        init_circuit = passmanager.run(self.init_circuit)
        layer_circuit = passmanager.run(self.layer_circuit)
        return QiskitCircuit(
            layout=self.layout,
            init_circuit=init_circuit,
            layer_circuit=layer_circuit,
            objective=self._objective,
            basemodel=self._basemodel,
            backend=backend,
            penalties=self._penalties,
            constr_idxs=self._constr_idxs,
        )

    def store(self) -> dict:
        return {
            "layout": asdict(self.layout),
            "init_circuit": qasm3.dumps(self.init_circuit),
            "layer_circuit": qasm3.dumps(self.layer_circuit),
            "objective": self._objective.to_serializable(),
            "penalties": list(self._penalties),
            "constr_idxs": list(self._constr_idxs),
        }

    @classmethod
    def load(
        self,
        input: dict,
        basemodel: dimod.ConstrainedQuadraticModel | None = None,
        backend: BackendV2 | None = None,
    ) -> QiskitCircuit:
        init_circuit = qasm3.loads(input["init_circuit"])
        layer_circuit = qasm3.loads(input["layer_circuit"])
        layout = QubitLayout(**input["layout"])
        penalties = input["penalties"]
        constr_idxs = input["constr_idxs"]
        objective = dimod.BinaryQuadraticModel.from_serializable(input["objective"])
        return QiskitCircuit(
            layout=layout,
            init_circuit=init_circuit,
            layer_circuit=layer_circuit,
            objective=objective,
            penalties=penalties,
            constr_idxs=constr_idxs,
            basemodel=basemodel,
            backend=backend,
        )
