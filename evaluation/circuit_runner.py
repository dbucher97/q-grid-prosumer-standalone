from copy import deepcopy
import pickle
import time
from uuid import UUID
from tqdm.auto import tqdm
import numpy as np
from sqlitedict import SqliteDict

from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2, Session
from qiskit import generate_preset_pass_manager
from evaluation.circuit_storage import CircuitStorage
from evaluation.metrics import EvaluationMetrics
from qaoa_pipeline.circuit.qaoa_parameters import QaoaParameters


class CircuitRunner:
    """Runs the circuits defined in a circuit storage object. Results will be stored
    relative to the circuit storage file, i.e. running `foo/storage.pkl` will produce
    `foo/storage.pkl-results.sqlite"""

    circuit_storage: CircuitStorage

    def __init__(
        self,
        circuit_storage: str,
        service: QiskitRuntimeService,
        backend: str,
        transpiler_options: dict | None = None,
        verbose: bool = False,
    ):
        """
        Parameters
        ----------
        circuit_storage : str
            filepath of the pickeled object
        service : QiskitRuntimeService
            The defined service for QiskitRuntime
        backend : str
            IBM backend to use
        transpiler_options : dict
            The dict defining transpiler options for the backend.
            Default: `{"optimization_level": 0}`
        """
        with open(circuit_storage, "rb") as f:
            self.circuit_storage = CircuitStorage.model_validate(pickle.load(f))

        self.transpiler_options = transpiler_options or {"optimization_level": 0}

        self.service = service
        self.backend = service.backend(backend)

        self.result_path = circuit_storage + "-result.sqlite"

        self.result = SqliteDict(self.result_path, autocommit=True)

        self.verbose = verbose

    def get_filtered_circuit_storage(self) -> CircuitStorage:
        """
        Returns filtered circuit storage with already present (or started) jobs removed

        Returns
        -------
        circuit_storage: CircuitStorage
        """
        circuit_storage = deepcopy(self.circuit_storage)

        circuit_storage.circuits = {
            k: v
            for k, v in circuit_storage.circuits.items()
            if k.hex not in self.result
        }

        return circuit_storage

    def run(self, session: Session | None = None):
        """Transpile and start execution of the jobs

        Parameters
        ----------
        session: Session | None
            If provided, run the jobs within a session
        """
        circuit_storage = self.get_filtered_circuit_storage()

        if len(circuit_storage.circuits) == 0:
            return

        pm = generate_preset_pass_manager(
            backend=self.backend, **self.transpiler_options
        )

        for c in tqdm(circuit_storage.circuits.values(), desc="Transpile"):
            c.circuit = c.circuit.transpile(pm, backend=self.backend)

        pubs = {
            k: [
                c.circuit.get_circuit(QaoaParameters.from_defined_params(np.array(p)))
                for p in c.parameters
            ]
            for k, c in circuit_storage.circuits.items()
        }

        if session is None:
            session = self.backend

        sampler = SamplerV2(mode=session)

        for k, pub in tqdm(pubs.items(), desc="Starting"):
            job = sampler.run(pub)

            self.result[k.hex] = job.job_id()

    def retrieve(self, poll: float = 1):
        """Retrieve started jobs

        Parameters
        ----------
        poll: float
            Polling interval for retrieving non-finished results
        """
        open_ids = {UUID(k): v for k, v in self.result.items() if isinstance(v, str)}

        bar = tqdm(desc="Retrieving", total=len(self.result))
        bar.update(len(self.result) - len(open_ids))

        general_metadata = {
            "backend": self.backend.name,
            "transpiler_options": self.transpiler_options,
        }

        while len(open_ids) > 0:
            for k, v in list(open_ids.items()):
                single_run = self.circuit_storage.circuits[k]

                if not isinstance(v, str):
                    continue
                retrieved_job = self.service.job(v)

                if not retrieved_job.in_final_state():
                    continue

                del open_ids[k]
                bar.update()

                if retrieved_job.errored():
                    self.result[k.hex] = {
                        "status": "ERROR",
                        "job_id": v,
                        "error_message": retrieved_job.error_message(),
                        **general_metadata,
                    }
                    if self.verbose:
                        print(f"Job '{single_run.problem}' failed.")
                    continue

                result = retrieved_job.result()
                result_list = []

                metrics = EvaluationMetrics(
                    *self.circuit_storage.cqms[single_run.problem]
                )
                if self.verbose:
                    print(f"Job '{single_run.problem}' results:")

                for data, params in zip(result, single_run.parameters):
                    counts = data.data.res.get_counts()
                    sampleset = single_run.circuit.sampleset_from_counts(counts)
                    params = QaoaParameters.from_defined_params(np.array(params))
                    result_dict = metrics.evaluate_circuit_performance_qiskit(
                        sampleset, params, single_run.circuit
                    )
                    result_dict["full_samples"] = sampleset.to_serializable(
                        use_bytes=True
                    )
                    result_list.append(result_dict)

                    if self.verbose:
                        print(
                            f"\tdepth={params.num_layers} "
                            f"p_opt={result_dict['p_opt']:.2e} "
                            f"tts={result_dict['tts']:.2e} "
                            f"rel_err={result_dict['rel_energy_error']:.2f}"
                        )

                self.result[k.hex] = {
                    "status": "SUCCESS",
                    "job_id": v,
                    "metadata": result.metadata,
                    **general_metadata,
                    "results": result_list,
                }

            time.sleep(poll)
