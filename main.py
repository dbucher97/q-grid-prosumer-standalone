import json
from qiskit_ibm_runtime import QiskitRuntimeService
from evaluation.circuit_runner import CircuitRunner

with open("../secrets.json") as f:
    secrets = json.load(f)

crn = secrets["crn"]
apikey = secrets["apikey"]

service = QiskitRuntimeService(channel="ibm_cloud", token=apikey, instance=crn)


file_name = "<REPLACE_THIS>.pkl"


circ_runner = CircuitRunner(
    circuit_storage=file_name,
    service=service,
    backend="simulator_statevector",
    verbose=True,
)

circ_runner.run()

circ_runner.retrieve()
