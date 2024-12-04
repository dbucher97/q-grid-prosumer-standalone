from __future__ import annotations
import dimod
from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator
from uuid import UUID, uuid4

from qaoa_pipeline.circuit.qiskit_circuit import QiskitCircuit


class SingleCircuit(BaseModel):
    problem: str
    circuit: QiskitCircuit
    # multiple parameter excecutions for single circuit possible
    parameters: list[list[float]]

    slug: UUID = Field(default_factory=uuid4)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def get_circuit(circuit, cqm: dimod.ConstrainedQuadraticModel) -> QiskitCircuit:
        return QiskitCircuit.load(basemodel=cqm)

    @field_validator("circuit", mode="before")
    def validate_circut(cls, value, context):
        if isinstance(value, QiskitCircuit):
            return value
        cqms = context.context
        return QiskitCircuit.load(value, cqms[context.data["problem"]][0])

    @field_serializer("circuit")
    def serialize_circuit(self, circuit):
        return circuit.store()


class CircuitStorage(BaseModel):
    cqms: dict[str, tuple[dimod.ConstrainedQuadraticModel, float]] = Field(
        default_factory=lambda: {}
    )
    circuits: dict[UUID, SingleCircuit] = Field(default_factory=lambda: {})

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def add_circuit(self, circ: SingleCircuit):
        self.circuits[circ.slug] = circ

    @field_validator("cqms", mode="before")
    @classmethod
    def validate_cqms(cls, cqms: dict[str, tuple[str, float]]):
        return {name: (dimod.lp.loads(lp), opt) for name, (lp, opt) in cqms.items()}

    @field_serializer("cqms")
    def serialize_cqms(self, cqms):
        return {name: (dimod.lp.dumps(lp), opt) for name, (lp, opt) in cqms.items()}

    @field_validator("circuits", mode="before")
    @classmethod
    def validate_circuits(cls, value, context):
        return {
            k: SingleCircuit.model_validate(v, context=context.data["cqms"])
            for k, v in value.items()
        }
