from dataclasses import dataclass, field
import functools
import numpy as np

import logging

logger = logging.getLogger(__name__)


def _mult(arr: list[int]):
    if len(arr) == 0:
        return 1
    else:
        return functools.reduce(lambda x, y: x * y, arr)


@dataclass
class QubitLayout:
    """Qubit Layout container. Convention: Qubits first, then qudits"""

    qubit_order: list[str] = field(default_factory=lambda: [])
    qudit_order: list[tuple[str, ...]] = field(default_factory=lambda: [])

    @property
    def qubits(self) -> int:
        return len(self.qubit_order)

    @property
    def qudits(self) -> list[int]:
        return list(map(len, self.qudit_order))

    @property
    def shape(self):
        """Shape with qubits in one dimension"""
        return tuple([(1 << self.qubits)] + self.qudits)

    @property
    def shape2(self):
        """Shape with qubits in separate dimensions"""
        return tuple([2] * self.qubits + list(self.qudits))

    @property
    def size(self):
        """Total size of Statevector (number of dimensions)"""
        return (1 << self.qubits) * _mult(self.qudits)

    @property
    def total(self):
        """Total number of qubits in layout"""
        return self.qubits + sum(self.qudits)

    @property
    def pos_starts(self):
        """Bit start of variables in key"""
        vals = [0]
        for s in self.qudits:
            if len(vals) == 1:
                vals.append(self.qubits)
            else:
                vals.append(vals[-1] + s)
        return vals

    @property
    def variables(self):
        return self.qubit_order + [
            v for vars in self.qudit_order for v in reversed(vars)
        ]

    def feasible_subspace_selection(self):
        if len(self.qudits) == 0:
            return np.arange(1 << self.qubits)

        starts = np.array(self.pos_starts)

        total = sum(self.qudits)
        starts = total - np.cumsum(np.array(self.qudits))
        shape = self.shape[1:]

        normal_selection = np.arange(1 << self.qubits) << total
        ohc_selection = np.sum(
            1 << np.indices(shape).reshape((len(shape), -1)) << starts[:, None],
            axis=0,
        )
        logger.debug(
            f"Feasible subspace selection (qubit_space={len(normal_selection)}, qudit_space={len(ohc_selection)})"
        )
        return np.add.outer(normal_selection, ohc_selection).flatten()

    def __post_init__(self):
        total_qudit_bits = sum(self.qudits)
        if total_qudit_bits > 0:
            non_overlapping_bits = len(set.union(*map(set, self.qudit_order)))
            assert (
                total_qudit_bits == non_overlapping_bits
            ), "Overlapping One-Hot Constraints detected"

    def subset_in_order(self, subset: set):
        total = self.total
        idx_map = {v: i for i, v in enumerate(self.variables)}
        ret = list(sorted(subset, key=lambda x: idx_map.get(x, total)))
        return ret
