from abc import ABC, abstractmethod
from numbers import Integral, Real
from typing import List


class Gate(ABC):
    '''A common base class for all gate objects.'''
    def __init__(self, qubits: List[int]):
        '''A gate has at minimum a set of qubits it acts on.'''
        for i, q in enumerate(qubits):
            if not (isinstance(q, Integral) and q > 0):
                raise RuntimeError(
                    f"Qubit indices must be positive integers, got {q} for"
                    f"entry {i}."
                )
        self._qubits = qubits

    @abstractmethod
    def to_tensor(self):
        '''A gate can be converted to a tensor, either parametrized or not.'''
        pass

    @abstractmethod
    def label(self):
        '''A gate has a label.'''
        pass


class OneQubitGate(Gate):
    '''A gate acting on exactly one qubit.'''
    def __init__(self, qubit: int):
        if not (isinstance(qubit, int)):
            raise RuntimeError(
                f"Qubit index of one qubit gate must be integer, got {qubit}."
            )
        super().__init__(list((qubit,)))


class OneQubitRotationGate(OneQubitGate):
    '''A common base class for all one qubit Pauli rotations.'''
    def __init__(self, qubit: int, theta: Real):
        super().__init__(qubit)
        self.theta = theta

    def _generator(theta):
        '''Generates'''
        pass


class RotationX(OneQubitRotationGate):
    pass
