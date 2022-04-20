#!/usr/bin/env python
# -*- coding: utf-8 -*-
from numbers import Integral
from funfact.lang._special import eye


class Circuit():
    '''A class to create quantum circuits and tensor expressions from them.'''
    def __init__(self, nbqubits, offset=0):
        if not (isinstance(nbqubits, Integral) and nbqubits > 0):
            raise RuntimeError(
                f'Number of qubits must be integer > 0, got {nbqubits}.'
            )
        if not (isinstance(offset, Integral) and offset >= 0):
            raise RuntimeError(
                f'Offset must be integer >= 0, got {offset}.'
            )

        self._nbqubits = nbqubits
        self._offset = offset
        self._qgates = []

    @property
    def nbqubits(self):
        return self._nbqubits

    @property
    def qubits(self):
        return [0, self._nbqubits-1] + self._offset

    def append(self, qobject):
        '''Add a quantum object at the end of this circuit.'''
        self._qgates.append(qobject)

    def to_tensor(self):
        tsrex = eye(2**self._nbqubits)
        for i, g in enumerate(self._qgates):
            if g.qubits[-1] >= self.nbqubits:
                raise RuntimeError(
                    f"Gate acts on qubit {g.qubits[-1]}, which is out of "
                    f"bounds for circuit with {self.nbqubits} qubits."
                )
            # gsize = g.qubits[-1] - g.qubits[0] + 1
            tsrex = (
                eye(2**g.qubits[0]) &
                g.to_tensor() &
                eye(2**(self.nbqubits - g.qubits[1] - 1))
            ) @ tsrex
        return tsrex
