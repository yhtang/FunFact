#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest  # noqa: F401
from unittest.mock import MagicMock as M
from funfact.util.iterable import as_namedtuple
from ._einop_compiler import EinopCompiler


_colon = slice(None)


@pytest.fixture
def intr():
    return EinopCompiler()


@pytest.mark.parametrize('test_case', [
    # ,->
    as_namedtuple(
        'NULL',
        tsrex=dict(
            lhs=M(
                live_indices=[],
                kron_indices=[],
            ),
            rhs=M(
                live_indices=[],
                kron_indices=[],
            ),
            precedence=None,
            reduction='reduction',
            pairwise='pairwise',
            outidx=None,
            live_indices=[],
            kron_indices=[]
        ),
        truth=as_namedtuple(
            'einspec',
            op_reduce='reduction',
            op_elementwise='pairwise',
            tran_lhs=(),
            tran_rhs=(),
            index_lhs=(),
            index_rhs=(),
            ax_contraction=(),
        )
    ),
    # i,j
    as_namedtuple(
        'OUTER',
        tsrex=dict(
            lhs=M(
                live_indices=['i'],
                kron_indices=[],
            ),
            rhs=M(
                live_indices=['j'],
                kron_indices=[],
            ),
            precedence=None,
            reduction='reduction',
            pairwise='pairwise',
            outidx=None,
            live_indices=['i', 'j'],
            kron_indices=[]
        ),
        truth=as_namedtuple(
            'einspec',
            op_reduce='reduction',
            op_elementwise='pairwise',
            tran_lhs=(),
            tran_rhs=(),
            index_lhs=(_colon, None),
            index_rhs=(None, _colon),
            ax_contraction=(),
        )
    ),
    # *i,*i
    as_namedtuple(
        'KRON1',
        tsrex=dict(
            lhs=M(
                live_indices=['i'],
                kron_indices=['i'],
            ),
            rhs=M(
                live_indices=['i'],
                kron_indices=['i'],
            ),
            precedence=None,
            reduction='reduction',
            pairwise='pairwise',
            outidx=None,
            live_indices=['i'],
            kron_indices=[]
        ),
        truth=as_namedtuple(
            'einspec',
            op_reduce='reduction',
            op_elementwise='pairwise',
            tran_lhs=(),
            tran_rhs=(),
            index_lhs=(_colon, None),
            index_rhs=(None, _colon),
            ax_contraction=(),
        )
    ),
    # *i,*i
    as_namedtuple(
        'KRON2',
        tsrex=dict(
            lhs=M(
                live_indices=['i', 'j'],
                kron_indices=['i', 'j'],
            ),
            rhs=M(
                live_indices=['i', 'j'],
                kron_indices=['i', 'j'],
            ),
            precedence=None,
            reduction='reduction',
            pairwise='pairwise',
            outidx=None,
            live_indices=['i', 'j'],
            kron_indices=[]
        ),
        truth=as_namedtuple(
            'einspec',
            op_reduce='reduction',
            op_elementwise='pairwise',
            tran_lhs=(),
            tran_rhs=(),
            index_lhs=(_colon, None, _colon, None),
            index_rhs=(None, _colon, None, _colon),
            ax_contraction=(),
        )
    ),
    # ij,ij
    as_namedtuple(
        'SUM',
        tsrex=dict(
            lhs=M(
                live_indices=['i', 'j'],
                kron_indices=[],
            ),
            rhs=M(
                live_indices=['i', 'j'],
                kron_indices=[],
            ),
            precedence=None,
            reduction='reduction',
            pairwise='pairwise',
            outidx=None,
            live_indices=[],
            kron_indices=[]
        ),
        truth=as_namedtuple(
            'einspec',
            op_reduce='reduction',
            op_elementwise='pairwise',
            tran_lhs=(),
            tran_rhs=(),
            index_lhs=(_colon, _colon),
            index_rhs=(_colon, _colon),
            ax_contraction=(0, 1),
        )
    ),
    # ij,j
    as_namedtuple(
        'MATVEC',
        tsrex=dict(
            lhs=M(
                live_indices=['i', 'j'],
                kron_indices=[],
            ),
            rhs=M(
                live_indices=['j'],
                kron_indices=[],
            ),
            precedence=None,
            reduction='reduction',
            pairwise='pairwise',
            outidx=None,
            live_indices=['i'],
            kron_indices=[]
        ),
        truth=as_namedtuple(
            'einspec',
            op_reduce='reduction',
            op_elementwise='pairwise',
            tran_lhs=(),
            tran_rhs=(),
            index_lhs=(_colon, _colon),
            index_rhs=(None, _colon),
            ax_contraction=(1,),
        )
    ),
    # i,ij
    as_namedtuple(
        'VECMAT',
        tsrex=dict(
            lhs=M(
                live_indices=['i'],
                kron_indices=[],
            ),
            rhs=M(
                live_indices=['i', 'j'],
                kron_indices=[],
            ),
            precedence=None,
            reduction='reduction',
            pairwise='pairwise',
            outidx=None,
            live_indices=['j'],
            kron_indices=[]
        ),
        truth=as_namedtuple(
            'einspec',
            op_reduce='reduction',
            op_elementwise='pairwise',
            tran_lhs=(),
            tran_rhs=(1, 0),
            index_lhs=(None, _colon),
            index_rhs=(_colon, _colon),
            ax_contraction=(1,),
        )
    ),
    # ij,jk
    as_namedtuple(
        'MATMAT',
        tsrex=dict(
            lhs=M(
                live_indices=['i', 'j'],
                kron_indices=[],
            ),
            rhs=M(
                live_indices=['j', 'k'],
                kron_indices=[],
            ),
            precedence=None,
            reduction='reduction',
            pairwise='pairwise',
            outidx=None,
            live_indices=['i', 'k'],
            kron_indices=[]
        ),
        truth=as_namedtuple(
            'einspec',
            op_reduce='reduction',
            op_elementwise='pairwise',
            tran_lhs=(),
            tran_rhs=(1, 0),
            index_lhs=(_colon, None, _colon),
            index_rhs=(None, _colon, _colon),
            ax_contraction=(2,),
        )
    ),
    # ij,jk->ijk
    as_namedtuple(
        'MATELEMMAT',
        tsrex=dict(
            lhs=M(
                live_indices=['i', 'j'],
                kron_indices=[],
            ),
            rhs=M(
                live_indices=['j', 'k'],
                kron_indices=[],
            ),
            precedence=None,
            reduction='reduction',
            pairwise='pairwise',
            outidx=None,
            live_indices=['i', 'j', 'k'],
            kron_indices=[]
        ),
        truth=as_namedtuple(
            'einspec',
            op_reduce='reduction',
            op_elementwise='pairwise',
            tran_lhs=(),
            tran_rhs=(),
            index_lhs=(_colon, _colon, None),
            index_rhs=(None, _colon, _colon),
            ax_contraction=(),
        )
    ),
    # ij,jk->ki
    as_namedtuple(
        'MATMAT_TRANSPOSE',
        tsrex=dict(
            lhs=M(
                live_indices=['i', 'j'],
                kron_indices=[],
            ),
            rhs=M(
                live_indices=['j', 'k'],
                kron_indices=[],
            ),
            precedence=None,
            reduction='reduction',
            pairwise='pairwise',
            outidx=None,
            live_indices=['k', 'i'],
            kron_indices=[]
        ),
        truth=as_namedtuple(
            'einspec',
            op_reduce='reduction',
            op_elementwise='pairwise',
            tran_lhs=(),
            tran_rhs=(1, 0),
            index_lhs=(None, _colon, _colon),
            index_rhs=(_colon, None, _colon),
            ax_contraction=(2,),
        )
    ),
    # ijk,k
    as_namedtuple(
        'TENVEC',
        tsrex=dict(
            lhs=M(
                live_indices=['i', 'j', 'k'],
                kron_indices=[],
            ),
            rhs=M(
                live_indices=['k'],
                kron_indices=[],
            ),
            precedence=None,
            reduction='reduction',
            pairwise='pairwise',
            outidx=None,
            live_indices=['i', 'j'],
            kron_indices=[]
        ),
        truth=as_namedtuple(
            'einspec',
            op_reduce='reduction',
            op_elementwise='pairwise',
            tran_lhs=(),
            tran_rhs=(),
            index_lhs=(_colon, _colon, _colon),
            index_rhs=(None, None, _colon),
            ax_contraction=(2,),
        )
    ),
    # ijk,jk
    as_namedtuple(
        'TENMAT',
        tsrex=dict(
            lhs=M(
                live_indices=['i', 'j', 'k'],
                kron_indices=[],
            ),
            rhs=M(
                live_indices=['j', 'k'],
                kron_indices=[],
            ),
            precedence=None,
            reduction='reduction',
            pairwise='pairwise',
            outidx=None,
            live_indices=['i'],
            kron_indices=[]
        ),
        truth=as_namedtuple(
            'einspec',
            op_reduce='reduction',
            op_elementwise='pairwise',
            tran_lhs=(),
            tran_rhs=(),
            index_lhs=(_colon, _colon, _colon),
            index_rhs=(None, _colon, _colon),
            ax_contraction=(1, 2),
        )
    ),
    # ijk,jl
    as_namedtuple(
        'MODEWISE',
        tsrex=dict(
            lhs=M(
                live_indices=['i', 'j', 'k'],
                kron_indices=[],
            ),
            rhs=M(
                live_indices=['j', 'l'],
                kron_indices=[],
            ),
            precedence=None,
            reduction='reduction',
            pairwise='pairwise',
            outidx=None,
            live_indices=['i', 'l', 'k'],
            kron_indices=[]
        ),
        truth=as_namedtuple(
            'einspec',
            op_reduce='reduction',
            op_elementwise='pairwise',
            tran_lhs=(0, 2, 1),
            tran_rhs=(1, 0),
            index_lhs=(_colon, None, _colon, _colon),
            index_rhs=(None, _colon, None, _colon),
            ax_contraction=(3,),
        )
    ),
    # ijk,il
    as_namedtuple(
        'MODEWISE',
        tsrex=dict(
            lhs=M(
                live_indices=['i', 'j', 'k'],
                kron_indices=[],
            ),
            rhs=M(
                live_indices=['i', 'l'],
                kron_indices=[],
            ),
            precedence=None,
            reduction='reduction',
            pairwise='pairwise',
            outidx=None,
            live_indices=['l', 'j', 'k'],
            kron_indices=[]
        ),
        truth=as_namedtuple(
            'einspec',
            op_reduce='reduction',
            op_elementwise='pairwise',
            tran_lhs=(1, 2, 0),
            tran_rhs=(1, 0),
            index_lhs=(None, _colon, _colon, _colon),
            index_rhs=(_colon, None, None, _colon),
            ax_contraction=(3,),
        )
    ),
    # ijk,jkl
    as_namedtuple(
        'TENTEN',
        tsrex=dict(
            lhs=M(
                live_indices=['i', 'j', 'k'],
                kron_indices=[],
            ),
            rhs=M(
                live_indices=['j', 'k', 'l'],
                kron_indices=[],
            ),
            precedence=None,
            reduction='reduction',
            pairwise='pairwise',
            outidx=None,
            live_indices=['i', 'l'],
            kron_indices=[]
        ),
        truth=as_namedtuple(
            'einspec',
            op_reduce='reduction',
            op_elementwise='pairwise',
            tran_lhs=(),
            tran_rhs=(2, 0, 1),
            index_lhs=(_colon, None, _colon, _colon),
            index_rhs=(None, _colon, _colon, _colon),
            ax_contraction=(2, 3),
        )
    ),
])
def test_einop_compiler(test_case, intr):
    input, truth = test_case
    _, spec = intr.ein(**input)
    assert spec == truth
