from ..backend import backend_manager as bm
from typing import Optional

from ..mesh import HomogeneousMesh
from ..functionspace.space import FunctionSpace as _FS
from ..utils import process_coef_func
from ..utils import is_scalar, is_tensor, fill_axis

from ..functional import bilinear_integral
from .integrator import LinearInt, OpInt, CellInt, enable_cache
from ..typing import TensorLike, Index, _S, CoefLike
from ..functionspace.functional import symmetry_span_array, symmetry_index

from ..mesh import TriangleMesh

from sympy import symbols, sin, cos, Matrix, lambdify

class HuZhangStressIntegrator(LinearInt, OpInt, CellInt):
    def __init__(self, coef: Optional[CoefLike]=None, q = None, lambda0 = 1.0, lambda1 = 1.0, *,
                 index: Index=_S,
                 batched: bool=False,
                 method: Optional[str]=None):
        super().__init__()
        self.coef = coef
        self.q = q
        self.lambda0 = lambda0
        self.lambda1 = lambda1
        self.index = index 
        self.batched = batched

    @enable_cache
    def to_global_dof(self, space: _FS) -> TensorLike:
        c2d0  = space.cell_to_dof()
        return c2d0

    @enable_cache
    def fetch(self, space: _FS):
        p = space.p
        q = self.q if self.q else p+3

        index = self.index

        mesh = space.mesh
        TD = mesh.top_dimension()
        cm = mesh.entity_measure('cell', index=index)
        qf = mesh.quadrature_formula(q, 'cell')

        bcs, ws = qf.get_quadrature_points_and_weights()
        phi = space.basis(bcs, index=index)
        if TD == 2:
            trphi = phi[..., 0] + phi[..., -1]
        if TD == 3:
            trphi = phi[..., 0] + phi[..., 3] + phi[..., -1]
        return cm, phi, trphi, ws, bcs, index

    def assembly(self, space: _FS) -> TensorLike:
        mesh = space.mesh 
        TD = mesh.top_dimension()
        batched = self.batched
        coef = self.coef
        lambda0, lambda1 = self.lambda0, self.lambda1 
        cm, phi, trphi, ws, bcs, index = self.fetch(space) 

        _, num = symmetry_index(d=TD, r=2)
        val = process_coef_func(coef, bcs=bcs, mesh=mesh, etype='cell', index=index)
        print('val.shape = ', val.shape, cm.shape, bm.size(val))
        if val is None:
            A  = lambda0*bm.einsum('q, c, cqld, cqmd, d->clm', ws, cm, phi, phi, num)
            A -= lambda1*bm.einsum('q, c, cql, cqm->clm', ws, cm, trphi, trphi)
        if is_scalar(val):
            A  = lambda0*bm.einsum('q, c, cqld, cqmd, d->clm', ws, cm, phi, phi, num) * val
            A -= lambda1*bm.einsum('q, c, cql, cqm->clm', ws, cm, trphi, trphi) * val
        elif is_tensor(val):
            #print('val.shape = ', val.shape, cm.shape)
            #ndim = val.ndim - int(batched)
            #if ndim == 4:
            
            #    raise TypeError(f"coef should be int, float or TensorLike, but got {type(coef)}.")
            #else:
                #val = fill_axis(coef, 4 if batched else 3)
            A  = lambda0*bm.einsum('q, c, cqld, cqmd, d, cq->clm', ws, cm, phi, phi, num, val)
            A -= lambda1*bm.einsum('q, c, cql, cqm, cq->clm', ws, cm, trphi, trphi, val)
        
        else:
            raise TypeError(f"coef should be int, float or TensorLike, but got {type(coef)}.")
        #print('val.shape = ', val.shape, cm.shape)
        #A  = lambda0*bm.einsum('q, c, cqld, cqmd, d, cq->clm', ws, cm, phi, phi, num, val)
        #A -= lambda1*bm.einsum('q, c, cql, cqm, cq->clm', ws, cm, trphi, trphi, val)
        return A




