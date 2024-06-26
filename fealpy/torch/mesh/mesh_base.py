
from typing import (
    Union, Optional, Dict, Sequence, overload, Callable,
    Literal, TypeVar
)

import numpy as np
import torch

from .. import logger
from . import functional as F
from . import mesh_kernel as K
from .quadrature import Quadrature

Tensor = torch.Tensor
Index = Union[Tensor, int, slice]
EntityName = Literal['cell', 'cell_location', 'face', 'face_location', 'edge']
_int_func = Callable[..., int]
_dtype = torch.dtype
_device = torch.device

_S = slice(None, None, None)
_T = TypeVar('_T')
_default = object()


##################################################
### Utils
##################################################

def mesh_top_csr(entity: Tensor, num_targets: int, location: Optional[Tensor]=None, *,
                 dtype: Optional[_dtype]=None) -> Tensor:
    r"""CSR format of a mesh topology relaionship matrix."""
    device = entity.device

    if entity.ndim == 1: # for polygon case
        if location is None:
            raise ValueError('location is required for 1D entity (usually for polygon mesh).')
        crow = location
    elif entity.ndim == 2: # for homogeneous case
        crow = torch.arange(
            entity.size(0) + 1, dtype=entity.dtype, device=device
        ).mul_(entity.size(1))
    else:
        raise ValueError('dimension of entity must be 1 or 2.')

    return torch.sparse_csr_tensor(
        crow,
        entity.reshape(-1),
        torch.ones(entity.numel(), dtype=dtype, device=device),
        size=(entity.size(0), num_targets),
        dtype=dtype, device=device
    )


def entity_str2dim(ds, etype: str) -> int:
    if etype == 'cell':
        return ds.top_dimension()
    elif etype == 'cell_location':
        return -ds.top_dimension()
    elif etype == 'face':
        TD = ds.top_dimension()
        # if TD <= 1:
        #     raise ValueError('the mesh has no face entity.')
        return TD - 1
    elif etype == 'face_location':
        TD = ds.top_dimension()
        # if TD <= 1:
        #     raise ValueError('the mesh has no face location.')
        return -TD + 1
    elif etype == 'edge':
        return 1
    elif etype == 'node':
        return 0
    else:
        raise KeyError(f'{etype} is not a valid entity attribute.')


def entity_dim2tensor(ds, etype_dim: int, index=None, *, default=_default):
    r"""Get entity tensor by its top dimension."""
    if etype_dim in ds._entity_storage:
        et = ds._entity_storage[etype_dim]
        if index is None:
            return et
        else:
            if et.ndim == 1:
                raise RuntimeError("index is not supported for flattened entity.")
            return et[index]
    else:
        if default is not _default:
            return default
        raise ValueError(f'{etype_dim} is not a valid entity attribute index '
                         f"in {ds.__class__.__name__}.")


def entity_dim2node(ds, etype_dim: int, index=None, dtype=None) -> Tensor:
    r"""Get the <entiry>_to_node sparse matrix by entity's top dimension."""
    entity = entity_dim2tensor(ds, etype_dim, index)
    location = entity_dim2tensor(ds, -etype_dim, default=None)
    return mesh_top_csr(entity, ds.number_of_nodes(), location, dtype=dtype)


##################################################
### Mesh Data Structure Base
##################################################

class MeshDS():
    _STORAGE_ATTR = ['cell', 'face', 'edge', 'cell_location','face_location']
    def __init__(self, NN: int, TD: int) -> None:
        self._entity_storage: Dict[int, Tensor] = {}
        self.NN = NN
        self.TD = TD

    @overload
    def __getattr__(self, name: EntityName) -> Tensor: ...
    def __getattr__(self, name: str):
        if name not in self._STORAGE_ATTR:
            return self.__dict__[name]
        etype_dim = entity_str2dim(self, name)
        return entity_dim2tensor(self, etype_dim)

    def __setattr__(self, name: str, value: torch.Any) -> None:
        if name in self._STORAGE_ATTR:
            if not hasattr(self, '_entity_storage'):
                raise RuntimeError('please call super().__init__() before setting attributes.')
            etype_dim = entity_str2dim(self, name)
            self._entity_storage[etype_dim] = value
        else:
            super().__setattr__(name, value)

    ### cuda
    def to(self, device: Union[_device, str, None]=None, non_blocking=False):
        for entity_tensor in self._entity_storage.values():
            entity_tensor.to(device, non_blocking=non_blocking)
        for attr in self.__dict__:
            value = self.__dict__[attr]
            if isinstance(value, torch.Tensor):
                self.__dict__[attr] = value.to(device, non_blocking=non_blocking)
        return self

    ### properties
    def top_dimension(self) -> int: return self.TD
    @property
    def itype(self) -> _dtype: return self.cell.dtype
    @property
    def device(self) -> _device: return self.cell.device

    ### counters
    def count(self, etype: Union[int, str]) -> int:
        """Return the number of entities of the given type."""
        if etype in ('node', 0):
            return self.NN
        if isinstance(etype, str):
            edim = entity_str2dim(self, etype)
        if -edim in self._entity_storage: # for polygon mesh
            return self._entity_storage[-edim].size(0) - 1
        return entity_dim2tensor(self, edim).size(0) # for homogeneous mesh

    def number_of_nodes(self): return self.NN
    def number_of_edges(self): return self.count('edge')
    def number_of_faces(self): return self.count('face')
    def number_of_cells(self): return self.count('cell')

    @overload
    def entity(self, etype: Union[int, str], index: Optional[Index]=None) -> Tensor: ...
    @overload
    def entity(self, etype: Union[int, str], index: Optional[Index]=None, *, default: _T) -> Union[Tensor, _T]: ...
    def entity(self, etype: Union[int, str], index: Optional[Index]=None, *, default=_default):
        """Get entities in mesh structure.

        Args:
            index (int | slice | Tensor): The index of the entity.
            etype (int | str): The topology dimension of the entity, or name
            'cell' | 'face' | 'edge'. Note that 'node' is not available in data structure.
            For polygon meshes, the names 'cell_location' | 'face_location' may also be
            available, and the `index` argument is applied on the flattened entity tensor.
            index (int | slice | Tensor): The index of the entity.
            default (Any): The default value if the entity is not found.

        Returns:
            Tensor: Entity or the default value.
        """
        if isinstance(etype, str):
            etype = entity_str2dim(self, etype)
        return entity_dim2tensor(self, etype, index, default=default)

    def total_face(self) -> Tensor:
        raise NotImplementedError

    def total_edge(self) -> Tensor:
        raise NotImplementedError

    ### topology
    def cell_to_node(self, index: Optional[Index]=None, *, dtype: Optional[_dtype]=None) -> Tensor:
        etype = self.top_dimension()
        return entity_dim2node(self, etype, index, dtype=dtype)

    def face_to_node(self, index: Optional[Index]=None, *, dtype: Optional[_dtype]=None) -> Tensor:
        etype = self.top_dimension() - 1
        return entity_dim2node(self, etype, index, dtype=dtype)

    def edge_to_node(self, index: Optional[Index]=None, *, dtype: Optional[_dtype]=None) -> Tensor:
        return entity_dim2node(self, 1, index, dtype)

    def cell_to_edge(self, index: Index=_S, *, dtype: Optional[_dtype]=None,
                     return_sparse=False) -> Tensor:
        if not hasattr(self, 'cell2edge'):
            raise RuntimeError('Please call construct() first or make sure the cell2edge'
                               'has been constructed.')
        cell2edge = self.cell2edge[index]
        if return_sparse:
            return mesh_top_csr(cell2edge[index, :2], self.number_of_edges(), dtype=dtype)
        else:
            return cell2edge[index]

    def face_to_cell(self, index: Index=_S, *, dtype: Optional[_dtype]=None,
                     return_sparse=False) -> Tensor:
        if not hasattr(self, 'face2cell'):
            raise RuntimeError('Please call construct() first or make sure the face2cell'
                               'has been constructed.')
        face2cell = self.face2cell[index]
        if return_sparse:
            return mesh_top_csr(face2cell[index, :2], self.number_of_cells(), dtype=dtype)
        else:
            return face2cell[index]

    ### boundary
    def boundary_node_flag(self) -> Tensor:
        """Return a boolean tensor indicating the boundary nodes.

        Returns:
            Tensor: boundary node flag.
        """
        NN = self.number_of_nodes()
        bd_face_flag = self.boundary_face_flag()
        kwargs = {'dtype': bd_face_flag.dtype, 'device': bd_face_flag.device}
        bd_face2node = self.entity('face', index=bd_face_flag)
        bd_node_flag = torch.zeros((NN,), **kwargs)
        bd_node_flag[bd_face2node.ravel()] = True
        return bd_node_flag

    def boundary_face_flag(self) -> Tensor:
        """Return a boolean tensor indicating the boundary faces.

        Returns:
            Tensor: boundary face flag.
        """
        return self.face2cell[:, 0] == self.face2cell[:, 1]

    def boundary_cell_flag(self) -> Tensor:
        """Return a boolean tensor indicating the boundary cells.

        Returns:
            Tensor: boundary cell flag.
        """
        NC = self.number_of_cells()
        bd_face_flag = self.boundary_face_flag()
        kwargs = {'dtype': bd_face_flag.dtype, 'device': bd_face_flag.device}
        bd_face2cell = self.face2cell[bd_face_flag, 0]
        bd_cell_flag = torch.zeros((NC,), **kwargs)
        bd_cell_flag[bd_face2cell.ravel()] = True
        return bd_cell_flag

    def boundary_node_index(self): return self.boundary_node_flag().nonzero().ravel()
    # TODO: finish this:
    # def boundary_edge_index(self): return self.boundary_edge_flag().nonzero().ravel()
    def boundary_face_index(self): return self.boundary_face_flag().nonzero().ravel()
    def boundary_cell_index(self): return self.boundary_cell_flag().nonzero().ravel()


    ### Homogeneous Mesh ###
    def is_homogeneous(self) -> bool:
        """Return True if the mesh is homogeneous.

        Returns:
            bool: Homogeneous indicator.
        """
        return self.cell.ndim == 2

    ccw: Tensor
    localEdge: Tensor
    localFace: Tensor

    number_of_vertices_of_cells: _int_func = lambda self: self.cell.shape[-1]
    number_of_nodes_of_cells = number_of_vertices_of_cells
    number_of_edges_of_cells: _int_func = lambda self: self.localEdge.shape[0]
    number_of_faces_of_cells: _int_func = lambda self: self.localFace.shape[0]
    number_of_vertices_of_faces: _int_func = lambda self: self.localFace.shape[-1]
    number_of_vertices_of_edges: _int_func = lambda self: self.localEdge.shape[-1]

    def total_face(self) -> Tensor:
        NVF = self.number_of_faces_of_cells()
        cell = self.entity(self.TD)
        local_face = self.localFace
        total_face = cell[..., local_face].reshape(-1, NVF)
        return total_face

    def total_edge(self) -> Tensor:
        NVE = self.number_of_vertices_of_edges()
        cell = self.entity(self.TD)
        local_edge = self.localEdge
        total_edge = cell[..., local_edge].reshape(-1, NVE)
        return total_edge

    def construct(self):
        if not self.is_homogeneous():
            raise RuntimeError('Can not construct for a non-homogeneous mesh.')

        NC = self.cell.shape[0]
        NFC = self.cell.shape[1]

        totalFace = self.total_face()
        _, i0_np, j_np = np.unique(
            torch.sort(totalFace, dim=1)[0].cpu().numpy(),
            return_index=True,
            return_inverse=True,
            axis=0
        )
        self.face = totalFace[i0_np, :] # this also adds the edge in 2-d meshes
        NF = i0_np.shape[0]

        i1_np = np.zeros(NF, dtype=i0_np.dtype)
        i1_np[j_np] = np.arange(NFC*NC, dtype=i0_np.dtype)

        self.cell2edge = torch.from_numpy(j_np).to(self.device).reshape(NC, NFC)
        self.cell2face = self.cell2edge

        face2cell_np = np.stack([i0_np//NFC, i1_np//NFC, i0_np%NFC, i1_np%NFC], axis=-1)
        self.face2cell = torch.from_numpy(face2cell_np).to(self.device)
        self.edge2cell = self.face2cell

        if self.TD == 3:
            NEC = self.number_of_edges_of_cells()

            total_edge = self.total_edge()
            _, i2, j = np.unique(
                torch.sort(total_edge, dim=1)[0].cpu().numpy(),
                return_index=True,
                return_inverse=True,
                axis=0
            )
            self.edge = total_edge[i2, :]
            self.cell2edge = torch.from_numpy(j).to(self.device).reshape(NC, NEC)

        elif self.TD == 2:
            self.edge2cell = self.face2cell

        logger.info(f"Mesh toplogy relation constructed, with {NF} edge (or face), "
                    f"on device {self.device}")


##################################################
### Mesh Base
##################################################

class Mesh():
    ds: MeshDS
    node: Tensor

    def to(self, device: Union[_device, str, None]=None, non_blocking: bool=False):
        self.ds.to(device, non_blocking)
        self.node = self.node.to(device, non_blocking)
        return self

    @property
    def ftype(self) -> _dtype: return self.node.dtype
    @property
    def device(self) -> _device: return self.node.device
    def geo_dimension(self) -> int: return self.node.shape[-1]
    def top_dimension(self) -> int: return self.ds.top_dimension()
    GD = property(geo_dimension)
    TD = property(top_dimension)

    def multi_index_matrix(self, p: int, etype: int) -> Tensor:
        return F.multi_index_matrix(p, etype, dtype=self.ds.itype, device=self.device)

    def count(self, etype: Union[int, str]) -> int: return self.ds.count(etype)
    def number_of_cells(self) -> int: return self.ds.number_of_cells()
    def number_of_faces(self) -> int: return self.ds.number_of_faces()
    def number_of_edges(self) -> int: return self.ds.number_of_edges()
    def number_of_nodes(self) -> int: return self.ds.number_of_nodes()
    def entity(self, etype: Union[int, str], index: Optional[Index]=None) -> Tensor:
        if etype in ('node', 0):
            return self.node if index is None else self.node[index]
        else:
            return self.ds.entity(etype, index)

    def entity_barycenter(self, etype: Union[int, str], index: Optional[Index]=None) -> Tensor:
        """Get the barycenter of the entity.

        Args:
            etype (int | str): The topology dimension of the entity, or name
            'cell' | 'face' | 'edge' | 'node'. Returns sliced node if 'node'.
            index (int | slice | Tensor): The index of the entity.

        Returns:
            Tensor: A 2-d tensor containing barycenters of the entity.
        """
        if etype in ('node', 0):
            return self.node if index is None else self.node[index]

        node = self.node
        if isinstance(etype, str):
            etype = entity_str2dim(self.ds, etype)
        etn = entity_dim2node(self.ds, etype, index, dtype=node.dtype)
        return F.entity_barycenter(etn, node)

    def edge_length(self, index: Index=_S, out=None) -> Tensor:
        """Calculate the length of the edges.

        Args:
            index (int | slice | Tensor, optional): Index of edges.
            out (Tensor, optional): The output tensor. Defaults to None.

        Returns:
            Tensor: Length of edges, shaped [NE,].
        """
        edge = self.entity(1, index=index)
        return F.edge_length(self.node[edge], out=out)

    def edge_normal(self, index: Index=_S, unit: bool=False, out=None) -> Tensor:
        """Calculate the normal of the edges.

        Args:
            index (int | slice | Tensor, optional): Index of edges.
            unit (bool, optional): _description_. Defaults to False.
            out (Tensor, optional): _description_. Defaults to None.

        Returns:
            Tensor: _description_
        """
        edge = self.entity(1, index=index)
        return F.edge_normal(self.node[edge], unit=unit, out=out)

    def edge_unit_normal(self, index: Index=_S, out=None) -> Tensor:
        """Calculate the unit normal of the edges.
        Equivalent to `edge_normal(index=index, unit=True)`.
        """
        return self.edge_normal(index=index, unit=True, out=out)

    def integrator(self, q: int, etype: Union[int, str]='cell', qtype: str='legendre') -> Quadrature:
        """Get the quadrature points and weights."""
        raise NotImplementedError

    def shape_function(self, bc: Tensor, p: int=1, *, index: Index=_S,
                       variable: str='u', mi: Optional[Tensor]=None) -> Tensor:
        """Shape function value on the given bc points, in shape (..., ldof).

        Args:
            bc (Tensor): The bc points, in shape (..., NVC).
            p (int, optional): The order of the shape function. Defaults to 1.
            index (int | slice | Tensor, optional): The index of the cell.
            variable (str, optional): The variable name. Defaults to 'u'.
            mi (Tensor, optional): The multi-index matrix. Defaults to None.

        Returns:
            Tensor: The shape function value with shape (..., ldof). The shape will\
            be (..., 1, ldof) if `variable == 'x'`.
        """
        raise NotImplementedError(f"shape function is not supported by {self.__class__.__name__}")

    def grad_shape_function(self, bc: Tensor, p: int=1, *, index: Index=_S,
                            variable: str='u', mi: Optional[Tensor]=None) -> Tensor:
        raise NotImplementedError(f"grad shape function is not supported by {self.__class__.__name__}")

    def hess_shape_function(self, bc: Tensor, p: int=1, *, index: Index=_S,
                            variable: str='u', mi: Optional[Tensor]=None) -> Tensor:
        raise NotImplementedError(f"hess shape function is not supported by {self.__class__.__name__}")


class HomogeneousMesh(Mesh):
    # entity
    def entity_barycenter(self, etype: Union[int, str], index: Optional[Index]=None) -> Tensor:
        node = self.entity('node')
        if etype in ('node', 0):
            return node if index is None else node[index]
        entity = self.ds.entity(etype, index)
        return F.homo_entity_barycenter(entity, node)

    def bc_to_point(self, bcs: Union[Tensor, Sequence[Tensor]],
                    etype: Union[int, str]='cell', index: Index=_S) -> Tensor:
        """Convert barycenter coordinate points to cartesian coordinate points
        on mesh entities.
        """
        node = self.entity('node')
        entity = self.ds.entity(etype, index)
        # TODO: finish this
        # ccw = getattr(self.ds, 'ccw', None)
        ccw = None
        return F.bc_to_points(bcs, node, entity, ccw)

    ### ipoints
    def interpolation_points(self, p: int, index: Index=_S) -> Tensor:
        raise NotImplementedError

    def cell_to_ipoint(self, p: int, index: Index=_S) -> Tensor:
        raise NotImplementedError

    def face_to_ipoint(self, p: int, index: Index=_S) -> Tensor:
        raise NotImplementedError

    def edge_to_ipoint(self, p: int, index: Index=_S) -> Tensor:
        """Get the relationship between edges and integration points."""
        NN = self.number_of_nodes()
        NE = self.number_of_edges()
        edges = self.ds.edge[index]
        kwargs = {'dtype': edges.dtype, 'device': self.device}
        indices = torch.arange(NE, **kwargs)[index]
        return torch.cat([
            edges[:, 0].reshape(-1, 1),
            (p-1) * indices.reshape(-1, 1) + torch.arange(p-1, **kwargs) + NN,
            edges[:, 1].reshape(-1, 1),
        ], dim=-1)


class SimplexMesh(HomogeneousMesh):
    # ipoints
    def number_of_local_ipoints(self, p: int, iptype: Union[int, str]='cell'):
        if isinstance(iptype, str):
            iptype = entity_str2dim(self.ds, iptype)
        return F.simplex_ldof(p, iptype)

    def number_of_global_ipoints(self, p: int):
        return F.simplex_gdof(p, self)

    # shape function
    def grad_lambda(self, index: Index=_S) -> Tensor:
        raise NotImplementedError

    def shape_function(self, bc: Tensor, p: int=1, *, index: Index=_S,
                       variable: str='u', mi: Optional[Tensor]=None) -> Tensor:
        TD = bc.shape[-1] - 1
        mi = mi or F.multi_index_matrix(p, TD, dtype=self.ds.itype, device=self.device)
        phi = K.simplex_shape_function(bc, p, mi)
        if variable == 'u':
            return phi
        elif variable == 'x':
            return phi.unsqueeze_(1)
        else:
            raise ValueError("Variable type is expected to be 'u' or 'x', "
                             f"but got '{variable}'.")

    def grad_shape_function(self, bc: Tensor, p: int=1, *, index: Index=_S,
                            variable: str='u', mi: Optional[Tensor]=None) -> Tensor:
        TD = bc.shape[-1] - 1
        mi = mi or F.multi_index_matrix(p, TD, dtype=self.ds.itype, device=self.device)
        R = K.simplex_grad_shape_function(bc, p, mi) # (NQ, ldof, bc)
        if variable == 'u':
            return R
        elif variable == 'x':
            Dlambda = self.grad_lambda(index=index)
            gphi = torch.einsum('...bm, kjb -> k...jm', Dlambda, R) # (NQ, NC, ldof, dim)
            # NOTE: the subscript 'k': NQ, 'm': dim, 'j': ldof, 'b': bc, '...': cell
            return gphi
        else:
            raise ValueError("Variable type is expected to be 'u' or 'x', "
                             f"but got '{variable}'.")
