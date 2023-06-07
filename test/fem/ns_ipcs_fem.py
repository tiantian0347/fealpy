import numpy as np
from  mumps import DMumpsContext
import scipy 

import matplotlib.pyplot as plt
import matplotlib

from  fealpy.timeintegratoralg import UniformTimeLine
from fealpy.mesh import MeshFactory as MF
from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.decorator import cartesian,barycentric
from fealpy.boundarycondition import DirichletBC
from fealpy.quadrature import GaussLegendreQuadrature
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix,hstack,vstack,spdiags

from scipy.linalg import solve
from fealpy.tools.show import showmultirate

from fealpy.mesh import TriangleMesh
from fealpy.functionspace import LagrangeFESpace
from fealpy.fem import ScalarDiffusionIntegrator, VectorMassIntegrator
from fealpy.fem import NSOperatorIntegrator,PressIntegrator
from fealpy.fem import BilinearForm,MixedBilinearForm
from fealpy.fem import LinearForm
from fealpy.fem import VectorSourceIntegrator, ScalarSourceIntegrator
from fealpy.fem import DirichletBC

## 参数解析
degree =2 
dim = 2
ns = 16
nt = 50
doforder = 'sdofs'

eps = 1e-12
T = 10
rho = 1
mu = 1
inp = 8.0
outp = 0.0

@cartesian
def walldirichlet(p):
    return 0
@cartesian
def is_in_flow_boundary(p):
    return np.abs(p[..., 0]) < eps 
@cartesian
def is_out_flow_boundary(p):
    return np.abs(p[..., 0] - 1.0) < eps

@cartesian
def is_p_boundary(p):
    return (np.abs(p[..., 0]) < eps) | (np.abs(p[..., 0] - 1.0) < eps)
  
@cartesian
def is_wall_boundary(p):
    return (np.abs(p[..., 1]) < eps) | (np.abs(p[..., 1] - 1.0) < eps)

@cartesian
def usolution(p):
    x = p[...,0]
    y = p[...,1]
    u = np.zeros(p.shape)
    u[...,0] = 4*y*(1-y)
    return u

@cartesian 
def psolution(p):
    x = p[...,0]
    y = p[...,1]
    pp = np.zeros(p.shape)
    pp = 8*(1-x)
    return pp

domain = [0, 1, 0, 1]

mesh = TriangleMesh.from_unit_square(nx=ns, ny=ns)
nuspace = LagrangeFESpace(mesh,p=2,doforder=doforder)
npspace = LagrangeFESpace(mesh,p=1,doforder=doforder)
ubc = DirichletBC(nuspace, usolution)
pbc = DirichletBC(npspace, psolution)

smesh = MF.boxmesh2d(domain, nx=ns, ny=ns, meshtype='tri')
tmesh = UniformTimeLine(0, T, nt) # 均匀时间剖分

uspace = LagrangeFiniteElementSpace(smesh, p=degree)
pspace = LagrangeFiniteElementSpace(smesh, p=degree-1)

u0 = nuspace.function(dim=2)
us = nuspace.function(dim=2)
u1 = nuspace.function(dim=2)

p0 = npspace.function()
p1 = npspace.function()

dt = tmesh.dt

# 第一个
Vbform0 = BilinearForm(2*(nuspace,))
Vbform0.add_domain_integrator(VectorMassIntegrator(rho/dt))
Vbform0.assembly()
A1 = Vbform0.get_matrix()

Vbform1 = BilinearForm(2*(nuspace,))
Vbform1.add_domain_integrator(NSOperatorIntegrator(mu))
Vbform1.assembly()
A2 = Vbform1.get_matrix()
A = A1+A2

Vbform3 = MixedBilinearForm((npspace,),(nuspace,nuspace))
Vbform3.add_domain_integrator(PressIntegrator())
Vbform3.assembly()
D = Vbform3.get_matrix()

#组装第二个方程的左端矩阵
Sbform = BilinearForm(npspace)
Sbform.add_domain_integrator(ScalarDiffusionIntegrator(c=1))
Sbform.assembly()
B = Sbform.get_matrix()

#组装第三个方程的左端矩阵
Vbform2 = BilinearForm(2*(nuspace,))
Vbform2.add_domain_integrator(VectorMassIntegrator(c=1))
Vbform2.assembly()
C = Vbform2.get_matrix()

ctx = DMumpsContext()
ctx.set_silent()
errorMatrix = np.zeros((2,nt),dtype=np.float64)

for i in range(2): 
    # 下一个的时间层 t1
    t1 = tmesh.next_time_level()
    print("t1=", t1)

    #组装第一个方程的右端向量
    fb1 = A1@u0.flatten(order='C')

    @barycentric
    def f1(bcs,index):
        b2 = np.einsum('imj,ikmj->ijk',u0(bcs,index),u0.grad_value(bcs,index))
        return rho*b2

    lform = LinearForm(2*(nuspace,))
    lform.add_domain_integrator(VectorSourceIntegrator(f1))
    lform.assembly()
    fb2 = lform.get_vector()  
    
    fb3 = A2@u0.flatten(order='C')
    fb4 = D@p0.flatten(order='C')

    b1 = fb1-fb2+fb4-mu*fb3
    AA,b1 = ubc.apply(A,b1)

    ctx.set_centralized_sparse(AA)
    x = b1.copy()
    ctx.set_rhs(x)
    ctx.run(job=6)
    us.flat[:] = x

    #组装第二个方程的右端向量
     
    @barycentric
    def f2(bcs,index):
        b2 = us.grad_value(bcs,index)[...,0,0,:]+us.grad_value(bcs,index)[...,1,1,:]
        return -1/dt*b2

    lform = LinearForm(pspace)
    lform.add_domain_integrator(ScalarSourceIntegrator(f2))
    lform.assembly()
    b21 = lform.get_vector()   
    b22 = B@p0
    b2 = b21+b22
    
    ctx.set_centralized_sparse(B)
    x = b2.copy()
    ctx.set_rhs(x)
    ctx.run(job=6)
    p1[:] = x[:]

    #组装第三个方程的右端向量
    @barycentric
    def f3(bcs,index):
        b3 = p1.grad_value(bcs,index)-p0.grad_value(bcs,index)
        return dt*b3

    lform = LinearForm((nuspace,)*2)
    lform.add_domain_integrator(VectorSourceIntegrator(f3))
    lform.assembly()
    tb2 = lform.get_vector()   
    
    tb1 = C@us.flatten(order='C')
    b3 = tb1 - tb2
    
    ctx.set_centralized_sparse(C)
    x = b3.copy()
    ctx.set_rhs(x)
    ctx.run(job=6)
    
    u1.flat[:] = x[:]

    co1 = usolution(smesh.node)
    NN = smesh.number_of_nodes()
    co2 = u1[:,:NN].transpose(1,0)
    errorMatrix[0,i] = np.abs(co1-co2).max()
    #errorMatrix[0,i] = uspace.integralalg.error(usolution,u1)
    #errorMatrix[1,i] = pspace.integralalg.error(psolution,p1)
    print("结果",np.sum(np.abs(u1))) 
    u0[:] = u1
    p0[:] = p1

    # 时间步进一层 
    tmesh.advance()

# 画图
'''
print(errorMatrix[0,:])
ctx.destroy()
fig1 = plt.figure()
node = smesh.node
x = tuple(node[:,0])
y = tuple(node[:,1])
NN = smesh.number_of_nodes()
u = u1.transpose()[:NN]
ux = tuple(u[:,0])
uy = tuple(u[:,1])

o = ux
norm = matplotlib.colors.Normalize()
cm = matplotlib.cm.copper
sm = matplotlib.cm.ScalarMappable(cmap=cm,norm=norm)
sm.set_array([])
plt.quiver(x,y,ux,uy,color=cm(norm(o)))
plt.colorbar(sm)
plt.show()
'''