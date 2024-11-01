#!/usr/bin/python3
'''!    	
	@Author: wpx
	@File Name: solver.py
	@Mail: wpx15673207315@gmail.com 
	@Created Time: Sat 26 Oct 2024 04:18:00 PM CST
	@bref 
	@ref 
'''  
from fealpy.backend import backend_manager as bm
from fealpy.decorator import cartesian,barycentric
from fealpy.fem import BilinearForm, LinearForm, BlockForm, LinearBlockForm
from fealpy.fem import (ScalarConvectionIntegrator, 
                        ScalarDiffusionIntegrator, 
                        ScalarMassIntegrator,
                        SourceIntegrator)

from fealpy.fem import (BoundaryFaceMassIntegrator,
                        BoundaryFaceSourceIntegrator)
                        
                        
                        


class Solver():
    def __init__(self, pde, mesh, pspace, phispace, uspace, dt, q=5):
        self.mesh = mesh
        self.phispace = phispace
        self.pspace = pspace
        self.uspace = uspace
        self.pde = pde
        self.dt = dt
        self.q = q
    
    def dirction_grad(self, u, n):
        result = bm.einsum('eld, ed', u(bcs, index), n[index,:])
        return result

    def CH_BForm(self):
        phispace = self.phispace
        dt = self.dt
        L_d = self.pde.L_d
        epsilon = self.pde.epsilon
        s = self.pde.s
        V_s = self.pde.V_s
        q = self.q

        A00 = BilinearForm(phispace)
        M = ScalarMassIntegrator(coef=3, q=q)
        self.phi_C = ScalarConvectionIntegrator(q=q)
        A00.add_integrator([M, self.phi_C])

        A01 = BilinearForm(phispace)
        A01.add_integrator(ScalarDiffusionIntegrator(coef=2*dt*L_d, q=q))
        
        A10 = BilinearForm(phispace)
        A10.add_integrator(ScalarDiffusionIntegrator(coef=-epsilon, q=q))
        A10.add_integrator(ScalarMassIntegrator(coef=-s/epsilon, q=q))
        A10.add_integrator(BoundaryFaceMassIntegrator(coef=-3/(2*dt*V_s), q=q, threshold=self.pde.is_wall_boundary))     

        A11 = BilinearForm(phispace)
        A11.add_integrator(ScalarMassIntegrator(q=q))

        A = BlockForm([[A00, A01], [A10, A11]]) 
        return A

    def CH_LForm(self):
        phispace = self.phispace
        dt = self.dt
        L_d = self.pde.L_d
        epsilon = self.pde.epsilon
        s = self.pde.s
        q = self.q

        L0 = LinearForm(phispace)
        self.phi_SI = SourceIntegrator(q=q)
        L0.add_integrator(self.phi_SI)

        L1 = LinearForm(phispace)
        self.mu_SI = SourceIntegrator(q=q)
        self.mu_BFSI = BoundaryFaceSourceIntegrator(q=q, threshold=self.pde.is_wall_boundary)
        L1.add_integrator(self.mu_SI)

        L = LinearBlockForm([L0, L1])
        return L

    def CH_update(self, u_0, u_1, phi_0, phi_1):
        dt = self.dt
        s = self.pde.s
        epsilon =self.pde.epsilon
        tangent = self.mesh.edge_unit_tangent()
        V_s = self.pde.V_s

        # BilinearForm
        @barycentric
        def C_coef(bcs, index):
            return 4*dt*u_1(bcs, index)
        self.phi_C.coef = C_coef
        self.phi_C.clear()
        
        # LinearForm 
        @barycentric
        def phi_coef(bcs, index):
            result = 4*phi_1(bcs, index) - phi_0(bcs, index) 
            result1 = bm.einsum('jid, jid->ji', u_0(bcs, index), phi_1.grad_value(bcs, index))
            result += 2*dt*result1
            return result
        self.phi_SI.source = phi_coef
        self.phi_SI.clear()
       
        @barycentric
        def mu_coef(bcs, index):
            result = -2*(1+s)*phi_1(bcs, index) + (1+s)*phi_0(bcs, index)
            result += 2*phi_1(bcs, index)**3 - phi_0(bcs, index)**3
            result /= epsilon
            return result
        self.mu_SI.source = mu_coef
        self.mu_SI.clear()
        
        
        @bartcentric
        def mu_BF_coef(bcs, index):
            result0 = (-4*phi_1(bcs, index) + phi_0(bcs, index))/2*(dt*V_s)
            result1 = 2*bm.einsum('eld, ed', u1(bcs, index), tangent[index,:])\
                      *bm.einsum('eld, ed', phi_1.grad_value(bcs, index), tangent[index,:])
        self.mu_BF_SI.clear()
        self.mu_BF_SI.source = mu_BF_coef

    def NS_BForm(self):
        pspace = self.pspace
        uspace = self.uspace
        dt = self.dt
        R = self.pde.R
        q = self.q

        A00 = BilinearForm(pspace)
        M = ScalarMassIntegrator(coef=3*R, q=q)
        self.u_C = ScalarConvectionIntegrator(q=q)
        D = ScalarDiffusionIntegrator(coef=2*dt, q=q)
        A00.add_integrator([M, self.u_C, D])
        ## TODO:差一个边界积分子

        A01_bform = BilinearForm((pspace, uspace))
        A01_bform.add_integrator(PressWorkIntegrator(-2*dt, q=q))
 
        A = BlockForm([[A00, A01], [A01.T, None]]) 
        return A

    def NS_LForm(self, q=None):
        pspace = self.pspace
        uspace = self.uspace
        dt = self.dt
        R = self.pde.R
        q = self.q

        L0 = LinearForm(pspace) 
        self.u_SI = SourceIntegrator(q=q)
        #差一个边界积分子
        L0.add_integrator(self.u_SI)

        L1 = LinearForm(uspace)
        L = LinearBlockForm([L0, L1])
        return L

    def NS_update(self, u_0, u_1, mu_2, phi_2):
        dt = self.dt
        R = self.pde.R
        lam = self.pde.lam

        def u_coef(bcs, index):
            result = R*4*u1(bcs, index) - u0(bcs, index)
            result += 2*R*dt*bm.einsum('jimd, jimd->jimd', u_0(bcs, index), u_1.grad_value(bcs, index))
            result += 2*lam*dt*mu_2(bcs, index)
            return 4*u1(bcs, index)

        self.u_SI.clear()
        self.u_SI.source = u_coef

