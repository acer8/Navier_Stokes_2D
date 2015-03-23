# -*- coding: utf-8 -*-
"""
Created on Wed Dec 31 13:36:42 2014

@author: acerlinux
"""

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import LinearOperator
import scipy.sparse
import scipy.sparse.linalg as slg
from pyamg import smoothed_aggregation_solver
from matplotlib import cm
#from pylab import plot, draw, axis, clf, title, ion, ioff, contourf, show, streamplot
#from time import sleep
#import pylab
import time
import sys
import copy
sys.path.append("/home/acerlinux/Documents/share_folder")
import structure2

# this class contains the linear system solvers for both velocity and pressure
# returns the linear system in sparse matrix form and linear operator form
class LinearSystem_solver():
    def __init__(self, Re, mesh):
        self.mesh = mesh
        self.Re = Re
    
    # linear systemas for velocities (in the form of sparse matrices)
    def Linsys_velocity_matrix(self, velocity):
        m = self.mesh.m
        n = self.mesh.n
        dt = self.mesh.dt
        dx = self.mesh.dx
        dy = self.mesh.dy
        Re = self.Re
        # for square domain only, lx = ly and dx = dy = dh
        dh = dx
        a = dt/(2*Re*dh**2)
        b = (Re*dh**2)/dt + 2

        # Dirichlet boundary condition is applied
        if velocity == "u":
            # construct matrix A: Au = rhs
            # A is symmetric and positive definite with dimension NxN
            N = m*(n-1)
            # block matrix
            maindiag = np.zeros(n-1)
            maindiag[:] = 2*b
            sidediag = np.zeros(n-2)
            sidediag[:] = -1
            B = scipy.sparse.diags([maindiag,sidediag,sidediag],[0,-1,1])
            A1 = scipy.sparse.kron(scipy.sparse.eye(m,m),B)
            #print A1.todense()            
            md = np.zeros(N)
            md[0:n-1] = 3.0
            md[-(n-1):] = 3.0
            sdl = -np.ones(N-(n-1))
            sdl[-(n-1):] = -2.0
            sdu = sdl[::-1]
            sdll = np.zeros((n-2)*(n-1))
            sdll[-(n-1):] = 0.2
            sduu = sdll[::-1]
            A2 = scipy.sparse.diags([md,sdl,sdu,sdll,sduu],[0,-(n-1),n-1,-2*(n-1),2*(n-1)])
            #print A2.todense()
            A = scipy.sparse.csc_matrix((A1+A2)*a)
            #print A.todense()
            #print np.linalg.cond(np.matrix(A.todense())), "condition number velocity"
            A_linop = scipy.sparse.linalg.aslinearoperator(A)
            return [A, A_linop]
        
        elif velocity == "v":
            # construct A: Av = rhs
            N = (m-1)*n
            # block matrix
            maindiag = np.zeros(n)
            maindiag[:] = 2*b
            maindiag[0] = 2*b+3
            maindiag[-1] = 2*b+3
            sidediagl = -np.ones(n-1)
            sidediagl[-1] = -2.0
            sidediagu = sidediagl[::-1]
            sdl = np.zeros(n-2)
            sdl[-1] = 0.2
            sdu = sdl[::-1]
            B = scipy.sparse.diags([maindiag,sidediagl,sidediagu,sdl,sdu],[0,-1,1,-2,2])
            A1 = scipy.sparse.kron(scipy.sparse.eye(m-1,m-1),B)
            sd = -np.ones(N-n)
            A2 = scipy.sparse.diags([sd,sd],[-n,n])
            A = scipy.sparse.csc_matrix((A1+A2)*a)
            #print A.todense()
            A_linop = scipy.sparse.linalg.aslinearoperator(A)

            return [A,A_linop]
    
    # the linear system solver for velocity fields
    # returns VelocityField instances (only interior points are calculated)
    # ALuv = [A, A_linop]: contains the lineary system in the sparse matrix and linear operator form
    # rhsuv = [rhsu, rhsv]: right hand side of u and v velocities (needs to be boundary corrected)
    def Linsys_velocity_solver(self, ALuv, rhsuv, tol=1e-12):
        m = self.mesh.m
        n = self.mesh.n
        dx = self.mesh.dx
        dy = self.mesh.dy
        # for square domain only, lx = ly and dx = dy = dh
        dh = dx
        uvl = []
        # only solving the interior points, rhsuv needs to be boundary corrected
        ## solve for u and v sequentially
        for i in xrange(2):
            ## for u
            if i == 0:
                N = m*(n-1)
                row = m
                col = n-1
            ## for v
            else:
                 N = (m-1)*n
                 row = m-1
                 col = n
            ## convert rhs into vector (m*(n-1))
            rhs = rhsuv.get_uv()[i]
            rhs = rhs.reshape(N)
            AL = ALuv[i]                
            A = AL[0]
            A_linop = AL[1]
            u = scipy.sparse.linalg.bicg(A_linop, rhs, tol)
            u = u[0].reshape(row, col)
            uvl.append(u)
            AL = []
            rhs = 0          
            row = 0
            col = 0
        # mstar: m* the Gauge variable in the form of VelocityField object
        mstar = structure2.VelocityField(uvl[0], uvl[1], self.mesh)
        return mstar
    
    # the Pressure Poisson lineary system
    # uses algebraic multigrid method
    def Poisson_pressure_matrix(self, preconditioner):
        m = self.mesh.m
        n = self.mesh.n
        dx = self.mesh.dx
        dy = self.mesh.dy
        # for square domain only, lx = ky and dx = dy = dh
        dh = dx
        # construct matrix A: Ap = rhs, p is pressure (with interior points)
        # Neumann boundary condition is applied
        # A is negative definite so use -A which is positive definite
        # block matrix                       
        maindiag = np.ones(n)
        maindiag[1:n-1] = (2*maindiag[1:n-1])
        sidediag = np.ones(n-1)
        B = scipy.sparse.diags([maindiag/(dh**2),-sidediag/(dh**2),-sidediag/(dh**2)],[0,-1,1])
        A1 = scipy.sparse.kron(scipy.sparse.eye(m,n),B)
        A2 = scipy.sparse.kron(B, scipy.sparse.eye(m,n))
        A = A1+A2
        A = scipy.sparse.csc_matrix(A)
        if preconditioner == "ILU":
            A_linop = scipy.sparse.linalg.aslinearoperator(A)
            A_ILU = slg.spilu(A,permc_spec='MMD_AT_PLUS_A')
            M = slg.LinearOperator(shape=(m*n,m*n),matvec=A_ILU.solve)
            #print A.todense()
            return [A_linop, M, A]
        
        elif preconditioner == "AMG":
            A = scipy.sparse.csr_matrix(A)
            B = np.ones((A.shape[0],1)) 
            mls = smoothed_aggregation_solver(A, B, max_coarse=10)
            #print A.todense()
            return mls, A
        
    # solves the Pressure Poisson problem using algebraic multigrid method
    def Poisson_pressure_solver(self, rhs, preconditioner, precd_AL, tol=1e-12):
        m = self.mesh.m
        n = self.mesh.n
        dt = self.mesh.dt
        dx = self.mesh.dx
        dy = self.mesh.dy
        # for square domain only, lx = ky and dx = dy = dh
        dh = dx
        
        # convert rhs into vector (m*n)
        rhs = rhs.get_value()
        #print type(rhs)
        rhs = (-rhs).reshape(m*n)
        N = m*n
        if preconditioner == "ILU":
            # use Incomplete LU to find a preconditioner
            A_linop = precd_AL[0]
            M = precd_AL[1]
            A = precd_AL[2]
            p = scipy.sparse.linalg.cg(A_linop, rhs, tol, maxiter=N, M=M)
            p = p[0].reshape(m,n)
            return structure2.CentredPotential(p, self.mesh)
        
        elif preconditioner == "AMG":
            # use Algebraic Multigrid method
            mls, A = precd_AL
            #print mls
            residuals = []
            #accelerated_residuals = []
            p = mls.solve(rhs,tol, accel='cg', residuals = residuals)
            (residuals[-1]/residuals[0])**(1.0/len(residuals))
            #accelerated_residuals = np.array(accelerated_residuals)/accelerated_residuals[0]
            #r = np.max(np.abs(accelerated_residuals))
            #print r, "residuals"
            p = p.reshape(m,n)
            Ap = A*np.matrix(np.ravel(p)).T
            r = rhs - np.array(Ap.T)
            #print r, "rp"
            #print np.max(np.abs(r)), "residual"
            # returns phi variable in the form of CentredPotential object
            return [structure2.CentredPotential(p, self.mesh), residuals]

# class Gauge_method(structure.mesh)
# the Gauge method main solver

class Gauge_method():
    def __init__(self, Re, mesh):
        self.Re = Re
        self.n = mesh.n
        self.m = mesh.m
        self.xu = mesh.xu
        self.yu = mesh.yu
        self.xv = mesh.xv
        self.yv = mesh.yv
        self.gds = mesh.gds
        self.sdomain = mesh.sdomain
        self.tdomain = mesh.tdomain
        self.Tn = mesh.Tn
        self.t0 = mesh.tdomain[0]
        self.dt = mesh.dt
        self.dx = mesh.dx
        self.dy = mesh.dy
        self.mesh = mesh
    
    # initial set up
    def setup(self, InCond_uv_init, Boundary_uv_type):
        ## InCond_uv: specifies the velocity initial condition 
        linsys_solver = LinearSystem_solver(self.Re, self.mesh)
        phi_mat_AMG = linsys_solver.Poisson_pressure_matrix("AMG")
        m1_mat = linsys_solver.Linsys_velocity_matrix("u")
        m2_mat = linsys_solver.Linsys_velocity_matrix("v")
        
        InCond_uvcmp = structure2.VelocityComplete(self.mesh, InCond_uv_init, 0).complete(Boundary_uv_type)
        uv_cmp = copy.copy(InCond_uvcmp)        
        mn_cmp = copy.copy(uv_cmp)
        initial_setup_parameters = [phi_mat_AMG, m1_mat, m2_mat, InCond_uvcmp, uv_cmp, mn_cmp]
        return initial_setup_parameters
        
    def iterative_solver(self, Boundary_uv_type, Tn, initial_setup_parameters, Forcing_term=0):
        n = self.n
        m = self.m
        dx = self.dx
        dy = self.dy
        dt = self.dt
        Re = self.Re
        phi_mat_AMG = initial_setup_parameters[0]
        m1_mat = initial_setup_parameters[1]
        m2_mat = initial_setup_parameters[2]
        # uvold_cmp: u and v velocity fields at time n-1
        # cmp: in the completed format (interior + boundary + ghost nodes)
        uvold_cmp = initial_setup_parameters[3]
        # uv_cmp: u and v at time n
        uv_cmp = initial_setup_parameters[4]
        # Gauge variable at time n (in the completed format)
        mn_cmp = initial_setup_parameters[5]
        # int: interior points only
        mn_int = structure2.VelocityField(mn_cmp.get_int_uv()[0], mn_cmp.get_int_uv()[1], self.mesh)
        # phiold: phi variable at time n-1
        phiold = np.zeros((m,n))
        phiold_cmp = structure2.CentredPotential(phiold, self.mesh).complete()
        # phin_cmp: phi variable at time n
        phin_cmp = np.copy(phiold_cmp)
        
        print Tn, "number of iterations"
        # main iterative solver
        for t in xrange(Tn):
            convc_uv = uv_cmp.non_linear_convection()
            preconvc_uv = uvold_cmp.non_linear_convection()
            diff_mn = mn_cmp.diffusion()
            rhs_mstar = mn_int + dt*(-1.5*convc_uv + 0.5*preconvc_uv + (1.0/(2*Re))*diff_mn +\
            Forcing_term)     
#            print rhs_mstar.get_uv()[0], "rhs_m1*"
#            print rhs_mstar.get_uv()[1], "rhs_m2*"
            
            # calculate the approximation to phi at time n+1
            gradphiuv = self.gradphi_app(phiold_cmp, phin_cmp)
            #print gradphiuv, "gradient of phi"
            # boundary correction step
            rhs_mstarcd = self.correct_boundary(rhs_mstar, t+1, Boundary_uv_type, gradphiuv)
#            print rhs_mstarcd.get_uv()[0], "rhs_m1* corrected"
#            print rhs_mstarcd.get_uv()[1], "rhs_m2* corrected"
            
            # solving for the Gauge variable m* 
            Linsys_solve = LinearSystem_solver(Re, self.mesh)
            mstar = Linsys_solve.Linsys_velocity_solver([m1_mat,m2_mat],  rhs_mstarcd)
#            print mstar.get_uv()[0], "m1*"
#            print mstar.get_uv()[1], "m2*"
            mstarcmp1, uvbnd_value = structure2.VelocityComplete(self.mesh, [mstar.get_uv()[0],  mstar.get_uv()[1]], t+1).complete(Boundary_uv_type, return_bnd=True)
            div_mstar = mstarcmp1.divergence()
            # solving for the phi variable
            [phi, residuals] = Linsys_solve.Poisson_pressure_solver(div_mstar, "AMG", phi_mat_AMG)
            #print phi.get_value(), "phi"
            if t == 0:
                #div_mn = np.zeros((m,n))
                div_mn = div_mstar
            else:
                div_mn = mn_cmp.divergence()
#            print div_mn.get_value(), "div_mn"	 
            # correct (normalise) phi 
            phiacd = self.phi_correction(phi, phin_cmp, div_mstar, div_mn)
#            print phiacd.get_value(), "phi corrected"
            
            # pressure correction step
            p = phiacd/dt - 1.0/(2*Re)*(div_mstar+div_mn)
            phiold_cmp = np.copy(phin_cmp)
            phin_cmp = np.copy(phi.complete())
            # velocity update stemp
            gradphi = phi.gradient()
    ##        print gradphi[0], "gradphi u"
    ##        print gradphi[1], "gradphi v"
            uvn_int = mstar - gradphi
#            print uvn_int.get_uv()[0], "u new interior"
#            print uvn_int.get_uv()[1], "v new interior"
            uvold_cmp = copy.copy(uv_cmp)
            uv_cmp = structure2.VelocityComplete(self.mesh, [uvn_int.get_uv()[0],  uvn_int.get_uv()[1]], t+1).complete(Boundary_uv_type)
#            print uv_cmp.get_uv()[0], "u new complete"
#            print uv_cmp.get_uv()[1], "v new complete"
#            print uv_cmp.get_int_uv()[0], "u new interior"
#            print uv_cmp.get_int_uv()[1], "v new interior"            
            # complete mstar
            mn_cmp = self.complete_mstar(mstar, uvbnd_value, phin_cmp)
            mn_int = structure2.VelocityField(mn_cmp.get_int_uv()[0], mn_cmp.get_int_uv()[1], self.mesh)            
#            print mn_cmp.get_uv()[0], "mn1 complete"
#            print mn_cmp.get_uv()[1], "mn2 complete"
            print "iteration "+str(t)
            #break
        return uv_cmp, p

    ## this function calculates graident of phi at time n+1
    # using second order approximation to gradient of phi^(n+1). Used in correcting m*
    # phi^{n+1} appro 2*phi^n - phi^{n-1}
    def gradphi_app(self, phiold_cmp, phin_cmp):
        n = self.n
        m = self.m
        dx = self.dx
        dy = self.dy
        dt = self.dt
        
        phiapp_cmp = 2*phin_cmp - phiold_cmp
        gradphiu = (phiapp_cmp[:,1:n+2] - phiapp_cmp[:,0:n+1])/dx
        gradphiv = (phiapp_cmp[1:m+2,:] - phiapp_cmp[0:m+1,:])/dy
        # obtain gradphiu North and South boundary by cubic interpolation
        gradphiuN = 5.0/16*(gradphiu[0,:] +3*gradphiu[1,:] - gradphiu[2,:]+0.2*gradphiu[3,:])
        gradphiuS = 5.0/16*(gradphiu[-1,:] +3*gradphiu[-2,:] - gradphiu[-3,:]+0.2*gradphiu[-4,:])
        gradphiu[0,:] = gradphiuN
        gradphiu[-1,:] = gradphiuS

        # obtain gradphiv West and East boundary by cubic interpolation
        gradphivW = 5.0/16*(gradphiv[:,0] +3*gradphiv[:,1] - gradphiv[:,2]+0.2*gradphiv[:,3])
        gradphivE = 5.0/16*(gradphiv[:,-1] +3*gradphiv[:,-2] - gradphiv[:,-3]+0.2*gradphiv[:,-4])
        gradphiv[:,0] = gradphivW
        gradphiv[:,-1] = gradphivE
        return [gradphiu, gradphiv]

    # boundary correction used in solving for Gauge variable
    def correct_boundary(self, rhs_mstar, t, Boundary_type, gradphiuv):
        # rhsuv is a VelocityField object with dimension interior u and v [(m*(n-1), (m-1)*n)]
        n = self.n
        m = self.m
        Re = self.Re
        dx = self.dx
        dy = self.dy
        dt = self.dt
        
        lam = dt/(2.0*Re)
        VC = structure2.VelocityComplete(self.mesh, [rhs_mstar.get_uv()[0], rhs_mstar.get_uv()[1]], t)
        gradphiu = gradphiuv[0]
        gradphiv = gradphiuv[1]
        
        if Boundary_type == "driven_cavity":
            uN = VC.bnd_driven_cavity('u')['N']
            uS = VC.bnd_driven_cavity('u')['S']
            uW = VC.bnd_driven_cavity('u')['W']
            uE = VC.bnd_driven_cavity('u')['E']
        
            vN = VC.bnd_driven_cavity('v')['N']
            vS = VC.bnd_driven_cavity('v')['S']
            vW = VC.bnd_driven_cavity('v')['W']
            vE = VC.bnd_driven_cavity('v')['E']

        elif Boundary_type == "Taylor":
            uN = VC.bnd_Taylor('u')['N'][1:n]
            uS = VC.bnd_Taylor('u')['S'][1:n]
            uW = VC.bnd_Taylor('u')['W']
            uE = VC.bnd_Taylor('u')['E']
        
            vN = VC.bnd_Taylor('v')['N']
            vS = VC.bnd_Taylor('v')['S']
            vW = VC.bnd_Taylor('v')['W'][1:m]
            vE = VC.bnd_Taylor('v')['E'][1:m]
        elif Boundary_type == "periodic_forcing_1":
            uN = VC.bnd_forcing_1('u')['N'][1:n]
            uS = VC.bnd_forcing_1('u')['S'][1:n]
            uW = VC.bnd_forcing_1('u')['W']
            uE = VC.bnd_forcing_1('u')['E']
        
            vN = VC.bnd_forcing_1('v')['N']
            vS = VC.bnd_forcing_1('v')['S']
            vW = VC.bnd_forcing_1('v')['W'][1:m]
            vE = VC.bnd_forcing_1('v')['E'][1:m]

        elif Boundary_type == "periodic_forcing_2":
            uN = VC.bnd_foring_2('u',t)['N'][1:n]
            uS = VC.bnd_foring_2('u',t)['S'][1:n]
            uW = VC.bnd_foring_2('u',t)['W']
            uE = VC.bnd_foring_2('u',t)['E']
        
            vN = VC.bnd_foring_2('v',t)['N']
            vS = VC.bnd_foring_2('v',t)['S']
            vW = VC.bnd_foring_2('v',t)['W'][1:m]
            vE = VC.bnd_foring_2('v',t)['E'][1:m]
                
        gradphiuW = gradphiu[1:m+1,0]
        gradphiuE = gradphiu[1:m+1,-1]
        gradphiuN = gradphiu[0,1:n]
        gradphiuS = gradphiu[-1,1:n]
        
        # North and South boundary
        uNbc = uN + gradphiuN
        uSbc = uS + gradphiuS

        resu1 = np.zeros((m,n-1))
        resu2 = np.zeros((m,n-1))
        resu1[0,:] = (16.0/5)*(uNbc)*(lam/(dy**2))
        resu1[-1,:] = (16.0/5)*(uSbc)*(lam/(dy**2))            
            
        # West and East boundary
        uWbc = uW
        uEbc = uE
        resu2[:,0] = (uWbc)*(lam/(dx**2))
        resu2[:,-1] = (uEbc)*(lam/(dx**2))
        resu = resu1+resu2
        
        resv1 = np.zeros((m-1,n))
        resv2 = np.zeros((m-1,n))
        
        gradphivN = gradphiv[0,1:n+1]
        gradphivS = gradphiv[-1,1:n+1]
        gradphivW = gradphiv[1:m,0]
        gradphivE = gradphiv[1:m,-1]

        # North and South boundary
        vNbc = vN
        vSbc = vS
        resv2[0,:] = vNbc*(lam/(dy**2))
        resv2[-1,:] = vSbc*(lam/(dy**2))

        # West and East boundary
        vWbc = vW + gradphivW
        vEbc = vE + gradphivE
        resv1[:,0] = (16.0/5)*vWbc*(lam/(dx**2))
        resv1[:,-1] = (16.0/5)*vEbc*(lam/(dx**2))
        
        resv = resv1+resv2
        rhs_mstarcd = rhs_mstar + [resu, resv]
        
        return rhs_mstarcd

    # correct (normalise) phi variable (eliminating the unwanted costant from the Pressure Poisson solver)
    def phi_correction(self, phi, phin_cmp, div_mstar, div_mn):
        n = self.n
        m = self.m
        Re = self.Re
        dx = self.dx
        dy = self.dy
        dt = self.dt

        phia = phi.get_value() - phin_cmp[1:m+1,1:n+1]
        phiaW = 1.875*phia[:,0] - 1.25*phia[:,1] + 0.375*phia[:,2]
        div_mstarW = 1.875*div_mstar[:,0] - 1.25*div_mstar[:,1] + 0.375*div_mstar[:,2]
        div_mnW = 1.875*div_mn[:,0] - 1.25*div_mn[:,1] + 0.375*div_mn[:,2]
        
        phiaE = 1.875*phia[:,-1] - 1.25*phia[:,-2] + 0.375*phia[:,-3]
        div_mstarE = 1.875*div_mstar[:,-1] - 1.25*div_mstar[:,-2] + 0.375*div_mstar[:,-3]
        div_mnE = 1.875*div_mn[:,-1] - 1.25*div_mn[:,-2] + 0.375*div_mn[:,-3]
        
        phiaNW = 1.875*phiaW[0] - 1.25*phiaW[1] + 0.375*phiaW[2]
        div_mstarNW = 1.875*div_mstarW[0] - 1.25*div_mstarW[1] + 0.375*div_mstarW[2]
        div_mnNW = 1.875*div_mnW[0] - 1.25*div_mnW[1] + 0.375*div_mnW[2]

        phiaNE = 1.875*phiaE[0] - 1.25*phiaE[1] + 0.375*phiaE[2]
        div_mstarNE = 1.875*div_mstarE[0] - 1.25*div_mstarE[1] + 0.375*div_mstarE[2]
        div_mnNE = 1.875*div_mnE[0] - 1.25*div_mnE[1] + 0.375*div_mnE[2]

        phiaSW = 1.875*phiaW[-1] - 1.25*phiaW[-2] + 0.375*phiaW[-3]
        div_mstarSW = 1.875*div_mstarW[-1] - 1.25*div_mstarW[-2] + 0.375*div_mstarW[-3]
        div_mnSW = 1.875*div_mnW[-1] - 1.25*div_mnW[-2] + 0.375*div_mnW[-3]
        
        phiaSE = 1.875*phiaE[-1] - 1.25*phiaE[-2] + 0.375*phiaE[-3]
        div_mstarSE = 1.875*div_mstarE[-1] - 1.25*div_mstarE[-2] + 0.375*div_mstarE[-3]
        div_mnSE = 1.875*div_mnE[-1] - 1.25*div_mnE[-2] + 0.375*div_mnE[-3]

        cNW = phiaNW - (dt*(div_mstarNW+div_mnNW))/(2*Re)
        cNE = phiaNE - (dt*(div_mstarNE+div_mnNE))/(2*Re)
        cSW = phiaSW - (dt*(div_mstarSW+div_mnSW))/(2*Re)
        cSE = phiaSE - (dt*(div_mstarSE+div_mnSE))/(2*Re)
        c = (cNW+cNE+cSW+cSE)/4.0
        print c, "correction"
        phiacd = phia - c
        return structure2.CentredPotential(phiacd, self.mesh)

    # completing the Gauge variable at time n+1 
    def complete_mstar(self, mstar_int, uvbnd_value, phiacd_cmp):
        # complete m* using phi^(n+1)
        n = self.n
        m = self.m
        dx = self.dx
        dy = self.dy
        dt = self.dt    
        uN, uS, uW, uE = uvbnd_value[0]
        vN, vS, vW, vE = uvbnd_value[1]
        
        m1star_cmp = np.zeros((m+2,n+1))
        m2star_cmp = np.zeros((m+1,n+2))
        m1star_cmp[1:m+1,1:n] = mstar_int.get_uv()[0]
        m2star_cmp[1:m,1:n+1] = mstar_int.get_uv()[1]        
        m1star_cmp[1:m+1,0] = uW
        m1star_cmp[1:m+1,-1] = uE
        m2star_cmp[0,1:n+1] = vN
        m2star_cmp[-1,1:n+1] = vS        
        
        gdphi_cmpu = (phiacd_cmp[:,1:n+2] - phiacd_cmp[:,0:n+1])/dx
        gdphi_cmpuN = 5.0/16*(gdphi_cmpu[0,:] +3*gdphi_cmpu[1,:] - gdphi_cmpu[2,:]+0.2*gdphi_cmpu[3,:])
        gdphi_cmpuS = 5.0/16*(gdphi_cmpu[-1,:] +3*gdphi_cmpu[-2,:] - gdphi_cmpu[-3,:]+0.2*gdphi_cmpu[-4,:])

        # use phi^{n+1} just computed
        m1starN = uN + gdphi_cmpuN
        m1starS = uS + gdphi_cmpuS

        m1star_cmp[0,:] = (16.0/5)*m1starN - 3*m1star_cmp[1,:] + m1star_cmp[2,:] - 0.2*m1star_cmp[3,:]
        m1star_cmp[-1,:] = (16.0/5)*m1starS - 3*m1star_cmp[-2,:] + m1star_cmp[-3,:] - 0.2*m1star_cmp[-4,:]

        gdphi_cmpv = (phiacd_cmp[1:m+2,:] - phiacd_cmp[0:m+1,:])/dy
        gdphi_cmpvW = 5.0/16*(gdphi_cmpv[:,0] +3*gdphi_cmpv[:,1] - gdphi_cmpv[:,2]+0.2*gdphi_cmpv[:,3])
        gdphi_cmpvE = 5.0/16*(gdphi_cmpv[:,-1] +3*gdphi_cmpv[:,-2] - gdphi_cmpv[:,-3]+0.2*gdphi_cmpv[:,-4])
        m2starW = vW + gdphi_cmpvW
        m2starE = vE + gdphi_cmpvE
        m2star_cmp[:,0] = (16.0/5)*m2starW - 3*m2star_cmp[:,1] + m2star_cmp[:,2] - 0.2*m2star_cmp[:,3]
        m2star_cmp[:,-1] = (16.0/5)*m2starE - 3*m2star_cmp[:,-2] + m2star_cmp[:,-3] - 0.2*m2star_cmp[:,-4]

        return structure2.VelocityField(m1star_cmp, m2star_cmp, self.mesh)
        
# class calculates the error norms for the solver
class Error():
    def __init__(self, uv_cmp, uv_exact_bnd, p, p_exact, div_uv, mesh):
        self.mesh = mesh
        self.uv_cmp = uv_cmp
        self.uv_bnd = uv_cmp.get_bnd_uv()
        self.uv_exact_bnd = uv_exact_bnd
        self.p_exact = p_exact
        self.p = p
        self.div_uv = div_uv

    def velocity_error(self):
        n = self.mesh.n
        m = self.mesh.m
        # m: row, n: col
        uebnd = self.uv_bnd[0] - self.uv_exact_bnd.get_uv()[0]
        vebnd = self.uv_bnd[1] - self.uv_exact_bnd.get_uv()[1]
        L1 = []
        L2 = []
        Linf = []
        
        for x in [uebnd, vebnd]:
            xv = np.ravel(x)
            a=sum(abs(xv[:])**2)/(m**2)
            Linfx = abs(xv[:]).max()
            L1x = sum(abs(xv[:]))/(m**2)
            L1.append(L1x)
            L2x = np.sqrt(a)
            L2.append(L2x)
            Linf.append(Linfx)
        ubnderror = {'L1': L1[0], 'L2': L2[0], 'Linf': Linf[0]}
        vbnderror = {'L1': L1[1], 'L2': L2[1], 'Linf': Linf[1]}

        return ubnderror, vbnderror
    
    def pressure_error(self):
        n = self.mesh.n
        m = self.mesh.m
        
        perror = self.p - self.p_exact
        pv = np.ravel(perror.get_value())
        a=sum(abs(pv[:])**2)/(m**2)
        Linfp = abs(pv[:]).max()
        L1p = sum(abs(pv[:]))/(m**2)
        L2p = np.sqrt(a)
        perror_dict = {'L1': L1p, 'L2': L2p, 'Linf': Linfp}

        return perror_dict
