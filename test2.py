# -*- coding: utf-8 -*-
"""
Created on Mon Dec 29 22:26:38 2014

@author: acerlinux
"""

import sys
sys.path.append("/home/acerlinux/Documents/share_folder")
from mpl_toolkits.mplot3d import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import structure2
import solvers2

#gridsize = [60,60]
gridsize = [30,30]
#gridsize = [4,4]
m, n = gridsize
#spatial_domain = [[0,1],[0,1]]
spatial_domain = [[-np.pi/4.0, np.pi/4.0],[-np.pi/4.0, np.pi/4.0]]
time_domain = [0,1]
#N_iterations = 10
CFL = 0.5
Re = 1.0

mesh = structure2.mesh(gridsize,spatial_domain,time_domain,CFL)
print mesh.dx, "dx"
print mesh.dt, "dt"
Xubnd = mesh.ubndmg("x")
Yubnd = mesh.ubndmg("y")
Xvbnd = mesh.vbndmg("x")
Yvbnd = mesh.vbndmg("y")
XPint = mesh.pintmg("x")
YPint = mesh.pintmg("y")
tend = mesh.tdomain[1]

#ic_uv_init = structure2.InitialCondition(mesh).zero_uv()
# uses Taylor flow problem for testing
ic_uv_init = structure2.InitialCondition(mesh).Taylor_uv()

# use Gauge method
Gauge = solvers2.Gauge_method(Re, mesh)
# initial set up
init_setup = Gauge.setup(ic_uv_init, "Taylor")
# iterative solve process
uvf_cmp, pf = Gauge.iterative_solver("Taylor", mesh.Tn, init_setup)
# comparison and error analysis
uv_exact_bnd, p_exact = structure2.Exact_solutions(mesh, Re, mesh.Tn).Exact_solutions("Taylor")
div_uvf = uvf_cmp.divergence()

Error = solvers2.Error(uvf_cmp, uv_exact_bnd, pf, p_exact, div_uvf, mesh)
Velocity_error = Error.velocity_error()
Pressure_error = Error.pressure_error()
print "U velocity error is %s " % Velocity_error[0]
print "V velocity error is %s " % Velocity_error[1]
print "Pressure error is %s " % Pressure_error

uf_bnd = uvf_cmp.get_bnd_uv()[0]
vf_bnd = uvf_cmp.get_bnd_uv()[1]

fig1 = plt.figure(figsize=(6,6), dpi=100)
ax1 = fig1.gca(projection='3d')
#ax1.plot_wireframe(Xuint,Yuint,uf_int)
ax1.plot_surface(Xubnd,Yubnd,uf_bnd,rstride=1, cstride=1, cmap=cm.jet, linewidth=0.2)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('Plot of Numerical U Velocity at Time = '+str(tend))
#plt.savefig('Gauge_Taylor_U_exact_grid_'+str(n)+'.jpg', bbox_inches='tight')
#plt.savefig('Gauge_unf1_U_exact_grid_'+str(n)+'.pdf', bbox_inches='tight')

fig2 = plt.figure(figsize=(6,6), dpi=100)
ax2 = fig2.gca(projection='3d')
#ax2.plot_wireframe(Xvbnd,Yvbnd,vf_bnd)
ax2.plot_surface(Xvbnd,Yvbnd,vf_bnd,rstride=1, cstride=1, cmap=cm.jet, linewidth=0.2)
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_title('Plot of Numerical V Velocity at Time = '+str(tend))
#plt.savefig('Gauge_Taylor_V_exact_tend_grid_'+str(n)+'.jpg', bbox_inches='tight')
#plt.savefig('Gauge_Taylor_V_exact_tend_grid_'+str(n)+'.pdf', bbox_inches='tight')

fig3 = plt.figure(figsize=(6,6), dpi=100)
ax3 = fig3.gca(projection='3d')
ax3.plot_surface(XPint, YPint, pf.get_value() ,rstride=1, cstride=1, cmap=cm.jet, linewidth=0.2)
ax3.set_xlabel('x')
ax3.set_ylabel('y')
ax3.set_title('Plot of Numerical Pressure at Time = '+str(tend))
#plt.savefig('Gauge_Taylor_pf_tend_grid_'+str(n)+'.jpg', bbox_inches='tight')
#plt.savefig('Gauge_Taylor_pf_tend_grid_'+str(n)+'.pdf', bbox_inches='tight')

fig4 = plt.figure(figsize=(6,6), dpi=100)
ax4 = fig4.gca(projection='3d')
#ax4.plot_wireframe(Xubnd,Yubnd,U_exact)
ax4.plot_surface(Xubnd,Yubnd,uv_exact_bnd.get_uv()[0],rstride=1, cstride=1, cmap=cm.jet, linewidth=0.2)
ax4.set_xlabel('x')
ax4.set_ylabel('y')
ax4.set_title('Plot of Analytical U Velocity at Time = '+str(tend))
#plt.savefig('Gauge_Taylor_U_exact_tend_grid_'+str(n)+'.jpg', bbox_inches='tight')
#plt.savefig('Gauge_Taylor_U_exact_tend_grid_'+str(n)+'.pdf', bbox_inches='tight')

fig5 = plt.figure(figsize=(6,6), dpi=100)
ax5 = fig5.gca(projection='3d')
#ax5.plot_wireframe(Xvbnd,Yvbnd,V_exact)
ax5.plot_surface(Xvbnd,Yvbnd,uv_exact_bnd.get_uv()[1],rstride=1, cstride=1, cmap=cm.jet, linewidth=0.2)
ax5.set_xlabel('x')
ax5.set_ylabel('y')
ax5.set_title('Plot of Analytical V Velocity at Time = '+str(tend))
#plt.savefig('Gauge_Taylor_V_exact_tend_grid_'+str(n)+'.jpg', bbox_inches='tight')
#plt.savefig('Gauge_Taylor_V_exact_tend_grid_'+str(n)+'.pdf', bbox_inches='tight')

fig6 = plt.figure(figsize=(6,6), dpi=100)
ax6 = fig6.gca(projection='3d')
#ax6.plot_wireframe(XPint,YPint,P_exact)
ax6.plot_surface(XPint,YPint,p_exact.get_value(),rstride=1, cstride=1, cmap=cm.jet, linewidth=0.2)
ax6.set_xlabel('x')
ax6.set_ylabel('y')
ax6.set_title('Plot of Analytical Pressure at Time = '+str(tend))
#plt.savefig('Gauge_Taylor_P_exact_tend_grid_'+str(n)+'.jpg', bbox_inches='tight')
#plt.savefig('Gauge_Tyalor_P_exact_tend_grid_'+str(n)+'.pdf', bbox_inches='tight')
##
fig7 = plt.figure(figsize=(6,6), dpi=100)
ax7 = fig7.gca(projection='3d')
#ax6.plot_wireframe(XPint,YPint,P_exact)
ax7.plot_surface(XPint,YPint,pf.get_value()-p_exact.get_value(),rstride=1, cstride=1, cmap=cm.jet, linewidth=0.2)
ax7.set_xlabel('x')
ax7.set_ylabel('y')
ax7.set_title('Plot of Pressure error at Time = '+str(tend))
#plt.savefig('Gauge_Taylor_P_exact_tend_grid_'+str(n)+'.jpg', bbox_inches='tight')
#plt.savefig('Gauge_Tyalor_P_exact_tend_grid_'+str(n)+'.pdf', bbox_inches='tight')

plt.show()
