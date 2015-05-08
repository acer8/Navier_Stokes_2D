# -*- coding: utf-8 -*-
"""
Created on Mon Dec 29 22:26:38 2014

@author: acerlinux
"""

import sys
from mpl_toolkits.mplot3d import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import structure2
import solvers2

def get_inputs():
	xl, xr = raw_input('Enter the end points of the spatial domain (e.g 0,1): ').split(',')
	try:
		xl = float(xl)
		xr = float(xr)
	except ValueError:
		print 'End points must be float or integers, try again'
		xl, xr = raw_input('Enter the end points of the spatial domain (e.g 0,1): ').split(',')
		xl = float(xl)
		xr = float(xr)

	t0, tf = raw_input('Enter the start and end time (e.g 0,1): ').split(',')
	try:
		t0 = float(t0)
		tf = float(tf)
	except ValueError:
		print 'Time must be float or integers, try again'
		xl, xr = raw_input('Enter the start and end time (e.g 0,1): ').split(',')
		t0 = float(t0)
		tf = float(tf)
	try:
		gridsize = int(raw_input('Enter the size of the spatial grid (e.g 30): '))
	except ValueError:
		print 'Grid size must be integers, try again'
		gridsize = int(raw_input('Enter the size of the spatial grid (e.g 30): '))
	
	method_dict = {1:'Gauge', 2:'Alg1', 3:'Alg2', 4:'Alg3'}
	try:
		method_index = int(raw_input('Choose the numerical algorithm from below [1:Gauge, 2:Alg1, 3:Alg2, 4:Alg3]: '))
	except ValueError:
		method_index = int(raw_input('index of the method must be integers, try again: '))
	method = method_dict[method_index]

	test_problem_dict = {1:'Taylor', 2:'periodic_forcing_1', 
				3:'periodic_forcing_2', 4:'periodic_forcing_3', 5:'driven_cavity'}
	try:				
		test_problem_index = int(raw_input('Choose the test problem from below [1:Taylor, 2:periodic_forcing_1, 3:periodic_forcing_2, 4:periodic_forcing_3, 5:driven_cavity]: '))
	except 	ValueError:
		test_problem_index = int(raw_input('index of the test problem must be integers, try again: '))
	test_problem_name = test_problem_dict[test_problem_index]
	plot_option = raw_input('plot result optional (Y/N): ')
	if 'Y' in plot_option:
		plot_option = True
	elif 'y' in plot_option:
		plot_option = True
	else:
		plot_option = False

	return xl, xr, t0, tf, gridsize, method, test_problem_name, plot_option

def run_Navier_Stokes_solver(xl, xr, t0, tf, gridsize, method, test_problem_name, plot_option, CFL=0.5, Re=1.0):
	grid_size_domain = [gridsize, gridsize]
	m, n = grid_size_domain
	spatial_domain = [[xl,xr],[xl,xr]]
	time_domain = [t0,tf]
	print 'start'
	mesh = structure2.mesh(grid_size_domain,spatial_domain,time_domain,CFL,Re)
	print mesh.dx, "dx"
	print mesh.dt, "dt"
	Xubnd = mesh.ubndmg("x")
	Yubnd = mesh.ubndmg("y")
	Xvbnd = mesh.vbndmg("x")
	Yvbnd = mesh.vbndmg("y")
	XPint = mesh.pintmg("x")
	YPint = mesh.pintmg("y")
	tend = mesh.tdomain[1]
	
	if method == 'Gauge':
		ic_uv_init = structure2.InitialCondition(mesh).select_initial_conditions(test_problem_name)[0]
		# use Gauge method
		Gauge = solvers2.Gauge_method(Re, mesh)
		# initial set up
		init_setup = Gauge.setup(ic_uv_init, test_problem_name)
		# iterative solve process
		uvf_cmp, pf = Gauge.iterative_solver(test_problem_name, mesh.Tn, init_setup)

	# comparison and error analysis
	if test_problem_name == 'driven_cavity':
		# no analytical solutions available
		pass
	else:
		uv_exact_bnd, p_exact = structure2.Exact_solutions(mesh, Re, mesh.Tn).Exact_solutions(test_problem_name)
		div_uvf = uvf_cmp.divergence()
		
		Error = solvers2.Error(uvf_cmp, uv_exact_bnd, pf, p_exact, div_uvf, mesh)
		Velocity_error = Error.velocity_error()
		Pressure_error = Error.pressure_error()
		print "U velocity error is %s " % Velocity_error[0]
		print "V velocity error is %s " % Velocity_error[1]
		print "Pressure error is %s " % Pressure_error
	
	if plot_option == False:
		return None
	else:
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
		
		if test_problem_name == 'driven_cavity':
			# no analytical solutions availabe
			pass
		else:
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

if __name__ == "__main__":
	xl, xr, t0, tf, gridsize, method, test_problem_name, plot_option = get_inputs()
	run_Navier_Stokes_solver(xl, xr, t0, tf, gridsize, method, test_problem_name, plot_option)

	

