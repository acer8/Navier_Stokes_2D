# -*- coding: utf-8 -*-
from __future__ import division
import sys
from mpl_toolkits.mplot3d import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import structure2
import solvers2

def get_inputs():
	test_problem_dict = {1:'Taylor', 2:'periodic_forcing_1', 
				3:'periodic_forcing_2', 4:'periodic_forcing_3', 5:'driven_cavity'}
	test_problem_index = raw_input('Choose the test problem from below [1:Taylor, 2:periodic_forcing_1, 3:periodic_forcing_2, 4:periodic_forcing_3, 5:driven_cavity]: ')
	while test_problem_index == '':
		# take default
		test_problem_index = 1
	try:				
		test_problem_index = int(test_problem_index)
	except 	ValueError:
		test_problem_index = int(raw_input('index of the test problem must be integers, try again: '))
	test_problem_name = test_problem_dict[test_problem_index]

	method_dict = {1:'Gauge', 2:'Alg1', 3:'Alg2', 4:'Alg3'}
	method_index = raw_input('Choose the numerical algorithm from below [1:Gauge, 2:Alg1, 3:Alg2, 4:Alg3]: ')
	while method_index == '':
		# take default
		method_index = 1
	try:
		method_index = int(method_index)
	except ValueError:
		method_index = int(raw_input('index of the method must be integers, try again: '))
	method = method_dict[method_index]

	space_input = raw_input('Enter the end points of the spatial domain (e.g 0,1): ')
	while space_input == '':
		# take default (different for each problem)
		if test_problem_name == 'Taylor':
			space_input = [-np.pi/4.0, np.pi/4.0]
		elif test_problem_name == 'periodic_forcing_1':
			space_input = [-1,1]
		else:
			space_input = [0,1]
	try:
		xl, xr = space_input.split(',')
		xl = float(xl)
		xr = float(xr)
	except:
		if isinstance(space_input, list) == True:
			xl = space_input[0]
			xr = space_input[1]
		else:
			xl, xr = raw_input('End points must be float or integers, separate them by a comma (,) now try again: ').split(',')
			xl = float(xl)
			xr = float(xr)
	
	time_input = raw_input('Enter the start and end time (e.g 0,1): ')
	while time_input == '':
		# take default
		time_input = '0,1'
	try:
		t0, tf = time_input.split(',')
		t0 = float(t0)
		tf = float(tf)
	except ValueError:
		time_input = raw_input('Time must be float or integers, separate start and end time by a comma (,) now try again: ').split(',')
		t0 = float(time_input[0])
		tf = float(time_input[1])
	
	gridsize = raw_input('Enter the size of the spatial grid (e.g 30): ')
	while gridsize == '':
		# take default value
		gridsize = 30
	try:
		gridsize = int(gridsize)
	except ValueError:
		gridsize = int(raw_input('Grid size must be integers, try again: '))

	plot_option = raw_input('plot result optional (Y/N): ')
	if 'Y' in plot_option:
		plot_option = True
	elif 'y' in plot_option:
		plot_option = True
	elif plot_option == '':
		# take default
		plot_option = False
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
	print spatial_domain, 'spatial domain'
	print time_domain, 'time domain'
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
		uvf_cmp, pf, gradp = Gauge.iterative_solver(test_problem_name, mesh.Tn, init_setup)
#		print pf.get_value()

	elif method == 'Alg1':
		ic_init = structure2.InitialCondition(mesh).select_initial_conditions(test_problem_name)
		# use Gauge method
		Alg1 = solvers2.Alg1_method(Re, mesh)
		# initial set up
		init_setup = Alg1.setup(ic_init, test_problem_name)
		# iterative solve process
		uvf_cmp, pf, gradp = Alg1.iterative_solver(test_problem_name, mesh.Tn, init_setup)
#		print uvf_cmp.get_uv(), pf.get_value()
	
	elif method == 'Alg2':
		ic_uv_init = structure2.InitialCondition(mesh).select_initial_conditions(test_problem_name)[0]
		# use Alg2 method
		Alg2 = solvers2.Alg2_method(Re, mesh)
		# initial set up
		init_setup = Alg2.setup(ic_uv_init, test_problem_name)
		# iterative solve process
		uvf_cmp, pf, gradp = Alg2.iterative_solver(test_problem_name, mesh.Tn, init_setup)
#		print pf.get_value()
	
	# comparison and error analysis
	if test_problem_name == 'driven_cavity':
		# no analytical solutions available
		pass
	else:
		uv_exact_bnd, p_exact, gradp_exact = structure2.Exact_solutions(mesh, Re, mesh.Tn).Exact_solutions(test_problem_name)
		div_uvf = uvf_cmp.divergence()
		print np.sum(p_exact.get_value()), 'integral of exact pressure'
#		print p_exact.get_value(), 'exact pressure'
#		pc = np.sum(p_exact.get_value())
#		area_domain = (spatial_domain[0][1] - spatial_domain[0][0])**2
#		d = pc/area_domain
#		print area_domain, 'area of domain'
#		print d, 'constant'
#		p_exact = p_exact - pc
#		print p_exact.get_value(), 'exact pressure normalised'
#		print np.sum(p_exact.get_value()), 'integral of exact pressure normalised' 	
		Error = solvers2.Error(uvf_cmp, uv_exact_bnd, pf, p_exact, gradp, gradp_exact, div_uvf, mesh)
		Velocity_error = Error.velocity_error()
		Pressure_error = Error.pressure_error()
		gradpu_error, gradpv_error, avg_gradp_error = Error.pressure_gradient_error()
		print "U velocity error is %s " % Velocity_error[0]
		print "V velocity error is %s " % Velocity_error[1]
		print "Pressure error is %s " % Pressure_error
		print "gradient P u error is %s " % gradpu_error
		print "gradient P v error is %s " % gradpv_error
		print "average gradient Pressure error is %s " % avg_gradp_error
	
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

	

