import numpy as np

from airspace_phase import PlaneODE2D, n_traj
from openmdao.api import Problem, Group, pyOptSparseDriver, DirectSolver
from dymos import Phase
from itertools import combinations

import pickle

np.random.seed(2)

p = Problem(model=Group())

p.driver = pyOptSparseDriver()
p.driver.options['optimizer'] = 'SNOPT'
p.driver.options['dynamic_simul_derivs'] = True
#p.driver.set_simul_deriv_color('coloring.json')
#p.driver.opt_settings['Start'] = 'Cold'
p.driver.opt_settings['iSumm'] = 6


phase = Phase(transcription='gauss-lobatto',
              ode_class=PlaneODE2D,
              num_segments=20,

              transcription_order=3,
              compressed=True)

p.model.add_subsystem('phase0', phase)

max_time = 6500.0

phase.set_time_options(initial_bounds=(0, 0), duration_bounds=(1, max_time))


locations = []
thetas = []

rand_st = np.random.uniform(0, 2*np.pi)
thetas = np.linspace(rand_st, rand_st + 2*np.pi, 2*n_traj + 1)[:-1]
np.random.shuffle(thetas)

r = 4000.0
for i in range(n_traj):
    # trajectories random start/end locations in circle of radius 4000 around this point
    center_x = 0
    center_y = 0

    t_start = np.random.uniform(1, max_time/4.0)
    t_end = np.random.uniform(1.2*max_time/2.0, max_time)


    print("schedule:", t_start, t_end)

    theta = thetas[2*i]
    theta2 = thetas[2*i + 1]


    start_x = center_x + r*np.cos(theta)
    start_y = center_y + r*np.sin(theta)

    end_x = center_x + r*np.cos(theta2)
    end_y = center_y + r*np.sin(theta2)

    locations.append([start_x, end_x, start_y, end_y])

    phase.set_state_options('p%dx' % i,
                            scaler=0.01, defect_scaler=0.1)
    phase.set_state_options('p%dy' % i,
                            scaler=0.01, defect_scaler=0.1)

    phase.add_boundary_constraint('p%dx' % i, loc='initial', equals=start_x)
    phase.add_boundary_constraint('p%dy' % i, loc='initial', equals=start_y)
    phase.add_boundary_constraint('p%dx' % i, loc='final', equals=end_x)
    phase.add_boundary_constraint('p%dy' % i, loc='final', equals=end_y)


    phase.add_control('p%dvx' % i, rate_continuity=False, units='m/s', 
                      opt=True, upper=10, lower=-10.0, scaler=200.0, adder=-10)
    phase.add_control('p%dvy' % i, rate_continuity=False, units='m/s', 
                      opt=True, upper=10, lower=-10.0, scaler=200.0, adder=-10)

phase.add_objective('time', loc='final', scaler=1.0) #71000

p.setup()

phase = p.model.phase0


p['phase0.t_initial'] = 0.0
p['phase0.t_duration'] = max_time

for i in range(n_traj):
    start_x, end_x, start_y, end_y = locations[i]

    p['phase0.states:p%dx' % i] = phase.interpolate(ys=[start_x, end_x], nodes='state_input')
    p['phase0.states:p%dy' % i] = phase.interpolate(ys=[start_y, end_y], nodes='state_input')

p.run_driver()

exp_out = phase.simulate(times='all', record=False)



