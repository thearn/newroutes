import numpy as np

from openmdao.api import ExplicitComponent, Group
from plane import PlanePath2D
from dymos import ODEOptions

n_traj = 2

class PlaneODE2D(Group):
    ode_options = ODEOptions()

    ode_options.declare_time(units='s', targets = ['p%d.time' % i for i in range(n_traj)])

    # dynamic trajectories
    for i in range(n_traj):
        ode_options.declare_state(name='p%dx' % i, rate_source='p%d.x_dot' % i, 
                                  targets=['p%d.x' % i], units='m')
        ode_options.declare_state(name='p%dy' % i, rate_source='p%d.y_dot' % i, 
                                  targets=['p%d.y' % i], units='m')

        ode_options.declare_parameter(name='p%dvx' % i, targets = 'p%d.vx' % i, 
                                      units='m/s')
        ode_options.declare_parameter(name='p%dvy' % i, targets = 'p%d.vy' % i, 
                                      units='m/s')

    def initialize(self):   
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        for i in range(n_traj):
            self.add_subsystem(name='p%d' % i,
                           subsys=PlanePath2D(num_nodes=nn))