import numpy as np

from openmdao.api import ExplicitComponent


class PlanePath2D(ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('departure_time', val=0.0)
        self.add_input('destination_x', val=0.0)
        self.add_input('destination_y', val=0.0)

        self.add_input(name='time',
                       val=np.ones(nn),
                       units='s')

        self.add_input(name='x',
                       val=np.ones(nn),
                       desc='aircraft position x',
                       units='m')

        self.add_input(name='y',
                       val=np.ones(nn),
                       desc='aircraft position y',
                       units='m')

        self.add_input(name='vx',
                       val=np.ones(nn),
                       desc='aircraft velocity magnitude x',
                       units='m/s')

        self.add_input(name='vy',
                       val=np.ones(nn),
                       desc='aircraft velocity magnitude y',
                       units='m/s')

        self.add_output(name='x_dot',
                        val=np.zeros(nn),
                        desc='downrange (longitude) velocity',
                        units='m/s')

        self.add_output(name='y_dot',
                        val=np.zeros(nn),
                        desc='crossrange (latitude) velocity',
                        units='m/s')

        self.add_output(name='distance_to_destination',
                        val=np.zeros(nn),
                        desc='S')

        self.add_output(name='departure_hold',
                        val=0.0)

        ar = np.arange(nn)

        self.declare_partials('x_dot', 'vx', rows=ar, cols=ar)
        self.declare_partials('y_dot', 'vy', rows=ar, cols=ar)
        self.declare_partials('departure_hold', ['vx', 'vy'])
        self.declare_partials('distance_to_destination', ['x', 'y', 'time'], rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        x = inputs['x']
        y = inputs['y']
        t = inputs['time']
        ts = inputs['departure_time']

        dx = inputs['destination_x']
        dy = inputs['destination_y']

        vx = inputs['vx']
        vy = inputs['vy']

        outputs['x_dot'] = vx
        outputs['y_dot'] = vy

        dist = np.sqrt((x - dx)**2 + (y - dy)**2)
        mask = (np.tanh(t - ts) + 1.0) / 2.0

        outputs['departure_hold'] = np.sum((vx**2 + vy**2) * (1 - mask))

        outputs['distance_to_destination'] = dist * mask


    def compute_partials(self, inputs, partials):
        x = inputs['x']
        y = inputs['y']
        t = inputs['time']
        ts = inputs['departure_time']

        dx = inputs['destination_x']
        dy = inputs['destination_y']

        vx = inputs['vx']
        vy = inputs['vy']

        partials['x_dot', 'vx'] = 1.0
        partials['y_dot', 'vy'] = 1.0

        dist = np.sqrt((x - dx)**2 + (y - dy)**2)
        dist[np.where(dist < 1)] = 1.0
        mask = (np.tanh(t - ts) + 1.0) / 2.0
        dt = -0.5*np.tanh(t - ts)**2 + 0.5

        partials['departure_hold', 'vx'] = (1 - mask) * (2 * vx)
        partials['departure_hold', 'vy'] = (1 - mask) * (2 * vy)
        #partials['departure_hold', 'time'] = (1 - mask) * (2 * vx)

        partials['distance_to_destination', 'x'] = (x - dx)/dist * mask
        partials['distance_to_destination', 'y'] = (y - dy)/dist * mask
        partials['distance_to_destination', 'time'] = dist * dt



if __name__ == '__main__':
    from openmdao.api import Problem, Group

    n = 20

    p = Problem()
    p.model = Group()
    p.model.add_subsystem('test', PlanePath2D(num_nodes = n), promotes=['*'])
    p.setup()

    p['departure_time'] = 3.0
    p['destination_x'] = 100.0
    p['destination_y'] = 200.0

    p['x'] = np.random.uniform(0, 100, n)
    p['y'] = np.random.uniform(0, 200, n)
    p['time'] = np.linspace(0, 10, n)

    p['vx'] = np.random.uniform(0.01, 20, size=n)
    p['vy'] = np.random.uniform(0.01, 20, size=n)


    p.run_model()
    p.check_partials(compact_print=True)

