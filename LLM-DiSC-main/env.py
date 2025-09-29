from dynamics import UnicycleDynamics, FirstOrderDynamics
from obs import CircleObstacle, PolygonObstacle, RectangleObstacle
import math

class INFO:
    def __init__(self):
        # agent
        self.dynamics_model = UnicycleDynamics()
        self.begin_state = [[1.5, 0.5, 1.57], [3.5, 0.5, 1.57], [4.5, 0.5, 1.57], [6.5, 0.5, 1.57], [8.5, 0.5, 1.57], [9.5, 0.5, 1.57], [11.5, 0.5, 1.57], [14.5, 0.5, 1.57], [15.5, 0.5, 1.57], [18.5, 0.5, 1.57]]
        # self.dynamics_model = FirstOrderDynamics()

        self.begin_ctrl = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
        self.agent_radius = 0.25
        self.agent_num = len(self.begin_state)
        self.target_pos = [[18, 19], [17, 19], [15, 19], [12, 19], [10, 19], [9, 19], [7.5, 19], [5, 19], [3, 19], [1, 19]]


        # obstacle
        self.obs = []
        self.poly_pos = [[(15.2, 8.3), (16.6, 12.0), (12.8, 12.6), (12.3, 11.9), (12.4, 9.0), (15.2, 8.3)],[(3.3, 6.8), (7.9, 6.3), (8.9, 7.6), (4.2, 8.0), (3.3, 6.8)],[(15.8, 2.5), (13.2, 6.4), (14, 3.2), (15.8, 2.5)],[(7.2, 11.2), (4.2, 17.4), (4.2, 12.0), (7.2, 11.2)],[(0.9, 5.2), (0.2, 3.2), (1.2, 1.8), (2.4, 1.3), (0.9, 5.2)],[(17.5, 6.1), (19.4, 5.3), (19.8, 5.8), (18.1, 6.1), (17.5, 6.1)]]
        for poly_coords in self.poly_pos:
            polygon_obstacle = PolygonObstacle(poly_coords)
            self.obs.append(polygon_obstacle)
        self.obs_num = len(self.poly_pos)

        # system
        self.maxStep = 5000
        self.iteration_time = 10
        self.dt = 0.05

        #cbf zeta*h^eta
        self.zeta = zeta
        self.eta = eta
