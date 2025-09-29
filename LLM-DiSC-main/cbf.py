import math
import numpy as np
from cvxopt import matrix, solvers
from obs import RectangleObstacle
from obs import CircleObstacle
from obs import PolygonObstacle
from dynamics import DynamicsModel


# CBF controller
class CBFController:
    def __init__(self, dynamics_model, agentR, zeta, eta):
        self.dynamics_model = dynamics_model
        self.agentR = agentR
        self.zeta = zeta
        self.eta = eta
        self.safe_threshold = 0.1



    def get_alpha_h(self, h_value):
        return self.zeta * (h_value ** self.eta)
        # return self.zeta * (1-math.exp(-self.eta*h_value))


    def get_cbf(self, delta_x, delta_y, safe_distance):
        h = delta_x ** 2 + delta_y ** 2 - safe_distance ** 2
        return h

    def compute_lie_derivatives(self, x, derivative_h):
        dynamics_f, dynamics_g = self.dynamics_model.control_input_mapping(x)
        Lfh = np.dot(derivative_h, dynamics_f)
        Lgh = np.dot(derivative_h, dynamics_g)
        return Lfh, Lgh

    def calculate_cbf_constraint(self, derivative_h, derivative_x, h_value):

        alpha_h = self.get_alpha_h(h_value)
        value = np.dot(derivative_h, derivative_x) + alpha_h
        return value

    def cbf_constraint(self, ctrlU, index, allCtrl, allX, obs, agentNum, obsNum):
        """
        Calculate the constraint values of the CBF inequality
        """
        x = allX[index]
        pos = x[:2]
        G = []
        H = []

        # constraint about obstacles
        h_obs_all = []
        constraint_obs_all = []
        for j in range(obsNum):
            dis_x, dis_y, dis = obs[j].distance_to_agent(pos)
            if isinstance(obs[j], CircleObstacle):
                safe_distance = obs[j].radius + self.agentR
            else:
                safe_distance = self.agentR
            h_obs = self.get_cbf(dis_x, dis_y, safe_distance)
            h_obs_safe = h_obs-0.21
            h_obs_all.append(h_obs_safe)
            alpha_h = self.get_alpha_h(h_obs)
            derivative_h = self.dynamics_model.get_derivative_h(x, dis_x, dis_y)
            Lfh, Lgh = self.compute_lie_derivatives(x, derivative_h)
            g = -Lgh
            h = Lfh + alpha_h
            G.append(g)
            H.append(h)
            # Calculate the CBF constraint values of obstacles in combination with the dynamic model
            constraint_obs = Lfh + np.dot(Lgh, ctrlU) + alpha_h
            constraint_obs_all.append(constraint_obs)

        # constraint about robots
        constraint_agent_all = []
        h_agent_all = []

        for i in range(agentNum):
            if i != index:
                agent_p = allX[i][:2]
                dp_a = pos - agent_p
                h_agent = self.get_cbf(dp_a[0], dp_a[1], 2 * self.agentR + 0.03) # agent之间的安全距离增加阈值
                h_agent_safe = h_agent-0.41
                h_agent_all.append(h_agent_safe)
                alpha_h = self.get_alpha_h(h_agent)

                derivative_h = self.dynamics_model.get_derivative_h(x, dp_a[0], dp_a[1])
                Lfh, Lgh = self.compute_lie_derivatives(x, derivative_h)
                Lfh_i, Lgh_i = self.compute_lie_derivatives(allX[i], derivative_h)
                constraint_agent = Lfh + np.dot(Lgh, ctrlU) - Lfh_i - np.dot(Lgh_i, allCtrl[i]) + alpha_h
                constraint_agent_all.append(constraint_agent)


                g = -Lgh
                h = Lfh + alpha_h - Lfh_i - np.dot(Lgh_i, allCtrl[i])
                G.append(g)
                H.append(h)

        return constraint_obs_all, constraint_agent_all, G, H, h_obs_all, h_agent_all

    def correct_U(self, ctrlU, G, H):
        """
        Use QP to optimize and correct the control input U
        """
        p = matrix(np.eye(2), tc='d')
        q = -matrix(ctrlU, tc='d')
        G = np.array(G)
        G = matrix(G, tc='d')
        H = matrix(H)

        try:
            solvers.options['show_progress'] = False
            sol = solvers.qp(p, q, G, H)
            newU = np.array(sol['x']).flatten()
            return {"status": "success", "data": newU}
        except Exception as e:
            return {"status": "error", "message": str(e)}
