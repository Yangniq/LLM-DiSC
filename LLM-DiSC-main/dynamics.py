import numpy as np


class DynamicsModel:
    def apply_dynamics(self, x, u):
        """
        Apply the dynamic model to return the new state update
        x: Current state
        u: control input
        """
        raise NotImplementedError("apply_dynamics needs to be implemented in the subclass")

    def update_state(self, x, u, dt):
        """
        Update the status using the Euler method
        """
        dx = self.apply_dynamics(x, u)
        return x + dx * dt

    def control_input_mapping(self, x):
        """
        return the new state update
        """
        raise NotImplementedError("apply_dynamics needs to be implemented in the subclass")

    def get_derivative_h(self, x, delta_x, delta_y):
        """
        According to the dynamic model, return the partial derivative of h with respect to x
        """
        raise NotImplementedError("apply_dynamics needs to be implemented in the subclass")



class FirstOrderDynamics(DynamicsModel):
    def apply_dynamics(self, x, u):
        """
        dx/dt = u
        """
        return np.array(u)

    def control_input_mapping(self, x):
        """
        x: [px, py]
        """
        g_matrix = np.eye(len(x))
        f_matrix = np.zeros_like(x)

        return f_matrix, g_matrix

    def get_derivative_h(self, x, delta_x, delta_y):
        return np.array([2*delta_x, 2*delta_y])



class UnicycleDynamics(DynamicsModel):
    def apply_dynamics(self, x, u):
        """
        x: [px, py, theta]
        u: [v, omega]
        """
        px, py, theta = x
        v, omega = u

        #
        dx = v * np.cos(theta)
        dy = v * np.sin(theta)
        dtheta = omega

        return np.array([dx, dy, dtheta])

    def control_input_mapping(self, x):

        theta = x[2]
        g_matrix = np.array([
            [np.cos(theta), 0],
            [np.sin(theta), 0],
            [0, 1]
        ])
        f_matrix = np.zeros_like(x)

        return f_matrix, g_matrix

    def get_derivative_h(self, x, delta_x, delta_y):
        return np.array([2*delta_x, 2*delta_y, 0])
