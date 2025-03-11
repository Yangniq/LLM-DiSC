
import numpy as np

def get_control(index, all_state, all_ctrl, target_pos, obs_pos, agent_R, state_info):
    def wrap_angle(angle):
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
    
    # Extract current state of the agent
    state = all_state[index]
    target = target_pos
    px, py, theta = state
    v_max = 1.0  # Max linear velocity
    
    # Attractive force towards the target
    target_vec = np.array([target[0] - px, target[1] - py])
    distance_to_target = np.linalg.norm(target_vec)
    angle_to_target = np.arctan2(target_vec[1], target_vec[0])
    desired_theta = wrap_angle(angle_to_target)
    
    # Proportional gain for control
    Kp_v = 1.0 
    Kp_omega = 2.0 
    
    v = Kp_v * distance_to_target
    omega = Kp_omega * wrap_angle(desired_theta - theta)
    
    safety_margin = 1.5 * agent_R
    
    def point_to_line_distance(point, line_start, line_end):
        line_vec = np.array(line_end) - np.array(line_start)
        point_vec = np.array(point) - np.array(line_start)
        line_len = np.linalg.norm(line_vec)
        line_unitvec = line_vec / line_len
        point_vec_scaled = point_vec / line_len
        t = np.dot(line_unitvec, point_vec_scaled)
        t = np.clip(t, 0.0, 1.0)
        nearest = np.array(line_start) + t * line_vec
        dist = np.linalg.norm(np.array(point) - nearest)
        return dist, nearest
    
    # Helper function to adjust control according to CBF constraints
    def apply_cbf_constraints(px, py, theta, v, omega, ox, oy, R, is_obstacle=True):
        h = (px - ox)**2 + (py - oy)**2 - R**2
        phi = 2 * ((px - ox) * v * np.cos(theta) + (py - oy) * v * np.sin(theta)) + 2 * h
        if phi < 0:
            # Calculate a safe direction away from the obstacle or other agent
            safe_direction = np.array([px - ox, py - oy])
            safe_direction /= np.linalg.norm(safe_direction)
            v_target = 0.5 * v_max  # Reduce target velocity
            omega_target = np.arctan2(safe_direction[1], safe_direction[0]) - theta
            omega_target = wrap_angle(omega_target)
            return v_target, omega_target
        else:
            return v, omega

    # Obstacle avoidance and CBF application
    for obs in obs_pos:
        for i in range(len(obs) - 1):
            dist_to_obs, closest_point = point_to_line_distance((px, py), obs[i], obs[i+1])
            if dist_to_obs < safety_margin:
                v, omega = apply_cbf_constraints(px, py, theta, v, omega, closest_point[0], closest_point[1], agent_R, is_obstacle=True)
    
    # Avoid other agents
    for i, other_state in enumerate(all_state):
        if i != index:
            dist_to_agent = np.linalg.norm(np.array([px - other_state[0], py - other_state[1]]))
            if dist_to_agent < 2 * agent_R:
                v, omega = apply_cbf_constraints(px, py, theta, v, omega, other_state[0], other_state[1], 2 * agent_R, is_obstacle=False)
    
    # Ensure the control inputs are within reasonable limits
    v = np.clip(v, -v_max, v_max)
    omega = np.clip(omega, -np.pi/4, np.pi/4)

    control_input = [v, omega]
    return control_input
