
import numpy as np

def get_control(index, all_state, all_ctrl, target_pos, obs_pos, agent_R, state_info):
    params = {
        'k_v': 1.0,  # Proportional gain for linear velocity
        'k_omega': 1.5,  # Proportional gain for angular velocity
        'obs_r': 1.0,  # Radius of influence for obstacles
        'agent_r': agent_R * 2,  # Safe distance from other agents
        'v_max': 1.0,  # Maximum linear velocity
        'omega_max': np.pi,  # Maximum angular velocity
    }

    def wrap_angle(theta):
        return np.arctan2(np.sin(theta), np.cos(theta))

    def compute_repulsive_force(agent_pos, obstacle_vertices, influence_radius):
        force = np.zeros(2)
        for i in range(len(obstacle_vertices) - 1):  # looping through each edge of the polygon
            obs_edge_start = np.array(obstacle_vertices[i])
            obs_edge_end = np.array(obstacle_vertices[i + 1])
            edge_vect = obs_edge_end - obs_edge_start
            agent_vect = agent_pos - obs_edge_start
            edge_length = np.linalg.norm(edge_vect)
            if edge_length > 0:
                edge_unit_vect = edge_vect / edge_length
                proj_length = np.dot(agent_vect, edge_unit_vect)
                if 0 <= proj_length <= edge_length:
                    closest_point = obs_edge_start + proj_length * edge_unit_vect
                else:
                    closest_point = obs_edge_start if proj_length < 0 else obs_edge_end
                dist = np.linalg.norm(agent_pos - closest_point)
                if dist < influence_radius:
                    repulsion = (influence_radius - dist) / influence_radius * (agent_pos - closest_point) / dist
                    force += repulsion
        return force

    current_pos = np.array(all_state[index][:2])
    current_theta = all_state[index][2]

    # Calculating attractive force towards target
    target_vector = np.array(target_pos) - current_pos
    theta_target = np.arctan2(target_vector[1], target_vector[0])
    dist_to_target = np.linalg.norm(target_vector)
    
    if dist_to_target < 0.1:  # Close enough to the target
        return [0, 0]

    # Computing repulsive forces from obstacles
    repulsive_force = np.zeros(2)
    for obstacle in obs_pos:
        repulsive_force += compute_repulsive_force(current_pos, obstacle, params['obs_r'])

    # Computing repulsive forces from other agents
    for i, state in enumerate(all_state):
        if i != index:
            other_agent_pos = np.array(state[:2])
            dist_to_agent = np.linalg.norm(current_pos - other_agent_pos)
            if dist_to_agent < params['agent_r']:
                repulsion = (params['agent_r'] - dist_to_agent) / params['agent_r'] * (current_pos - other_agent_pos) / dist_to_agent
                repulsive_force += repulsion

    total_force = target_vector / dist_to_target + repulsive_force
    theta_desired = np.arctan2(total_force[1], total_force[0])

    # Calculating controls
    v = params['k_v'] * dist_to_target
    omega = params['k_omega'] * wrap_angle(theta_desired - current_theta)

    v = np.clip(v, -params['v_max'], params['v_max'])
    omega = np.clip(omega, -params['omega_max'], params['omega_max'])

    return [v, omega]
