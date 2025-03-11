
def get_control(index, all_state, all_ctrl, target_pos, obs_pos, agent_R, state_info):
    import numpy as np
    # Parameters
    Ka = 2.0    # Increased attractive gain
    Kr = 1.5    # Decreased repulsive gain
    rho0 = 2.0  # Reduced influence distance
    k_omega = 2.0
    v_max = 1.0
    epsilon = 0.1  # Threshold distance to target

    # Extract the current state of the agent
    current_state = all_state[index]
    agent_pos = np.array([current_state[0], current_state[1]])
    agent_theta = current_state[2]

    # Compute distance to the target
    goal_pos = np.array(target_pos)
    to_goal_vec = goal_pos - agent_pos
    dist_to_goal = np.linalg.norm(to_goal_vec)

    # Check if the agent has reached the target
    if dist_to_goal < epsilon:
        # Agent has reached the target; stop moving
        control_input = [0.0, 0.0]
        return control_input

    # Compute Attractive force towards the goal
    Fa = Ka * to_goal_vec / dist_to_goal  # Normalize to have consistent magnitude

    # Repulsive forces from obstacles
    Fr = np.array([0.0, 0.0])
    # For each obstacle
    for polygon in obs_pos:
        closest_point = get_closest_point_on_polygon(agent_pos, polygon)
        vector_to_obstacle = agent_pos - closest_point
        dist = np.linalg.norm(vector_to_obstacle)
        if dist <= rho0:
            # Compute repulsive force
            Fr += Kr * (1.0 / dist - 1.0 / rho0) * (vector_to_obstacle) / (dist ** 3)

    # Repulsive forces from other agents
    for i, other_state in enumerate(all_state):
        if i != index:
            other_pos = np.array([other_state[0], other_state[1]])
            vector_to_agent = agent_pos - other_pos
            dist = np.linalg.norm(vector_to_agent)
            if dist <= rho0:
                # Compute repulsive force
                Fr += Kr * (1.0 / dist - 1.0 / rho0) * (vector_to_agent) / (dist ** 3)

    # Sum forces
    F = Fa + Fr

    # Compute desired heading
    theta_d = np.arctan2(F[1], F[0])

    # Compute angular error
    angle_diff = (theta_d - agent_theta + np.pi) % (2 * np.pi) - np.pi  # Normalize between [-pi, pi]

    # Compute control inputs
    omega = k_omega * angle_diff

    # Modify v to prevent it from becoming too small
    v = v_max * (1 / (1 + abs(angle_diff)))  # Inverse function to maintain forward motion

    # Ensure minimum speed
    v_min = 0.1
    v = max(v, v_min)

    # Return control input
    control_input = [v, omega]
    return control_input

def get_closest_point_on_polygon(point, polygon):
    import numpy as np
    min_dist = float('inf')
    closest_point = None
    for i in range(len(polygon) - 1):  # Last vertex is same as first
        p1 = np.array(polygon[i])
        p2 = np.array(polygon[i + 1])
        cp = closest_point_on_line_segment(point, p1, p2)
        dist = np.linalg.norm(point - cp)
        if dist < min_dist:
            min_dist = dist
            closest_point = cp
    return closest_point

def closest_point_on_line_segment(p, a, b):
    import numpy as np
    # Compute the projection of point p onto line segment ab
    ap = p - a
    ab = b - a
    ab_norm_sq = np.dot(ab, ab)
    if ab_norm_sq == 0:
        return a
    t = np.dot(ap, ab) / ab_norm_sq
    t = max(0, min(1, t))  # Clamp t to [0,1]
    return a + t * ab
