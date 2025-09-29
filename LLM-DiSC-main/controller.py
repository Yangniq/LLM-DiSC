
def get_control(index, all_state, all_ctrl, target_pos, obs_pos, agent_R, state_info):
    import numpy as np

    # Constants
    # Maximum speeds
    MAX_SPEED = 1.0  # Max linear speed (units per second)
    MAX_OMEGA = 1.0  # Max angular speed (radians per second)
    # Threshold to consider the agent has reached the target
    GOAL_THRESHOLD = 0.1

    # Gain constants
    K_ATTRACTIVE = 1.0        # Gain for attractive potential
    K_REPULSIVE = 0.5         # Gain for repulsive potential from obstacles
    K_REPULSIVE_AGENT = 0.8   # Gain for repulsive potential from other agents
    REPULSIVE_RANGE = 2.0     # Effective range of repulsive force
    REPULSIVE_AGENT_RANGE = 1.0  # Effective range of repulsive force from other agents

    # Get the agent's current state
    x, y, theta = all_state[index]

    # Get the agent's target position
    goal_x, goal_y = target_pos

    # Compute distance to goal
    dist_to_goal = np.hypot(goal_x - x, goal_y - y)

    # If the agent is close enough to the goal, stop moving
    if dist_to_goal < GOAL_THRESHOLD:
        control_input = [0.0, 0.0]
        return control_input

    # Compute attractive force towards the goal
    force_attr_x = K_ATTRACTIVE * (goal_x - x)
    force_attr_y = K_ATTRACTIVE * (goal_y - y)

    # Initialize repulsive forces
    force_rep_x = 0.0
    force_rep_y = 0.0

    # Compute repulsive forces from obstacles
    for obs in obs_pos:
        # For each obstacle polygon
        obs_vertices = obs  # List of vertices [(x1,y1), (x2,y2), ...]
        num_vertices = len(obs_vertices)
        # Loop over each edge of the polygon
        for i in range(num_vertices - 1):
            # Get the edge from obs_vertices[i] to obs_vertices[i+1]
            p1 = np.array(obs_vertices[i])
            p2 = np.array(obs_vertices[i + 1])

            # Compute the closest point on the edge to the agent
            # Using projection of point onto line segment
            line_vec = p2 - p1
            p1_to_agent = np.array([x, y]) - p1
            line_len = np.dot(line_vec, line_vec)
            if line_len == 0:
                # p1 and p2 are the same point
                closest_point = p1
            else:
                t = np.clip(np.dot(p1_to_agent, line_vec) / line_len, 0.0, 1.0)
                closest_point = p1 + t * line_vec

            # Compute distance to the closest point
            dist = np.linalg.norm(np.array([x, y]) - closest_point)

            if dist < REPULSIVE_RANGE:
                # Compute repulsive force
                repulsive_strength = K_REPULSIVE * (1.0 / dist - 1.0 / REPULSIVE_RANGE) / (dist ** 2)
                force_rep_x += repulsive_strength * (x - closest_point[0]) / dist
                force_rep_y += repulsive_strength * (y - closest_point[1]) / dist

    # Compute repulsive forces from other agents
    for i, other_state in enumerate(all_state):
        if i != index:
            other_x, other_y, _ = other_state
            # Compute distance to other agent
            dist = np.hypot(other_x - x, other_y - y)
            if dist < REPULSIVE_AGENT_RANGE + 2 * agent_R:
                # Compute repulsive force
                repulsive_strength = K_REPULSIVE_AGENT * (1.0 / dist - 1.0 / (REPULSIVE_AGENT_RANGE + 2 * agent_R)) / (dist ** 2)
                force_rep_x += repulsive_strength * (x - other_x) / dist
                force_rep_y += repulsive_strength * (y - other_y) / dist

    # Total forces
    total_force_x = force_attr_x + force_rep_x
    total_force_y = force_attr_y + force_rep_y

    # Compute desired heading angle
    desired_theta = np.arctan2(total_force_y, total_force_x)

    # Compute the difference between current and desired heading
    angle_diff = desired_theta - theta
    # Wrap angle_diff to [-pi, pi]
    angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi

    # Compute control inputs
    # Gain for angular velocity
    K_OMEGA = 2.0
    omega = K_OMEGA * angle_diff
    omega = np.clip(omega, -MAX_OMEGA, MAX_OMEGA)

    # Set linear speed proportional to the alignment with the desired heading
    # Slow down if the heading error is large
    v = MAX_SPEED * (1 - min(abs(angle_diff), np.pi/2)/(np.pi/2))
    v = np.clip(v, 0.0, MAX_SPEED)

    control_input = [v, omega]
    return control_input
