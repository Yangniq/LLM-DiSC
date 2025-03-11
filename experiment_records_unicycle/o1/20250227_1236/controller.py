
import math

def get_control(index, all_state, all_ctrl, target_pos, obs_pos, agent_R, state_info):
    # Constants
    k_attr = 1.0            # Gain for attractive potential
    k_rep_obs = 1.0         # Gain for obstacle repulsive potential
    d0_obs = 2.0            # Influence distance for obstacles
    k_rep_agent = 1.0       # Gain for agent repulsive potential
    d0_agent = 1.0          # Influence distance for other agents
    k_omega = 2.0           # Gain for omega (angular velocity control)
    v_max = 1.0             # Maximum linear velocity
    k_v = 0.5               # Gain to reduce speed when turning
    safe_distance = agent_R + 0.1  # Minimal allowed distance to obstacles and agents

    # Get the current state of the agent
    x, y, theta = all_state[index]

    # Get the target position
    x_t, y_t = target_pos

    # Compute the attractive force towards the target
    F_attr_x = k_attr * (x_t - x)
    F_attr_y = k_attr * (y_t - y)

    # Initialize repulsive forces
    F_rep_x = 0.0
    F_rep_y = 0.0

    # Obstacle avoidance
    for obstacle in obs_pos:
        num_vertices = len(obstacle)
        for i in range(num_vertices - 1):
            x1, y1 = obstacle[i]
            x2, y2 = obstacle[i + 1]
            dist, (closest_x, closest_y) = point_to_segment_distance(x, y, x1, y1, x2, y2)
            if dist < d0_obs:
                # Compute the repulsive force magnitude
                F_rep_mag = k_rep_obs * (1.0 / dist - 1.0 / d0_obs) / (dist ** 2)
                # Compute the direction of the repulsive force
                F_rep_dir_x = (x - closest_x) / dist
                F_rep_dir_y = (y - closest_y) / dist
                # Update repulsive force components
                F_rep_x += F_rep_mag * F_rep_dir_x
                F_rep_y += F_rep_mag * F_rep_dir_y

    # Avoidance of other agents
    for i, state in enumerate(all_state):
        if i != index:
            x_other, y_other, _ = state
            dist = math.hypot(x - x_other, y - y_other)
            if dist < d0_agent:
                # Compute the repulsive force magnitude
                F_rep_mag = k_rep_agent * (1.0 / dist - 1.0 / d0_agent) / (dist ** 2)
                # Compute the direction of the repulsive force
                F_rep_dir_x = (x - x_other) / dist
                F_rep_dir_y = (y - y_other) / dist
                # Update repulsive force components
                F_rep_x += F_rep_mag * F_rep_dir_x
                F_rep_y += F_rep_mag * F_rep_dir_y

    # Compute the net force
    F_net_x = F_attr_x + F_rep_x
    F_net_y = F_attr_y + F_rep_y

    # Compute the desired heading angle
    psi_desired = math.atan2(F_net_y, F_net_x)

    # Compute heading error
    theta_error = psi_desired - theta
    # Normalize theta_error to [-pi, pi]
    theta_error = (theta_error + math.pi) % (2 * math.pi) - math.pi

    # Control law for omega
    omega = k_omega * theta_error

    # Linear velocity control
    # Reduce speed when heading error is large or obstacles are close
    v = v_max * math.exp(-k_v * abs(theta_error))

    # If close to the target, stop moving
    distance_to_target = math.hypot(x_t - x, y_t - y)
    if distance_to_target < 0.5:
        v = 0.0
        omega = 0.0

    # Limit omega to a maximum value
    omega_max = math.pi  # Max angular speed of 180 degrees per second
    if omega > omega_max:
        omega = omega_max
    elif omega < -omega_max:
        omega = -omega_max

    # Return the control input
    control_input = [v, omega]
    return control_input

def point_to_segment_distance(px, py, x1, y1, x2, y2):
    # Compute the distance from point (px, py) to segment (x1,y1)-(x2,y2)
    # Returns the distance and the closest point on the segment
    line_mag = math.hypot(x2 - x1, y2 - y1)
    if line_mag < 1e-6:
        # The segment is a point
        dist = math.hypot(px - x1, py - y1)
        closest_point = (x1, y1)
        return dist, closest_point

    # Parameter u in [0,1] representing the projection of P onto the line segment
    u = ((px - x1)*(x2 - x1) + (py - y1)*(y2 - y1)) / (line_mag**2)

    if u < 0:
        # Closest point is (x1, y1)
        closest_x, closest_y = x1, y1
    elif u > 1:
        # Closest point is (x2, y2)
        closest_x, closest_y = x2, y2
    else:
        # Closest point is between (x1, y1) and (x2, y2)
        closest_x = x1 + u*(x2 - x1)
        closest_y = y1 + u*(y2 - y1)
    dist = math.hypot(px - closest_x, py - closest_y)
    return dist, (closest_x, closest_y)
