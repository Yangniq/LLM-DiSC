
def get_control(index, all_state, all_ctrl, target_pos, obs_pos, agent_R, state_info):
    import numpy as np
    import math

    # Get own state
    my_state = all_state[index]
    x, y, theta = my_state

    # Get target position
    x_target, y_target = target_pos

    # Compute attractive force towards target
    force_x = x_target - x
    force_y = y_target - y

    dist_to_target = np.hypot(force_x, force_y)

    if dist_to_target > 0:
        force_x /= dist_to_target
        force_y /= dist_to_target
    else:
        force_x = 0.0
        force_y = 0.0

    # Parameters
    v_max = 1.0       # Max linear speed
    omega_max = 1.0   # Max angular speed
    sensing_range = 5.0  # Sensing range for obstacle avoidance

    # Function to compute distance from point to a line segment
    def point_to_segment_distance(x, y, p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:
            # p1 and p2 are the same point
            return np.hypot(x - x1, y - y1), (x1, y1)
        else:
            t = ((x - x1) * dx + (y - y1) * dy) / (dx * dx + dy * dy)
            t = max(0.0, min(1.0, t))
            proj_x = x1 + t * dx
            proj_y = y1 + t * dy
            return np.hypot(x - proj_x, y - proj_y), (proj_x, proj_y)

    # Avoid obstacles
    for obs in obs_pos:
        min_dist = sensing_range
        closest_point = None
        for i in range(len(obs) - 1):
            p1 = obs[i]
            p2 = obs[i + 1]
            dist, proj_point = point_to_segment_distance(x, y, p1, p2)
            if dist < min_dist:
                min_dist = dist
                closest_point = proj_point
        if min_dist < sensing_range and closest_point is not None:
            # Compute repulsive force
            diff_x = x - closest_point[0]
            diff_y = y - closest_point[1]
            dist = max(min_dist, 0.0001)  # Avoid division by zero
            repulsive_force = (1.0 / dist - 1.0 / sensing_range) / (dist * dist)
            force_x += repulsive_force * (diff_x / dist)
            force_y += repulsive_force * (diff_y / dist)

    # Avoid other agents
    for i, other_state in enumerate(all_state):
        if i != index:
            x_o, y_o, _ = other_state
            diff_x = x - x_o
            diff_y = y - y_o
            dist = np.hypot(diff_x, diff_y)
            if dist < sensing_range and dist > 0.0:
                # Compute repulsive force
                dist_safe = max(dist, 2 * agent_R)
                repulsive_force = (1.0 / dist_safe - 1.0 / sensing_range) / (dist_safe * dist_safe)
                force_x += repulsive_force * (diff_x / dist_safe)
                force_y += repulsive_force * (diff_y / dist_safe)

    # Compute desired heading
    desired_theta = math.atan2(force_y, force_x)
    angle_diff = desired_theta - theta
    # Normalize angle_diff to [-pi, pi]
    angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi

    # Control gains
    K_omega = 1.0
    K_v = 0.5

    # Compute control inputs
    omega = K_omega * angle_diff
    omega = np.clip(omega, -omega_max, omega_max)

    # Adjust linear speed based on alignment with desired direction
    v = K_v * dist_to_target * max(0.0, math.cos(angle_diff))
    v = min(v, v_max)

    # Stop if close to target
    if dist_to_target < 0.5:
        v = 0.0
        omega = 0.0

    control_input = [v, omega]
    return control_input
