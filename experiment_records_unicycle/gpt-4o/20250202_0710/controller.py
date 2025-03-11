
import numpy as np

def get_control(index, all_state, all_ctrl, target_pos, obs_pos, agent_R, state_info):
    # Extract current state of the agent
    current_state = all_state[index]
    px, py, theta = current_state
    
    # Extract target position for the agent
    tx, ty = target_pos
    
    # Control parameters
    k_v = 1.0  # Gain for linear velocity
    k_omega = 1.0  # Gain for angular velocity
    v_max = 1.0  # Maximum linear velocity
    omega_max = np.pi  # Maximum angular velocity
    safety_margin = 1.0  # Safe distance from obstacles
    
    # Calculate desired direction to the target
    direction_to_target = np.arctan2(ty - py, tx - px)
    distance_to_target = np.hypot(tx - px, ty - py)
    angle_to_target = direction_to_target - theta
    
    # Normalize the angle to the range [-pi, pi]
    angle_to_target = (angle_to_target + np.pi) % (2 * np.pi) - np.pi

    # Proportional control to reach the target
    v = k_v * distance_to_target
    omega = k_omega * angle_to_target

    # Ensure velocity doesn't exceed maximum values
    v = np.clip(v, 0, v_max)
    omega = np.clip(omega, -omega_max, omega_max)

    # Initialize nearest obstacle parameters
    nearest_obstacle_distance = float('inf')
    closest_point = None

    # Obstacle avoidance using a repulsive potential field with CBF consideration
    for obs in obs_pos:
        for i in range(len(obs)):
            ox1, oy1 = obs[i]
            ox2, oy2 = obs[(i + 1) % len(obs)]
            test_point = closest_point_on_segment(px, py, ox1, oy1, ox2, oy2)
            dist_to_edge = np.hypot(px - test_point[0], py - test_point[1])
            h = (px - test_point[0]) ** 2 + (py - test_point[1]) ** 2 - agent_R ** 2
            
            if dist_to_edge < nearest_obstacle_distance:
                nearest_obstacle_distance = dist_to_edge
                closest_point = test_point

    # If near an obstacle, adjust control inputs using CBF constraints
    if closest_point is not None and nearest_obstacle_distance < safety_margin:
        cx, cy = closest_point
        h = (px - cx) ** 2 + (py - cy) ** 2 - agent_R ** 2
        dh = 2 * (px - cx) * v * np.cos(theta) + 2 * (py - cy) * v * np.sin(theta)
        
        if dh + 2 * h < 0:
            safe_direction = np.array([px - cx, py - cy])
            safe_direction /= np.linalg.norm(safe_direction)
            v_target = 0.5 * v_max  # Reduce velocity
            omega_target = np.arctan2(safe_direction[1], safe_direction[0]) - theta
            omega_target = (omega_target + np.pi) % (2 * np.pi) - np.pi
            v = np.clip(v_target, 0, v_max)
            omega = np.clip(omega_target, -omega_max, omega_max)

    # Agent collision avoidance using CBF constraints
    for i, other_state in enumerate(all_state):
        if i == index:
            continue
        xp, yp, _ = other_state
        dist_to_agent = np.hypot(px - xp, py - yp)
        h_agent = (px - xp) ** 2 + (py - yp) ** 2 - (2 * agent_R) ** 2
        dh_agent = 2 * (px - xp) * v * np.cos(theta) + 2 * (py - yp) * v * np.sin(theta)
        
        if dist_to_agent < 2 * agent_R or (dh_agent + 2 * h_agent) < 0:
            safe_direction = np.array([px - xp, py - yp])
            safe_direction /= np.linalg.norm(safe_direction)
            target_direction = np.arctan2(safe_direction[1], safe_direction[0])
            omega_target = target_direction - theta
            omega_target = (omega_target + np.pi) % (2 * np.pi) - np.pi
            v = 0  # Stop movement
            omega = np.clip(omega_target, -omega_max, omega_max)  # Adjust rotation

    return [v, omega]

def closest_point_on_segment(px, py, ax, ay, bx, by):
    # Compute the closest point on the line segment (ax, ay)-(bx, by) to point (px, py)
    ab = np.array([bx - ax, by - ay])
    ap = np.array([px - ax, py - ay])
    t = np.dot(ap, ab) / np.dot(ab, ab)
    t = np.clip(t, 0.0, 1.0)
    closest_point = np.array([ax, ay]) + t * ab
    return closest_point
