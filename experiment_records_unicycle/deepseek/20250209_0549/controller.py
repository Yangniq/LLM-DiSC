
import math

def get_control(index, all_state, all_ctrl, target_pos, obs_pos, agent_R, state_info):
    # Parameters
    K_att = 1.0
    K_rep_obs = 2.0
    K_rep_agent = 2.0
    K_omega = 3.0
    max_v = 1.5
    stopping_radius = 0.5
    d_safety_obs = 0.5
    d_safety_agent = 0.5
    d0_obs = agent_R + d_safety_obs
    d0_agent = 2 * agent_R + d_safety_agent

    current_state = all_state[index]
    px, py, theta = current_state
    current_pos = (px, py)
    target_x, target_y = target_pos

    # Check if the agent has reached the target
    dx_target = target_x - px
    dy_target = target_y - py
    distance_to_target = math.hypot(dx_target, dy_target)
    if distance_to_target < stopping_radius:
        return [0.0, 0.0]

    # Attractive force towards target
    F_att_x = K_att * dx_target
    F_att_y = K_att * dy_target

    # Repulsion from obstacles
    F_rep_obs_x = 0.0
    F_rep_obs_y = 0.0
    for polygon in obs_pos:
        min_dist = float('inf')
        closest_point = None
        for i in range(len(polygon) - 1):
            p1 = polygon[i]
            p2 = polygon[i + 1]
            a_to_b = (p2[0] - p1[0], p2[1] - p1[1])
            a_to_q = (px - p1[0], py - p1[1])
            denominator = (a_to_b[0] ** 2 + a_to_b[1] ** 2)
            if denominator < 1e-8:
                t = 0.0
            else:
                t = (a_to_q[0] * a_to_b[0] + a_to_q[1] * a_to_b[1]) / denominator
            t = max(0.0, min(1.0, t))
            closest_x = p1[0] + t * a_to_b[0]
            closest_y = p1[1] + t * a_to_b[1]
            dist = math.hypot(px - closest_x, py - closest_y)
            if dist < min_dist:
                min_dist = dist
                closest_point = (closest_x, closest_y)
        if min_dist < d0_obs and min_dist > 0:
            dir_x = px - closest_point[0]
            dir_y = py - closest_point[1]
            norm = math.hypot(dir_x, dir_y)
            if norm > 0:
                dir_x /= norm
                dir_y /= norm
                magnitude = K_rep_obs * (1 / min_dist - 1 / d0_obs) * (1 / (min_dist ** 2))
                F_rep_obs_x += dir_x * magnitude
                F_rep_obs_y += dir_y * magnitude

    # Repulsion from other agents
    F_rep_agent_x = 0.0
    F_rep_agent_y = 0.0
    for j in range(len(all_state)):
        if j == index:
            continue
        other_agent = all_state[j]
        ox, oy = other_agent[0], other_agent[1]
        dx = px - ox
        dy = py - oy
        distance = math.hypot(dx, dy)
        if distance < d0_agent and distance > 0:
            dir_x = dx / distance
            dir_y = dy / distance
            magnitude = K_rep_agent * (1 / distance - 1 / d0_agent) * (1 / (distance ** 2))
            F_rep_agent_x += dir_x * magnitude
            F_rep_agent_y += dir_y * magnitude

    # Total force
    F_total_x = F_att_x + F_rep_obs_x + F_rep_agent_x
    F_total_y = F_att_y + F_rep_obs_y + F_rep_agent_y

    if F_total_x == 0 and F_total_y == 0:
        return [0.0, 0.0]

    desired_angle = math.atan2(F_total_y, F_total_x)
    error = desired_angle - theta
    error = (error + math.pi) % (2 * math.pi) - math.pi
    omega = K_omega * error

    # Linear velocity control
    if distance_to_target < stopping_radius:
        v = 0.0
    else:
        v = max_v

    return [v, omega]
