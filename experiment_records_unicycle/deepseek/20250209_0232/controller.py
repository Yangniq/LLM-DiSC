
import math

def get_control(index, all_state, all_ctrl, target_pos, obs_pos, agent_R, state_info):
    # Constants for tuning
    K_attract = 1.0
    K_repel_obs = 2.0
    K_repel_agent = 2.0
    d_safe_obs = 0.5
    d_safe_agent = 0.5
    max_v = 1.5
    max_omega = 3.0
    K_omega = 3.0
    slowing_radius = 2.0

    px, py, theta = all_state[index]
    target_x, target_y = target_pos

    # Check if the agent is close enough to the target
    distance_to_target = math.hypot(target_x - px, target_y - py)
    if distance_to_target < 0.1:
        return [0.0, 0.0]

    # Compute attraction force towards target
    dx_attract = target_x - px
    dy_attract = target_y - py
    mag_attract = math.hypot(dx_attract, dy_attract)
    if mag_attract > 1e-5:
        dx_attract = (dx_attract / mag_attract) * K_attract
        dy_attract = (dy_attract / mag_attract) * K_attract
    else:
        dx_attract, dy_attract = 0.0, 0.0

    # Compute repulsion from obstacles
    repulsion_obs_x, repulsion_obs_y = 0.0, 0.0
    for polygon in obs_pos:
        for i in range(len(polygon) - 1):
            A = polygon[i]
            B = polygon[i + 1]
            x1, y1 = A
            x2, y2 = B

            # Calculate closest point on segment A-B to (px, py)
            abx = x2 - x1
            aby = y2 - y1
            apx = px - x1
            apy = py - y1
            t = (apx * abx + apy * aby) / (abx**2 + aby**2 + 1e-8)
            t = max(0.0, min(1.0, t))
            closest_x = x1 + abx * t
            closest_y = y1 + aby * t

            dx_closest = px - closest_x
            dy_closest = py - closest_y
            distance = math.hypot(dx_closest, dy_closest)
            effective_distance = distance - agent_R

            if effective_distance < d_safe_obs and effective_distance > 0:
                mag_dir = math.hypot(dx_closest, dy_closest)
                if mag_dir > 1e-5:
                    dir_x = dx_closest / mag_dir
                    dir_y = dy_closest / mag_dir
                    repulsion_strength = K_repel_obs / (effective_distance ** 2 + 1e-5)
                    repulsion_obs_x += dir_x * repulsion_strength
                    repulsion_obs_y += dir_y * repulsion_strength

    # Compute repulsion from other agents
    repulsion_agents_x, repulsion_agents_y = 0.0, 0.0
    for j in range(len(all_state)):
        if j == index:
            continue
        ox, oy, _ = all_state[j]
        dx = ox - px
        dy = oy - py
        distance = math.hypot(dx, dy)
        effective_distance = distance - 2 * agent_R

        if effective_distance < d_safe_agent and effective_distance > 0:
            dir_x = px - ox
            dir_y = py - oy
            mag_dir = math.hypot(dir_x, dir_y)
            if mag_dir > 1e-5:
                dir_x /= mag_dir
                dir_y /= mag_dir
                repulsion_strength = K_repel_agent / (effective_distance ** 2 + 1e-5)
                repulsion_agents_x += dir_x * repulsion_strength
                repulsion_agents_y += dir_y * repulsion_strength

    # Total force components
    total_force_x = dx_attract + repulsion_obs_x + repulsion_agents_x
    total_force_y = dy_attract + repulsion_obs_y + repulsion_agents_y

    # Compute desired angle and adjust angle difference
    desired_angle = math.atan2(total_force_y, total_force_x)
    angle_diff = desired_angle - theta
    angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi  # Normalize to [-pi, pi]

    # Angular velocity control
    omega = K_omega * angle_diff
    omega = max(-max_omega, min(max_omega, omega))

    # Linear velocity control
    if distance_to_target < slowing_radius:
        v = max_v * (distance_to_target / slowing_radius)
    else:
        v = max_v

    # Reduce speed if avoiding obstacles or agents
    if (repulsion_obs_x != 0 or repulsion_obs_y != 0) or (repulsion_agents_x != 0 or repulsion_agents_y != 0):
        v *= 0.5

    v = max(0.0, min(max_v, v))

    return [v, omega]
