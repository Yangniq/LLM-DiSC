
import math

def get_control(index, all_state, all_ctrl, target_pos, obs_pos, agent_R, state_info):
    current_state = all_state[index]
    px, py, theta = current_state
    target_x, target_y = target_pos
    
    # Check if the agent has reached the target
    dx_target = target_x - px
    dy_target = target_y - py
    distance_to_target = math.hypot(dx_target, dy_target)
    if distance_to_target < 0.5:
        return [0.0, 0.0]
    
    # Controller parameters
    K_attract = 1.0
    K_repel_obs = 0.8
    K_repel_agent = 0.5
    K_omega = 3.0
    v_nominal = 1.5
    safe_distance_agent = 2.0 * agent_R + 0.5  # Buffer distance for agents
    
    # Calculate attractive force towards target
    attraction = [dx_target, dy_target]
    attraction_mag = math.hypot(attraction[0], attraction[1])
    if attraction_mag > 1e-6:
        attraction = [K_attract * attraction[0]/attraction_mag, K_attract * attraction[1]/attraction_mag]
    else:
        attraction = [0.0, 0.0]
    
    # Calculate repulsive forces from obstacles
    repulsion_obs = [0.0, 0.0]
    for polygon in obs_pos:
        for i in range(len(polygon) - 1):
            A = polygon[i]
            B = polygon[i+1]
            ax, ay = A
            bx, by = B
            
            # Vector AB and AP
            abx = bx - ax
            aby = by - ay
            apx = px - ax
            apy = py - ay
            
            # Project AP onto AB to find closest point
            t = (apx * abx + apy * aby) / (abx**2 + aby**2 + 1e-8)
            t = max(0.0, min(1.0, t))
            closest_x = ax + t * abx
            closest_y = ay + t * aby
            
            # Distance to the closest point
            dx_closest = px - closest_x
            dy_closest = py - closest_y
            distance = math.hypot(dx_closest, dy_closest)
            
            if distance < 2.0 * agent_R:
                if distance < 1e-6:
                    continue
                direction_x = dx_closest / distance
                direction_y = dy_closest / distance
                magnitude = K_repel_obs / (distance ** 2)
                repulsion_obs[0] += direction_x * magnitude
                repulsion_obs[1] += direction_y * magnitude
    
    # Calculate repulsive forces from other agents
    repulsion_agent = [0.0, 0.0]
    for i, other_state in enumerate(all_state):
        if i == index:
            continue
        ox, oy, _ = other_state
        dx_agent = ox - px
        dy_agent = oy - py
        distance = math.hypot(dx_agent, dy_agent)
        
        if distance < safe_distance_agent:
            if distance < 1e-6:
                continue
            direction_x = (px - ox) / distance
            direction_y = (py - oy) / distance
            magnitude = K_repel_agent / (distance ** 2)
            repulsion_agent[0] += direction_x * magnitude
            repulsion_agent[1] += direction_y * magnitude
    
    # Sum all forces to get resultant direction
    total_force_x = attraction[0] + repulsion_obs[0] + repulsion_agent[0]
    total_force_y = attraction[1] + repulsion_obs[1] + repulsion_agent[1]
    
    # Determine desired heading from resultant force
    if total_force_x == 0 and total_force_y == 0:
        desired_theta = theta
    else:
        desired_theta = math.atan2(total_force_y, total_force_x)
    
    # Adjust angular velocity to turn towards desired_theta
    delta_theta = desired_theta - theta
    delta_theta = (delta_theta + math.pi) % (2 * math.pi) - math.pi
    omega = K_omega * delta_theta
    
    # Adjust linear velocity based on proximity to obstacles and agents
    v = v_nominal
    repulsion_total = math.hypot(repulsion_obs[0] + repulsion_agent[0], repulsion_obs[1] + repulsion_agent[1])
    if repulsion_total > 1.0:
        v = max(0.5, v_nominal - 1.0)
    
    return [v, omega]
