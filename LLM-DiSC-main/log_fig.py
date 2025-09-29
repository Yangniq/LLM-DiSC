import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Circle, Rectangle
import os
from env import INFO
from shapely.geometry import Polygon
from obs import RectangleObstacle, PolygonObstacle, CircleObstacle


def euclidean_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def figure(posInfo, info: INFO, run_directory):
    color_palette = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    labels = [f'Agent {i + 1}' for i in range(info.agent_num)]
    if info.agent_num > len(color_palette):
        colors = [color_palette[i % len(color_palette)] for i in range(info.agent_num)]
    else:
        colors = color_palette[:info.agent_num]


    plt.figure(figsize=(10, 10))

    legend_handles = []
    legend_labels = []

    targets = info.target_pos
    for i, target in enumerate(targets):
        plt.plot(target[0], target[1], color=colors[i], marker='*', markersize=10)
        if i == 0:
            legend_handles.append(plt.Line2D([0], [0], color='black', marker='*', linestyle='None', markersize=10))
            legend_labels.append('Target')


    obstacle_plotted = False
    for i, obstacle in enumerate(info.obs):
        if isinstance(obstacle, RectangleObstacle):
            x_min, y_min, x_max, y_max = obstacle.rect
            obstacle_patch = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=1.5, edgecolor='brown',
                                       facecolor='brown')
            plt.gca().add_patch(obstacle_patch)
            if not obstacle_plotted:
                legend_handles.append(obstacle_patch)
                legend_labels.append('Obstacle')
                obstacle_plotted = True

        elif isinstance(obstacle, PolygonObstacle):
            polygon_coords = obstacle.polygon.exterior.coords
            polygon = Polygon(polygon_coords)
            x, y = np.array(polygon_coords).T
            polygon_patch = plt.Polygon(np.column_stack((x, y)), color='brown', fill=False, linestyle='-',
                                        linewidth=1.5)
            plt.gca().add_patch(polygon_patch)
            if not obstacle_plotted:
                legend_handles.append(polygon_patch)
                legend_labels.append('Obstacle')
                obstacle_plotted = True


            buffered_polygon = polygon.buffer(info.agent_radius)
            buffered_x, buffered_y = np.array(buffered_polygon.exterior.coords).T
            buffered_polygon_patch = plt.Polygon(np.column_stack((buffered_x, buffered_y)), color='gray',
                                                 fill=False,
                                                 linestyle='--', linewidth=1.5)
            plt.gca().add_patch(buffered_polygon_patch)

            centroid =polygon.centroid
            plt.plot(centroid.x, centroid.y, 'kx', markersize=10)

        elif isinstance(obstacle, CircleObstacle):
            center_x, center_y = obstacle.center
            radius = obstacle.radius
            circle_patch = Circle((center_x, center_y), radius, color='brown', fill=False, linestyle='-', linewidth=1.5)
            plt.gca().add_patch(circle_patch)
            if not obstacle_plotted:
                legend_handles.append(circle_patch)
                legend_labels.append('Obstacle')
                obstacle_plotted = True


    # Plot the agents' trajectories
    for i, agent_positions in enumerate(posInfo):
        x = [pos[0] for pos in agent_positions]
        y = [pos[1] for pos in agent_positions]
        plt.plot(x, y, color=colors[i], linewidth=2)


        plt.plot(x[0], y[0], '^', color=colors[i], markersize=6)


        if i == 0:
            legend_handles.append(plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='black', markersize=8))
            legend_labels.append('Start')


    plt.legend(
        handles=legend_handles, labels=legend_labels,
        loc='center right',
        bbox_to_anchor=(1, 0.75),
        fontsize=15,
        frameon=True, edgecolor='black',
        framealpha=0.7, borderpad=1
    ).get_frame().set_linewidth(1)



    plt.grid(True)

    existing_files = [f for f in os.listdir(run_directory) if
                      f.startswith('movement_trajectories') and f.endswith('.png')]
    next_number = len(existing_files) + 1
    figure_path = os.path.join(run_directory, f'movement_trajectories_{next_number}.png')
    plt.savefig(figure_path)



    plt.figure(figsize=(10, 6))
# Plot the minimum distance at each time step.
    min_distances = []
    safe_distance = 2*info.agent_radius


    min_t = min(len(item) for item in posInfo)
    for t in range(min_t):
        distances = []

        for i in range(len(posInfo)):
            for j in range(i + 1, len(posInfo)):
                dist = euclidean_distance(posInfo[i][t], posInfo[j][t])
                distances.append(dist)

        if not distances:
            min_distances.append(0)
        else:
            min_distances.append(min(distances))


    plt.plot(range(len(min_distances)), min_distances, linewidth=1, label='Minimum Distance')

    plt.axhline(y=safe_distance, color='r', linestyle='--', linewidth=2, label=f'Safe Distance')
    plt.ylim(0)
    plt.grid(True)
    plt.legend(prop={'size': 16}, loc='center', bbox_to_anchor=(0.75, 0.25))

    distance_figure_path = os.path.join(run_directory, f'minimum_distance_over_time_{next_number}.png')
    plt.savefig(distance_figure_path)



def ctrl_figure(ctrlInfo, newCInfo, run_directory, agents):
    """
    Plot the control signals over time.
    """
    agent_num = agents

    plt.figure(figsize=(15, 10))
    for agent_index in range(agent_num):

        ctrl_info_agent = ctrlInfo[agent_index]
        new_cinfo_agent = newCInfo[agent_index]


        time_points = range(len(ctrl_info_agent))

        ctrl_x = [point[0] for point in ctrl_info_agent]
        ctrl_y = [point[1] for point in ctrl_info_agent]
        new_ctrl_x = [point[0] for point in new_cinfo_agent]
        new_ctrl_y = [point[1] for point in new_cinfo_agent]

        plt.subplot(agent_num, 1, agent_index + 1)

        plt.plot(time_points, ctrl_x, label='Standard Control Signal U_ref1')
        plt.plot(time_points, new_ctrl_x, label='Revised Control Signal U1', linestyle='--')

        plt.plot(time_points, ctrl_y, label='Standard Control Signal U_ref2')
        plt.plot(time_points, new_ctrl_y, label='Revised Control Signal U2', linestyle='--')

        plt.legend()

        plt.title('Control Signal Changes for Agent {}'.format(agent_index + 1))
        plt.xlabel('Time Points')
        plt.ylabel('Control Signal Value')
    plt.tight_layout()

    existing_files = [f for f in os.listdir(run_directory) if
                      f.startswith('control_signal_changes') and f.endswith('.png')]
    next_number = len(existing_files) + 1
    control_figure_path = os.path.join(run_directory, f'control_signal_changes_{next_number}.png')
    plt.savefig(control_figure_path)

