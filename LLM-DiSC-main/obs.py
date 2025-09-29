import numpy as np
from shapely.geometry import Point, Polygon
from shapely.ops import nearest_points


class Obstacle:
    def distance_to_agent(self, agent_pos):
        raise NotImplementedError("Subclasses should implement this method")



class RectangleObstacle(Obstacle):
    def __init__(self, x_min, y_min, x_max, y_max):
        self.rect = (x_min, y_min, x_max, y_max)

    def distance_to_agent(self, agent_pos):
        x_a, y_a = agent_pos
        x_min, y_min, x_max, y_max = self.rect
        nearest_x = min(max(x_a, x_min), x_max)
        nearest_y = min(max(y_a, y_min), y_max)
        dx = x_a - nearest_x
        dy = y_a - nearest_y
        distance = np.hypot(dx, dy)
        return dx, dy, distance



class PolygonObstacle(Obstacle):
    def __init__(self, polygon_points):
        # self.polygon_points = np.array(polygon_points)
        self.polygon = Polygon(polygon_points)

    def distance_to_agent(self, agent_pos):
        point = Point(agent_pos)

        nearest_point_pair = nearest_points(point, self.polygon)

        nearest_point_on_polygon = nearest_point_pair[1]

        dx = point.x - nearest_point_on_polygon.x
        dy = point.y - nearest_point_on_polygon.y

        distance = point.distance(self.polygon)
        return dx, dy, distance



class CircleObstacle(Obstacle):
    def __init__(self, center_x, center_y, radius):
        self.center = (center_x, center_y)
        self.radius = radius

    def distance_to_agent(self, agent_pos):
        x_a, y_a = agent_pos
        x_c, y_c = self.center
        dx = x_a - x_c
        dy = y_a - y_c
        distance = np.hypot(dx, dy)
        return dx, dy, distance
