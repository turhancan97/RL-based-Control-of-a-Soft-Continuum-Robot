import os

# Get the absolute path to the directory containing this script
dir_path = os.path.dirname(os.path.realpath(__file__))

# Construct the absolute path to the file
file_path = os.path.join(dir_path, 'task_space.npy')

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import matplotlib.path as mpltPath
from gym import spaces


class PolygonSpace(spaces.Space):
    def __init__(self):
        self.points = np.load(file_path)
        self.hull = ConvexHull(self.points)
        self.polygon = mpltPath.Path(self.points[self.hull.vertices])
        self.bounding_box = self.calculate_bounding_box()
        super(PolygonSpace, self).__init__((2,), np.float32)

    def calculate_bounding_box(self):
        min_x, min_y = np.min(self.points, axis=0)
        max_x, max_y = np.max(self.points, axis=0)
        return (min_x, min_y, max_x, max_y)

    def sample(self):
        min_x, min_y, max_x, max_y = self.bounding_box
        while True:
            x = np.random.uniform(min_x, max_x)
            y = np.random.uniform(min_y, max_y)
            if self.contains((x, y)):
                return np.array([x, y])

    def contains(self, point):
        return self.polygon.contains_point(point)

    def clip(self, point):
        # This is a placeholder; real implementation would require more complex logic
        # Possibly leverage computational geometry libraries for efficient implementation
        return np.clip(point, self.bounding_box[:2], self.bounding_box[2:])
