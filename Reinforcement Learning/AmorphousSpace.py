import os

# Get the absolute path to the directory containing this script
dir_path = os.path.dirname(os.path.realpath(__file__))

# Construct the absolute path to the file
file_path = os.path.join(dir_path, 'circles.txt')

import numpy as np
from gym import spaces

class AmorphousSpace(spaces.Space):
    """Custom space class for representing an amorphous observation space."""

    def __init__(self):
        """
        Initialize the amorphous space.

        Parameters
        ----------
        - circles : list
            A list of dictionaries representing the circular regions in the 
            space. Each dictionary should contain the following keys:
          - 'center' : list 
                The center of the circle (a 2D numpy array).
          - 'radius' : float
                The radius of the circle.
        """
        # read circles from a circles.txt file
        circles = []
        with open(file_path, 'r') as f:
            for line in f:
                x, y, r = line.strip().split(',')
                circles.append({'center': np.array([float(x), float(y)]), 'radius': float(r)})

        self.circles = circles
        self.low = np.array([circle['center'][0] - circle['radius'] for circle in circles])
        self.high = np.array([circle['center'][0] + circle['radius'] for circle in circles])
        super(AmorphousSpace, self).__init__((2,))

    def sample(self):
        """Sample a random point from the amorphous space."""
        # Choose a random circle
        circle = self.circles[np.random.randint(len(self.circles))]

        # Generate a random point within the circle
        angle = np.random.uniform(low=0, high=2*np.pi)
        distance = np.random.uniform(low=0, high=circle['radius'])
        x = circle['center'][0] + distance * np.cos(angle)
        y = circle['center'][1] + distance * np.sin(angle)
        return np.array([x, y])

    def contains(self, x):
        """Check if a point is within the bounds of the amorphous space."""
        for circle in self.circles:
            if np.linalg.norm(x - circle['center']) <= circle['radius']:
                return True
        return False

    def clip(self, x):
        """Clip a point to the bounds of the amorphous space."""
        if self.contains(x):
            return x
        else:
            # Find the nearest point on the boundary of the space
            min_distance = float('inf')
            nearest_point = None
            for circle in self.circles:
                distance = np.linalg.norm(x - circle['center'])
                if distance < min_distance:
                    min_distance = distance
                    nearest_point = circle['radius'] * (x - circle['center']) / distance + circle['center']
            return nearest_point
