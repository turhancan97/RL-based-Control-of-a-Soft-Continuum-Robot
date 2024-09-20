import sys # to include the path of the package
sys.path.append('../Reinforcement Learning') # AmorphousSpace.py is in the parent directory
from PolygonSpace import PolygonSpace

import numpy as np
import matplotlib.pyplot as plt

# Instantiate the PolygonSpace
polygon_space = PolygonSpace()

# Sample points from the space
sampled_points = np.array([polygon_space.sample() for _ in range(100)])

# Plot the polygon and the sampled points
plt.figure(figsize=(8, 6))
for simplex in polygon_space.hull.simplices:
    plt.plot(polygon_space.points[simplex, 0], polygon_space.points[simplex, 1], 'k-')

plt.plot(polygon_space.points[:, 0], polygon_space.points[:, 1], 'o', markersize=5, label='Polygon Vertices')
plt.scatter(sampled_points[:, 0], sampled_points[:, 1], alpha=0.5, color='red', label='Sampled Points')
plt.fill(polygon_space.points[polygon_space.hull.vertices, 0], polygon_space.points[polygon_space.hull.vertices, 1], alpha=0.3)
plt.title('Polygon Space and Sampled Points')
plt.legend()
plt.show()
