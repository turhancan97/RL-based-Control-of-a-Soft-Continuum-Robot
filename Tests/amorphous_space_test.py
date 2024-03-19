# %%
import sys # to include the path of the package
sys.path.append('../Reinforcement Learning') # AmorphousSpace.py is in the parent directory
from AmorphousSpace import AmorphousSpace

import numpy as np
import matplotlib.pyplot as plt

space = AmorphousSpace()
print('Shape of the State: ',space.shape[0]*2)
# Generate a random point in the space
point = space.sample() #[0.06,0.074]
print('Sample Point is ',point)

# Check if the point is within the bounds of the space
print('The point is within the bounds of the space: ',space.contains(point))

# Clip the point to the bounds of the space
clipped_point = space.clip(point)
print('Clip the point to the bounds of the space: ',clipped_point)

# Plot the amorphous space
fig, ax = plt.subplots()
for circle in space.circles:
    ax.add_artist(plt.Circle(circle['center'], circle['radius'], fill=False))
ax.scatter(clipped_point[0],clipped_point[1])
ax.set_title('Space for the Continuum Robot Consisting of Several Circular Shapes')
ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
ax.set_xlim(-0.3, 0.2)
ax.set_ylim(-0.15, 0.3)
plt.show()
# %%
