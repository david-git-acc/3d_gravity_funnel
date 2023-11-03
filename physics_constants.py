import numpy as np

# STORING ALL THE PHYSICS CONSTANTS HERE for easy access and modification

# Mew (u since you can't type greek characters) is the coefficient of friction
# 0 <= u <= 1, u = 0 -> no friction, u = 1 -> extreme friction
u = 0.05

# small g, strength of gravitational force on earth in ms^-2
g = 9.81

# e, the coefficient of restitution. Determines how much speed of the ball is lost upon collision with either a wall or ball
# 0 <= e <= 1, 0 -> total speed loss upon collisions, 1 -> perfectly elastic collisions
e = 0.8

# Size of the time step. As dt -> 0, the more accurate and stable the simulation (as real time is continuous)
# A large dt will produce noticeably discrete movements and decrease simulation stability
dt=0.01

# The sizes of the balls - the balls will be visualised as scatter points for efficiency of both design and performance
ball_sizes = 3*np.array([32, 70, 42, 42]) # 50

# Calculate approximately the radii of the balls - used to identify collisions
radii = np.sqrt(ball_sizes) * 0.012

# The masses of the balls will be directly proportional to their sizes - same density
m = np.copy(ball_sizes)