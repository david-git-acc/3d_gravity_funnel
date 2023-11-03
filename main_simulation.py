from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from numpy.ma import masked_where
from physics_constants import u,e,dt, ball_sizes
from physics_functions import z, calculate_motion, calculate_ball_collisions, calculate_wallphysics
from plot_constants import figsize,res,xy_bound, colours
from plot_functions import wall_marker_update, ball_marker_update, create_cylinder

# The purpose of this program is to create a 3D simulation of balls travelling along a funnel surface
# The model is based off the black hole simulation at the National Space Centre in Leicester, UK
# After running the simulation, the goal is to collect data and results about the balls such as their
# kinetic energy, number of collisions, etc and vary the settings of the simulation to analyse it 
# and find conclusions about how mainly the friction, coefficient of restitution but also other
# factors such as collision frequency affect the kinetic energy,velocity, etc over time of these balls.

# MODELLING ASSUMPTIONS:

# 1. Balls are placed on the surface of the funnel with some initial horizontal velocity - they have no vertical velocity 
#   ( a safe assumption since any vertical velocity would be stopped by the normal reaction force of the ball)
#
# 2. Negligible air resistance to the motion of the balls, as they have small surface area
#
# 3. There are no forces acting on the balls other than weight and friction, which is a constant multiple of the normal reaction
#
# 4. Energy can only be lost through friction or collisions via the coefficient of restitution, e
#
# 5. The balls have equal density - the greater the volume of the ball, the heavier the ball
#
# 6. Gravity/weight of the balls will acts act in the direction of steepest gradient descent
#
# 7. Collisions between balls can occur, but only between pairs of balls at any given timeframe 
#   (the smaller the timestep, the less this becomes an issue since 3-ball collisions are much rarer than 2-ball ones)
#
# 8. All balls are modelled as perfect spheres with uniform radii.

# STARTING FEATURES AND IMPORTANT STORAGE ARRAYS
# NOTE: All positions and velocities will be stored as complex numbers - the real part represents the x-coordinate
# and the imaginary part the y-coordinate - this was done to make coding easier and utilise python support for complex numbers

# This array stores the positions of the balls - the values here are their initial positions
position = 0.75*np.array([2+3j, 3-3.8j, 1-1j, 3+3j])

# This array stores the velocities of the balls - the values here are their initial velocities
velocity = 4* np.array([1-1.25j,-1-0.75j, -0.9+1.1j, 0.76+1.2j])

# Array to check each ball to see if it's currently in collision with a wall
# Needed to avoid triggering the wall collision code more than once for a single collision
collidedwithwallalready = np.array([False] * len(position))

# Array that stores the array indices of balls which have collided with each other
# Each element is of the form [a b] where a,b are the indices of the colliding balls
# Again needed to avoid triggering the same collision code more than once for a single collision
balls_collided_already = np.array([])

# This array stores the positions in xy coordinates, of where balls have collided with the wall boundary
# over the course of the entire simulation
boundary_collision_markers = np.array([])

# This array stores the positions in xy coordinates, of where balls have collided with each other
# also over the course of the entire simulation
ball_collision_markers = np.array([])

# The colour map we'll use for the 3D surface to show higher and lower altitude points more clearly
cmap = plt.get_cmap("gnuplot")

# Generate the plot - "computed_zorder" fixes a bug in matplotlib where the balls constantly appear to be below the surface
fig, ax = plt.subplots( figsize = figsize, subplot_kw={"projection" : "3d", "computed_zorder" : False})

# In order to maintain the same overall simulation speed, as dt decreases and each frame has less effect on the simulation,
# need to add more frames in the same time slot to keep up the same speed - hence reciprocal relationship
fps=int( 1/dt )

# Initialising the x-y coordinate arrays - these will be used to create the coordinate grid
x = np.linspace(-xy_bound,xy_bound, res)
y = -np.linspace(-xy_bound,xy_bound, res)

# X,Y represent the coordinate grids for the figure - this will be used to plot the surface
X,Y = np.meshgrid(x,y)

# z(X,Y) is used to determine the output - the Z-dimension for the given X-Y coordinates
# Use a mask so that it creates a circular-shaped funnel, which is what we want for the simulation
# Any values outside the bound will be ignored by the program, creating a circular shape via the circle cartesian equation
Z = masked_where(X**2 + Y**2 > xy_bound**2, z(X,Y))

# Inverted the x axis because it needs to start at (0,0) - previously it started at (-xy_bound, 0)
ax.invert_xaxis()

# Set these limits so it doesn't change the FOV of our simulation
ax.set_xlim([-xy_bound, xy_bound])
ax.set_ylim(-xy_bound,xy_bound) 

# Disable the axes - they are not relevant to the simulation
ax.set_axis_off()

ax.set_title(f"3D gravity funnel simulation",fontsize=20)

# Create the cylinder that wraps around the 3D funnel - this represents the edges of the simulation
# I made it a cylinder because that's the shape of the black hole simulation in the Space Centre at Leicester
create_cylinder(X,Y,Z)

# Plot the surface of the funnel using our coordinate grids
ax.plot_surface(X,Y,Z, alpha=0.65,cmap=cmap ) # rcount = 400, ccount = 400 

# Initialise the 3 scatter plots - we don't need to plot anything yet
balls = ax.scatter([],[],[], s=50, c="green", zorder=10)
boundary_collision_scatters = ax.scatter([],[],[])
ball_collision_scatters = ax.scatter([],[],[])

# This is not a scatter plot, I just needed a way to represent the information on the legend
info = ax.scatter([],[],[], marker="$?$", c="blue", label=f"μ = {u}, e = {e}")

# The main simulation function - this is called once per frame
# I decided to modularise my code and place all the physics functions in other programs to reduce clutter and huge files
def simulate(t):
    # Declaring all of our global variables - these variables will need to be modified and assigned to by every frame
    global position, velocity, balls, collidedwithwallalready, ball_collision_markers, ball_collision_scatters, balls_collided_already, boundary_collision_markers, boundary_collision_scatters
    
    # Determine the changes in displacement and velocity for the balls
    calculate_motion(position, velocity)
    
    # Update the array of balls that have collided, the markers showing where they collided, and whether new collisions
    # have occurred. If collisions do occur, then this function will enact them as well.
    (balls_collided_already,
     ball_collision_markers, 
     update_ball_markers) = calculate_ball_collisions(position, velocity, balls_collided_already, ball_collision_markers)
    
    # If there's a new collision, we need to update the markers to show where they've occurred.
    if update_ball_markers:
        
        ball_collision_scatters.remove()
        ball_collision_scatters = ball_marker_update(ball_collision_markers)
        
    # Update the array of balls that have collided with the boundary denoted by the cylinder, the markers showing where they 
    # collided, and whether new collisions have occurred. If collisions do occur, then this function will enact them as well.
    (collidedwithwallalready,
     boundary_collision_markers, 
     update_wall_markers) = calculate_wallphysics(position, velocity, collidedwithwallalready, boundary_collision_markers, xy_bound)    
    
    # If there's a new collsiion, we need to update the markers to show where they've occurred.
    if update_wall_markers:
        
        boundary_collision_scatters.remove()
        boundary_collision_scatters = wall_marker_update(boundary_collision_markers)
        
    # Get the new x and y coordinates of the balls after updating their positions
    new_xs = position.real
    new_ys = position.imag
    
    # Remove the current scatter plot of the balls so we can draw them again 
    balls.remove()
    
    # Set the Z-limit (vertical) for our plot. I kept the bottom limit deliberately
    # lower than the actual bottom of the cylinder so it would appear taller, and hence more similar to the simulation.
    ax.set_zlim(Z.min()-0.5, Z.max() + 1 )
    
    # Calculate the new z positions of the balls - these are the vertical positions of the balls on the surface
    # Since one of the assumptions we make is that the balls will never leave the surface once placed
    new_zs= z(new_xs,new_ys)
    balls = ax.scatter(new_xs,new_ys,new_zs + 0.0, s=ball_sizes, c=colours, zorder=10)
    
    # This slowly rotates the view of the plot to give different perspectives
    # I found a good rotation speed at 10 fps, so I multiplied the rate by 10/fps to maintain it regardless of fps
    ax.view_init(elev=30, azim=t * 10/fps)
    
    # I copy-pasted this code from stackoverflow, all it does is ensure that the legend labels always maintain consistent
    # ordering
    handles, labels = ax.get_legend_handles_labels()
    # sort both labels and handles by labels
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    ax.legend(handles, labels, loc="upper left")
    
# Create the simulation for 24 seconds
simulation = FuncAnimation(fig, simulate, interval = 1000/fps , frames=24*fps)

plt.show()
#simulation.save("clips/3dcliptest6.mp4", bitrate=4000, fps=fps)