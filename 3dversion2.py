from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from numpy.ma import masked_where
from physics_constants import u,e,dt, ball_sizes
from physics_functions import z, calculate_motion, calculate_collisions, calculate_wallphysics
from plot_constants import figsize,res,xy_bound, colours
from plot_functions import wall_marker_update, ball_marker_update, create_cylinder

# The purpose of this program is to create a 3D simulation of balls travelling along a funnel surface
# The model is based off the black hole simulation at the National Space Centre in Leicester, UK
# After running the simulation, the goal is to collect data and results about the balls such as their
# kinetic energy, number of collisions, etc and vary the settings of the simulation to analyse it 
# and find conclusions about how mainly the friction, coefficient of restitution but also other
# factors such as collision frequency affect the kinetic energy,velocity, etc over time of these balls.

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

x = np.linspace(-xy_bound,xy_bound, res)
y = -np.linspace(-xy_bound,xy_bound, res)

X,Y = np.meshgrid(x,y)
Z = masked_where(X**2 + Y**2 > xy_bound**2, z(X,Y))

ax.invert_xaxis()
ax.set_xlim([-xy_bound, xy_bound])
ax.set_ylim(-xy_bound,xy_bound) 
ax.set_axis_off()

ax.set_title(f"3D gravity funnel simulation",fontsize=20)

create_cylinder(X,Y,Z)

ax.plot_surface(X,Y,Z, alpha=0.65,cmap=cmap ) # rcount = 400, ccount = 400 

balls = ax.scatter([],[],[], s=50, c="green", zorder=10)
boundary_collision_scatters = ax.scatter([],[],[])
ball_collision_scatters = ax.scatter([],[],[])


info = ax.scatter([],[],[], marker="$?$", c="blue", label=f"μ = {u}, e = {e}")

def animate(t):
    global position, velocity, balls, collidedwithwallalready, ball_collision_markers, ball_collision_scatters, balls_collided_already, boundary_collision_markers, boundary_collision_scatters
    
    calculate_motion(position, velocity)
    
    (balls_collided_already,
     ball_collision_markers, 
     update_ball_markers) = calculate_collisions(position, velocity, balls_collided_already, ball_collision_markers)
    
    if update_ball_markers:
        
        ball_collision_scatters.remove()
        ball_collision_scatters = ball_marker_update(ball_collision_markers)
        
    (collidedwithwallalready,
     boundary_collision_markers, 
     update_wall_markers) = calculate_wallphysics(position, velocity, collidedwithwallalready, boundary_collision_markers, xy_bound)    
    
    if update_wall_markers:
        
        boundary_collision_scatters.remove()
        boundary_collision_scatters = wall_marker_update(boundary_collision_markers)
        
    
    new_xs = position.real
    new_ys = position.imag
    
    balls.remove()
    
    ax.set_zlim(Z.min()-0.5, Z.max() + 1 )
    
    new_zs= z(new_xs,new_ys)
    balls = ax.scatter(new_xs,new_ys,new_zs + 0.0, s=ball_sizes, c=colours, zorder=10)
    
    ax.view_init(elev=30, azim=t * 10/fps)
    
    handles, labels = ax.get_legend_handles_labels()
    # sort both labels and handles by labels
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    ax.legend(handles, labels, loc="upper left")
    
    
anim = FuncAnimation(fig, animate, interval = 1000/fps , frames=24*fps)

plt.show()
#anim.save("clips/3dcliptest6.mp4", bitrate=4000, fps=fps)