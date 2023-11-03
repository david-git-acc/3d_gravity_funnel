from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from numpy.ma import masked_where

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

# Colours of the balls
colours = ["yellow","green","blue", "black"] # original choice was ["yellow","violet","red"]

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

# The sizes of the balls - the balls will be visualised as scatter points for efficiency of both design and performance
sizes = 3*np.array([32, 70, 42, 42]) # 50

# Calculate approximately the radii of the balls - used to identify collisions
radii = np.sqrt(sizes) * 0.01

# The colour map we'll use for the 3D surface to show higher and lower altitude points more clearly
cmap = plt.get_cmap("gnuplot")

# This dict stores information about both the wall and ball collision markers - centralised here for ease of editing
point_info = {
    "marker colour" :  ["red","red"],
    "marker" : ["D","x"],
    "size" : [16,32],
    "label" : ["Ball collision points", "Boundary collision points"]
}

# The masses of the balls will be directly proportional to their sizes - same density
m = np.copy(sizes)

# Mew (u since you can't type greek characters) is the coefficient of friction
# 0 <= u <= 1, u = 0 -> no friction, u = 1 -> extreme friction
u = 0.05

# small g, strength of gravitational force on earth in ms^-2
g = 9.81

# This constant determines the depth of the depression in the middle of the funnel - the lower the a value, the deeper (min 0)
a = 0.5

# e, the coefficient of restitution. Determines how much speed of the ball is lost upon collision with either a wall or ball
# 0 <= e <= 1, 0 -> total speed loss upon collisions, 1 -> perfectly elastic collisions
e = 0.8

# this determines the area of the surface. The surface size will be a square with sides of length xy_bound
xy_bound = 5

# resolution of the plot - there will be res**2 points used to plot the surface, recommend at least res=50
res = 100

# Ratio between inches and pixels for a 1920x1080 screen
px = 1/96

# I didn't use 1920x1080 because there'd be too much empty space by the sides, want the viewers to focus on the plot 
figsize = (1280*px, 1080*px)

# Size of the time step. As dt -> 0, the more accurate and stable the simulation (as real time is continuous)
# A large dt will produce noticeably discrete movements and decrease simulation stability
dt=0.1

# In order to maintain the same overall simulation speed, as dt decreases and each frame has less effect on the simulation,
# need to add more frames in the same time slot to keep up the same speed - hence reciprocal relationship
fps=int( 1/dt )

# This is our 3D function that will plot the actual funnel surface - it's cone shaped.
z = lambda x,y : np.log(a + x**2 + y**2)

# These are the partial derivatives of z with respect to x and y; necessary for computing slope and forces acting on the balls
dz_dx = lambda x,y : 2*x / (a + x**2 + y**2)
dz_dy = lambda x,y : 2*y / (a + x**2 + y**2)

# This function maps a pair of numbers into a single number in a bijective way, using the base2 logarithm of prime composition
# Needed for checking for numpy array membership - the standard np.isin() only works as intended on 1D arrays
primecomposition = lambda a,b : np.log(b) / np.log(2) + a

# Generate the plot - "computed_zorder" fixes a bug in matplotlib where the balls constantly appear to be below the surface
fig, ax = plt.subplots( figsize = figsize, subplot_kw={"projection" : "3d", "computed_zorder" : False})

x = np.linspace(-xy_bound,xy_bound, res)
y = -np.linspace(-xy_bound,xy_bound, res)

X,Y = np.meshgrid(x,y)
Z = z(X,Y)
Z = masked_where(X**2 + Y**2 > xy_bound**2, Z)

def calculate_motion():
    global position, velocity
    
    xs,ys = position.real, position.imag
    
    zx_gradient = dz_dx(xs,ys)
    zy_gradient = dz_dy(xs,ys)
       
    gradient_vector = np.array([zx_gradient, zy_gradient, np.ones_like(zx_gradient)])
    norm = np.linalg.norm(gradient_vector,axis=0)
    
    gradient_vector *= m*g / norm    
    velocity_direction_x = np.sign(velocity.real)
    velocity_direction_y = np.sign(velocity.imag)

    theta_zx = np.arctan(zx_gradient)
    theta_zy = np.arctan(zy_gradient)
    
    parallel_forces = gradient_vector[0] + 1j*gradient_vector[1]
    frictional_forces = u * gradient_vector[2] * ( velocity_direction_x + 1j*velocity_direction_y )  

    parallel_forces += frictional_forces    
    
    parallel_forces.real *= np.cos(theta_zx)
    parallel_forces.imag *= np.cos(theta_zy)
    
    a = parallel_forces / m
    
    velocity += -a*dt
    
    position += velocity*dt
    

def calculate_wallphysics():
    
    global position, velocity, collidedwithwallalready, boundary_collision_scatters, boundary_collision_markers
    
    u_x,u_y = velocity.real, velocity.imag
    
    collided_with_wall = np.absolute(position) >= xy_bound

    collidedwithwallalready[collidedwithwallalready & ~collided_with_wall] = False
    just_collided = collided_with_wall & ~collidedwithwallalready
    if np.any(just_collided):
        boundary_collision_scatters.remove()
         
        x,y = position[just_collided].real,position[just_collided].imag
        u_x,u_y = velocity[just_collided].real, velocity[just_collided].imag
        
        gradient = -x/y
        theta = np.arctan(gradient)
        
        angle = 2*np.pi - theta
        
        rotated_u_x, rotated_u_y = [u_x * np.cos(angle) - u_y * np.sin(angle),
                                    u_x * np.sin(angle) + u_y * np.cos(angle) ]
        
        rotated_u_y *= -e
        
        v_x, v_y = [rotated_u_x * np.cos(theta) - rotated_u_y * np.sin(theta),
                    rotated_u_x * np.sin(theta) + rotated_u_y * np.cos(theta) ]
    
    
        velocity[just_collided] = v_x + 1j * v_y
        
        boundary_collision_markers = np.append(boundary_collision_markers,x+1j*y )
        
        collision_x = boundary_collision_markers.real
        collision_y = boundary_collision_markers.imag
        
        boundary_collision_scatters = ax.scatter(collision_x,collision_y,z(collision_x,collision_y), 
                                                 c=point_info["marker colour"][1],
                                                 s=point_info["size"][1],
                                                 marker=point_info["marker"][1], 
                                                 label=point_info["label"][1])
        
    
    collidedwithwallalready[just_collided] = True
    
def collision(A,B):
    global velocity

    u1 = velocity[A]
    u_1x, u_1y = u1.real, u1.imag
    m1 = m[A]
    
    u2 = velocity[B]
    u_2x, u_2y = u2.real, u2.imag
    m2 = m[B]
    
    v_1x = ( u_1x * (m1-e*m2) + m2*u_2x*(1+e) ) / (m1 + m2)
    v_2x = (m1*u_1x + m2*u_2x - m1*v_1x) / m2
    
    v_1y = ( u_1y * (m1-e*m2) + m2*u_2y*(1+e) ) / (m1 + m2)
    v_2y = (m1*u_1y + m2*u_2y - m1*v_1y) / m2

    v_1 = v_1x + v_1y * 1j
    v_2 = v_2x + v_2y * 1j
    
    velocity[A] = v_1
    velocity[B] = v_2    
    
def find_ball_collisions():
    
    zs = z(position.real, position.imag)
    
    object_position_A, object_position_B = np.meshgrid(position , position)
    radiusA, radiusB = np.meshgrid(radii, radii)
    z_A, z_B = np.meshgrid(zs, zs)
    
    sum_radii = radiusA + radiusB
    xy_diff = object_position_A - object_position_B
    z_diff = z_A - z_B 
    
    distances = np.linalg.norm([xy_diff.real,xy_diff.imag,z_diff],axis=0)

    collisions = np.argwhere( distances <= sum_radii )
    collisions = collisions[collisions[:,0] < collisions[:,1]]
    
    return collisions
    
    
def calculate_collisions():
    global velocity, balls_collided_already, ball_collision_markers, ball_collision_scatters
        
    collisions = find_ball_collisions()

    if collisions.size:
        
        colliding_object_As = collisions[:,0]
        colliding_object_Bs = collisions[:,1]
    
        if balls_collided_already.size:
        
            collision_ids = primecomposition(colliding_object_As, colliding_object_Bs)
            already_collided_ids = primecomposition(balls_collided_already[:,0], balls_collided_already[:,1])
    
            not_yet_collided = ~np.isin(collision_ids, already_collided_ids)
            
            unique_collisions = collisions[not_yet_collided]
            
            if unique_collisions.size:
            
                collision(unique_collisions[:,0], unique_collisions[:,1])
                    
            still_colliding = np.isin(already_collided_ids, collision_ids)
            
            balls_collided_already = balls_collided_already[still_colliding]
            
        else:
            
            unique_collisions = collisions
            collision(colliding_object_As, colliding_object_Bs)
    
        
        if balls_collided_already.size:
            balls_collided_already = np.append(balls_collided_already, unique_collisions,axis=0)
        else:
            balls_collided_already = unique_collisions
        
        if unique_collisions.size:
            
            ball_collision_scatters.remove()
            
            collision_points = (position[unique_collisions[:,0]] + position[unique_collisions[:,1]])/2
        
            ball_collision_markers = np.append(ball_collision_markers, collision_points)
            
            collisions_x, collisions_y = ball_collision_markers.real, ball_collision_markers.imag
            
            ball_collision_scatters = ax.scatter(collisions_x,collisions_y, z(collisions_x, collisions_y),
                                                 c=point_info["marker colour"][0],
                                                 s=point_info["size"][0],
                                                 marker=point_info["marker"][0],
                                                 label=point_info["label"][0])
            
    else:
        if balls_collided_already.size:
            balls_collided_already = np.array([])
            
def create_cylinder():
    
    bottom = Z.min()-2
    top = Z.max() + 0.25
    
    theta = np.linspace(0, 2 * np.pi, res)
    z_c = np.linspace(bottom, top, 100)
    theta, z_c = np.meshgrid(theta, z_c)

    x_c = xy_bound*np.cos(theta)
    y_c = xy_bound*np.sin(theta)
    
    cylinder_bottom = masked_where(X**2 + Y**2 > xy_bound**2 + 0.48 , np.ones_like(X)*bottom )
    
    ax.plot_surface(X,Y,cylinder_bottom, color="violet" , alpha=0.45)
    
    # Plot the surface of the cylinder
    ax.plot_surface(x_c, y_c, z_c, color="violet", alpha=0.45)




ax.invert_xaxis()
ax.set_xlim([-xy_bound, xy_bound])
ax.set_ylim(-xy_bound,xy_bound) 
ax.set_axis_off()

ax.set_title(f"3D gravity funnel simulation",fontsize=20)

create_cylinder()

ax.plot_surface(X,Y,Z, alpha=0.65,cmap=cmap ) # rcount = 400, ccount = 400

scattering = ax.scatter([],[],[], s=50, c="green", zorder=10)
boundary_collision_scatters = ax.scatter([],[],[],                                                 
                                        c=point_info["marker colour"][1],
                                        s=point_info["size"][1],
                                        marker=point_info["marker"][1], 
                                        label=point_info["label"][1])
ball_collision_scatters = ax.scatter([],[],[],                                                 
                                    c=point_info["marker colour"][0],
                                    s=point_info["size"][0],
                                    marker=point_info["marker"][0], 
                                    label=point_info["label"][0])


info = ax.scatter([],[],[], marker="$?$", c="blue", label=f"μ = {u}, e = {e}")

def animate(t):
    global position, velocity, scattering, collidedwithwallalready
    
    calculate_motion()
    calculate_collisions()
    calculate_wallphysics()

    new_xs = position.real
    new_ys = position.imag
    
    scattering.remove()
    
    
    ax.set_zlim(Z.min()-0.5, Z.max() + 1 )
    
    new_zs= z(new_xs,new_ys)
    scattering = ax.scatter(new_xs,new_ys,new_zs + 0.0, s=sizes, c=colours, zorder=10)
    
    
    ax.view_init(elev=30, azim=t * 10/fps)
    
    handles, labels = ax.get_legend_handles_labels()
    # sort both labels and handles by labels
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    ax.legend(handles, labels, loc="upper left")
    
    
anim = FuncAnimation(fig, animate, interval = 1000/fps , frames=24*fps)

plt.show()
#anim.save("3dcliptest6.mp4", bitrate=4000, fps=fps)