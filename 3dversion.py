from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from numpy.ma import masked_where

position = 0.75*np.array([2+3j, 3+3.8j, 2+2j])
velocity = 4* np.array([1.25-1.25j,1.5-1j, 0.5-1.9j])
colours = ["yellow","green","blue"] # original choice was ["yellow","violet","red"]
collidedwithwallalready = np.array([False] * len(position))
collidedtogetheralready = np.array([])
boundary_collision_markers = np.array([])
ball_collision_markers = np.array([])
cmap = plt.get_cmap("gnuplot")

point_info = {
    "marker colour" :  ["red","red"],
    "marker" : ["D","x"],
    "size" : [16,32],
    "label" : ["Ball collision points", "Boundary collision points"]
}

m = 1   
u = 0.025
g = 9.81
a = 0.5
e=0.5
xy_bound = 5
res = 100
px = 1/96

dt=0.01
fps=int( 1/dt )

z = lambda x,y : np.log(a + x**2 + y**2)
dz_dx = lambda x,y : 2*x / (a + x**2 + y**2)
dz_dy = lambda x,y : 2*y / (a + x**2 + y**2)

primecomposition = lambda a,b : np.log(b) / np.log(2) + a

#figsize = (1280*px, 720*px)
fig, ax = plt.subplots(  subplot_kw={"projection" : "3d", "computed_zorder" : False})

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
    norm = np.sqrt(zx_gradient ** 2 + zy_gradient **2 + 1)

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
    
    xs,ys = position.real,position.imag
    u_x,u_y = velocity.real, velocity.imag
    
    collided_with_wall = xs ** 2 + ys ** 2 >= xy_bound ** 2


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
    
def find_duplicates(records_array, decimals):
    
    records_array = np.round(records_array * decimals) / decimals

    # creates an array of indices, sorted by unique element
    idx_sort = np.argsort(records_array)

    # sorts records array so all unique elements are together 
    sorted_records_array = records_array[idx_sort]

    # returns the unique values, the index of the first occurrence of a value, and the count for each element
    vals, idx_start, count = np.unique(sorted_records_array, return_counts=True, return_index=True)

    # splits the indices into separate arrays
    res = np.split(idx_sort, idx_start[1:])

    #filter them with respect to their size, keeping only items occurring more than once
    vals = vals[count > 1]
    res = filter(lambda x: x.size > 1, res)

    return np.array([x[0:2] for x in res])
    
def collision(i,j):
    global velocity

    u1 = velocity[i]
    u_1x, u_1y = u1.real, u1.imag
    
    u2 = velocity[j]
    u_2x, u_2y = u2.real, u2.imag

    v_1x = e * (u_1x - u_2x)
    v_2x = e * (u_2x - u_1x)
    
    v_1y = e * (u_1y - u_2y)
    v_2y = e * (u_2y - u_1y)
    
    v_1 = v_1x + v_1y * 1j
    v_2 = v_2x + v_2y * 1j
    
    velocity[i] = v_1
    velocity[j] = v_2    
    
def check_collisions():
    global velocity, collidedtogetheralready, ball_collision_markers, ball_collision_scatters
        
    collisions = find_duplicates(position, 1.5)

    if collisions.size:
        
        colliding_object_As = collisions[:,0]
        colliding_object_Bs = collisions[:,1]
    
        if collidedtogetheralready.size:
        
            collision_ids = primecomposition(colliding_object_As, colliding_object_Bs)
            already_collided_ids = primecomposition(collidedtogetheralready[:,0], collidedtogetheralready[:,1])
    
            not_yet_collided = ~np.isin(collision_ids, already_collided_ids)
            
            unique_collisions = collisions[not_yet_collided]
            
            if unique_collisions.size:
            
                collision(unique_collisions[:,0], unique_collisions[:,1])
                    
            still_colliding = np.isin(already_collided_ids, collision_ids)
            
            collidedtogetheralready = collidedtogetheralready[still_colliding]
            
        else:
            
            unique_collisions = collisions
            collision(colliding_object_As, colliding_object_Bs)
    
        
        if collidedtogetheralready.size:
            collidedtogetheralready = np.append(collidedtogetheralready, unique_collisions,axis=0)
        else:
            collidedtogetheralready = unique_collisions
        
        if unique_collisions.size:
            
            ball_collision_scatters.remove()
        
            ball_collision_markers = np.append(ball_collision_markers, position[unique_collisions[:,0]])
            
            collisions_x, collisions_y = ball_collision_markers.real, ball_collision_markers.imag
            
            ball_collision_scatters = ax.scatter(collisions_x,collisions_y, z(collisions_x, collisions_y),
                                                 c=point_info["marker colour"][0],
                                                 s=point_info["size"][0],
                                                 marker=point_info["marker"][0],
                                                 label=point_info["label"][0])
        
    else:
        if collidedtogetheralready.size:
            collidedtogetheralready = np.array([])
            
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

ax.set_title(f"3D simulation")

create_cylinder()

ax.plot_surface(X,Y,Z, alpha=0.65,cmap=cmap)

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



def animate(t):
    global position, velocity, scattering, collidedwithwallalready, boundary_collision_scatters
    
    calculate_motion()
    
    check_collisions()
    
    calculate_wallphysics()

    
    
    new_xs = position.real
    new_ys = position.imag
    
    scattering.remove()
    
    
    ax.set_zlim(Z.min()-0.5, Z.max() + 1 )
    
    new_zs= z(new_xs,new_ys)
    scattering = ax.scatter(new_xs,new_ys,new_zs + 0.0, s=50, c=colours, zorder=10)
    
    
    ax.view_init(elev=30, azim=t * 10/fps)
    
    handles, labels = ax.get_legend_handles_labels()
    # sort both labels and handles by labels
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    ax.legend(handles, labels)
    
    
anim = FuncAnimation(fig, animate, interval = 1000/fps , frames=12*fps)

#plt.show()
anim.save("3dcliptest4.mp4" , fps=fps)