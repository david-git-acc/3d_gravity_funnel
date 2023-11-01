from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from numpy.ma import masked_where

position = 0.75*np.array([2+3j, 3-3.8j, 1-1j, 3+3j])
velocity = 4* np.array([1-1.25j,-1-0.75j, -0.9+1.1j, 0.76+1.2j])
colours = ["yellow","green","blue", "black"] # original choice was ["yellow","violet","red"]
collidedwithwallalready = np.array([False] * len(position))
collidedtogetheralready = np.array([])
boundary_collision_markers = np.array([])
ball_collision_markers = np.array([])
sizes = 3*np.array([32, 70, 42, 42]) # 50
radii = np.sqrt(sizes) * 0.01
cmap = plt.get_cmap("gnuplot")

point_info = {
    "marker colour" :  ["red","red"],
    "marker" : ["D","x"],
    "size" : [16,32],
    "label" : ["Ball collision points", "Boundary collision points"]
}

m = np.copy(sizes)
u = 0.05
g = 9.81
a = 0.5
e = 0.8
xy_bound = 5
res = 100
px = 1/96


figsize = (1280*px, 1080*px)
dt=0.012
fps=int( 1/dt )

z = lambda x,y : np.log(a + x**2 + y**2)
dz_dx = lambda x,y : 2*x / (a + x**2 + y**2)
dz_dy = lambda x,y : 2*y / (a + x**2 + y**2)

primecomposition = lambda a,b : np.log(b) / np.log(2) + a

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
    global velocity, collidedtogetheralready, ball_collision_markers, ball_collision_scatters
        
    collisions = find_ball_collisions()

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
            
            collision_points = (position[unique_collisions[:,0]] + position[unique_collisions[:,1]])/2
        
            ball_collision_markers = np.append(ball_collision_markers, collision_points)
            
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

ax.set_title(f"3D gravity funnel simulation",fontsize=20)

create_cylinder()

ax.plot_surface(X,Y,Z, alpha=0.65,cmap=cmap, rcount = 400, ccount = 400 ) 

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

#plt.show()
anim.save("3dcliptest6.mp4", bitrate=4000, fps=fps)