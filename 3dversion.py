from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from numpy.ma import masked_where

position = 0.75*np.array([2+3j, 3+3.8j, 2+2j])
velocity = 3* np.array([1.25-1.25j,1.5-1j, 0.5-1.9j])
colours = ["yellow","violet","red"]
collidedwithwallalready = np.array([False] * len(position))
m = 500
u = 0#0.025
g = 9.81
a = 0.025
e=0.5
xy_bound = 5
res = 100
px = 1/96

dt=0.1
fps=int( 2/dt )

z = lambda x,y : np.log(a + x**2 + y**2)
dz_dx = lambda x,y : 2*x / (a + x**2 + y**2)
dz_dy = lambda x,y : 2*y / (a + x**2 + y**2)

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
    
    global position, velocity, collidedwithwallalready, collision_scatters
    
    xs,ys = position.real,position.imag
    u_x,u_y = velocity.real, velocity.imag
    
    collided_with_wall = xs ** 2 + ys ** 2 >= xy_bound ** 2


    collidedwithwallalready[collidedwithwallalready & ~collided_with_wall] = False
    just_collided = collided_with_wall & ~collidedwithwallalready
    if np.any(just_collided):
        collision_scatters.remove()
         
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
        
        collision_scatters = ax.scatter(x,y,z(x,y), c="red",s=32,marker="x")
    
    collidedwithwallalready[just_collided] = True

def create_cylinder():
    theta = np.linspace(0, 2 * np.pi, res)
    z_c = np.linspace(Z.min()-3, Z.max() + 0.25, 100)
    theta, z_c = np.meshgrid(theta, z_c)

    x_c = xy_bound*np.cos(theta)
    y_c = xy_bound*np.sin(theta)
    
    # Plot the surface of the cylinder
    ax.plot_surface(x_c, y_c, z_c, color="green", alpha=0.45)

ax.invert_xaxis()
ax.set_xlim([-xy_bound, xy_bound])
ax.set_ylim(-xy_bound,xy_bound)

create_cylinder()

ax.plot_surface(X,Y,Z, color="blue", alpha=0.65)

scattering = ax.scatter([],[],[], s=50, c="green", zorder=10)
collision_scatters = ax.scatter([],[],[])
    


def animate(t):
    global position, velocity, scattering, collidedwithwallalready, collision_scatters
    
    calculate_motion()
    calculate_wallphysics()
    
    
    new_xs = position.real
    new_ys = position.imag
    
    scattering.remove()
    
    
    ax.set_zlim(Z.min()-0.5, Z.max() + 1 )
    
    new_zs= z(new_xs,new_ys)
    scattering = ax.scatter(new_xs,new_ys,new_zs + 0.1, s=50, c=colours, zorder=10)
    
    
    
    
anim = FuncAnimation(fig, animate, interval = 1000/fps , frames=10*fps)

plt.show()
#anim.save("3dcliptest4.mp4" , fps=fps)