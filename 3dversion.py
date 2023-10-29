from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from numpy.ma import masked_where

position = np.array([2+3j, 3+4j, 2+5j])
velocity = 1.75* np.array([1.25-1.25j,1.5-1.5j, 0.75-1.9j])
m = 0.5
u = 0.1
g = 9.81
a = 0.5
xy_bound = 5
res = 100
px = 1/96

dt=0.01
fps=int( 1/dt )

z = lambda x,y : np.log(a + x**2 + y**2)
dz_dx = lambda x,y : 2*x / (a + x**2 + y**2)
dz_dy = lambda x,y : 2*y / (a + x**2 + y**2)

fig, ax = plt.subplots(figsize = (1280*px, 720*px) , subplot_kw={"projection" : "3d", "computed_zorder" : False})

x = np.linspace(-xy_bound,xy_bound, res)
y = -np.linspace(-xy_bound,xy_bound, res)

X,Y = np.meshgrid(x,y)
Z = z(X,Y)
Z = masked_where(X**2 + Y**2 > xy_bound**2, Z)

theta = np.linspace(0, 2 * np.pi, res)
z_c = np.linspace(Z.min(), Z.max() + 0.25, 100)
theta, z_c = np.meshgrid(theta, z_c)

x_c = xy_bound*np.cos(theta)
y_c = xy_bound*np.sin(theta)

def calculate_physics():
    global position, velocity
    
    xs = position.real
    ys = position.imag
    
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
    


ax.invert_xaxis()

# Plot the surface of the cylinder
ax.plot_surface(x_c, y_c, z_c, color="green", alpha=0.45)

ax.plot_surface(X,Y,Z, color="blue", alpha=0.65, rcount = 500, ccount = 500)

scattering = ax.scatter([],[],[], s=50, c="green", zorder=10)

def animate(t):
    global position, velocity, scattering
    
    calculate_physics()
    
    new_xs = position.real
    new_ys = position.imag
    new_zs= z(new_xs,new_ys)
    
    scattering.remove()
    

    
    ax.set_zlim(Z.min()-0.5, Z.max() + 1 )
    
    scattering = ax.scatter(new_xs,new_ys,new_zs + 0.1, s=50, c="red", zorder=10)
    
    
    
    
anim = FuncAnimation(fig, animate, interval = 1000/fps , frames=10*fps)

#plt.show()
anim.save("3dcliptest4.mp4" , fps=fps)