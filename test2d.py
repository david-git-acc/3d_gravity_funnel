import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

x = np.array([-6]).astype("float64")
v = np.array([7]).astype("float64")
bound = 10

a=0.25
u=0.3
m=0.5
g=9.81
y = lambda x : np.log(a+x**2)
y_initial = y(x)
dy_dx = lambda x : 2*x / (a+x**2)
dt=0.01
fps=int(1/dt)
res=75

X = np.linspace(-bound,bound,res)

fig, ax = plt.subplots()

def animate(t):
    global x, v
    
    gradient = dy_dx(x)
    theta = np.arctan(gradient)
    
    friction = u*m*g*np.cos(theta)
    
    parallel_force = m*g*np.sin(theta) 
    sign = np.sign(v)
    
    parallel_force += sign * friction
    
    horizontal = parallel_force*np.cos(theta)
    
    a = horizontal / m
    
    v += -a*dt
    x += v*dt
    
    new_positions = y(x)
    
    plt.cla()
    
    plt.plot(X, y(X) , color="blue")
    
    plt.scatter(x, new_positions, c="green" , s=25)    

    
    
anim = FuncAnimation(fig, animate, frames=500, interval = 500 / fps)

plt.show()
#anim.save("2dcliptest.mp4", fps=fps)    
    

