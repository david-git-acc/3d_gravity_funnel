import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

g=9.81
m=0.01
res = 100
ranging = 5
fps=15

a=0.25
z = lambda x,y : np.log(a+x**2 + y**2)
dz_dx = lambda x,y: 2*x / ( a + x**2 + y**2)
dz_dy = lambda x,y: 2*y / ( a + x**2 + y**2)

ball_vels = np.array([[-2,2,0]]).astype("float64")
ball_poss = np.array([[3.25,3, z(3.25,3) ] ])
x = np.linspace(-ranging,ranging,res)
y = np.linspace(-ranging,ranging,res)

X,Y = np.meshgrid(x,y)


Z = z(X,Y)


u=0.1
z_current = z(ball_poss.real,ball_poss.imag)

# dt = the change in time per frame, measured in seconds.
dt = 0.1

def animate(t):
    
    global ball_poss , ball_vels,z_current,ball_vels
    
    plt.cla()

    xs = ball_poss[:,0]
    ys = ball_poss[:,1]
    zs = ball_poss[:,2]

    dx = dz_dx(xs,ys)
    dy = dz_dy(xs,ys)

    t1 = np.arctan(dx)
    t2 = np.arctan(dy)
    
    a_x, a_y,a_z = [-g * np.sin(t1) * np.cos(t1),
                    -g * np.sin(t2) * np.cos(t2),
                    -g * np.sin(t1) * np.sin(t1)]

    ball_vels[:,0] += a_x*dt
    ball_vels[:,1] += a_y*dt
    ball_vels[:,2] += a_z*dt
        
    xs += ball_vels[:,0]*dt
    ys += ball_vels[:,1]*dt
    zs += ball_vels[:,2]*dt

    ball_poss[:,0] = xs
    ball_poss[:,1] = ys
    ball_poss[:,2] = zs
    
    z_surfacevalues = z(xs,ys)
    
    zs = np.maximum(zs, z_surfacevalues)
    
    ax.plot_surface(X,Y,Z,color="blue",alpha=0.6,zorder=1)
    ax.scatter(xs,ys,zs, s=25, c="green",zorder=10)

fig, ax = plt.subplots(subplot_kw={"projection" : "3d"})

anim = FuncAnimation(fig, animate, interval=1000/fps , frames=500)

plt.show()
#anim.save("testclip3.mp4", fps = fps )