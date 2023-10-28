import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

g=9.81
m=0.01
res = 100
ranging = 5
fps=15

ball_vels = np.array([-0.25+0.25j, -0.32 +0.18j, -0.1+0.1j,-0.25+0.25j]) / (4*5/3*fps / 100)
ball_poss = np.array([3.25+3j, 2-2j, 1-4j, 2.75+2.5j  ])
x = np.linspace(-ranging,ranging,res)
y = np.linspace(-ranging,ranging,res)

X,Y = np.meshgrid(x,y)

a=0.25
z = lambda x,y : np.log(a+x**2 + y**2)
dz_dx = lambda x,y: 2*x / ( a + x**2 + y**2)
dz_dy = lambda x,y: 2*y / ( a + x**2 + y**2)

Z = z(X,Y)


u=0.1
k1=25/fps
k2=0.99
z_current = z(ball_poss.real,ball_poss.imag)

def animate(t):
    
    global ball_poss , ball_vels,z_current,ball_vels
    
    plt.cla()

    xs = ball_poss.real
    ys = ball_poss.imag

    dx = dz_dx(xs,ys)
    dy = dz_dy(xs,ys)

    t1 = np.arctan(dx)
    t2 = np.arctan(dy)
    
    a_x, a_y = [g*np.sin(t1)*np.cos(t1),
                g*np.sin(t2)*np.cos(t2)]

    ball_vels += -a_x / (4*5/3*fps)
    ball_vels += 1j*-a_y / (4*5/3*fps)

    ball_poss += ball_vels
    
    new_xs = ball_poss.real
    new_ys = ball_poss.imag
    
    z_current = z(new_xs,new_ys)
    
    ax.plot_surface(X,Y,Z,color="blue",alpha=0.6,zorder=1)
    ax.scatter(new_xs, new_ys,z_current, s=25, c="green",zorder=10)

fig, ax = plt.subplots(subplot_kw={"projection" : "3d"})

anim = FuncAnimation(fig, animate, interval=1000/fps , frames=500)

plt.show()
#anim.save("testclip3.mp4", fps = fps )