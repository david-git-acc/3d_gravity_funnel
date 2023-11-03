import numpy as np
import matplotlib.pyplot as plt
from plot_constants import point_info, res , xy_bound
from physics_functions import z
from numpy.ma import masked_where

def ball_marker_update(ball_collision_markers):
    
        collisions_x, collisions_y = ball_collision_markers.real, ball_collision_markers.imag
            
        return plt.gca().scatter(collisions_x,collisions_y, z(collisions_x, collisions_y),
                                                 c=point_info["marker colour"][0],
                                                 s=point_info["size"][0],
                                                 marker=point_info["marker"][0],
                                                 label=point_info["label"][0])
        
def wall_marker_update(boundary_collision_markers):      
        collision_x = boundary_collision_markers.real
        collision_y = boundary_collision_markers.imag
        
        return plt.gca().scatter(collision_x,collision_y,z(collision_x,collision_y), 
                                                 c=point_info["marker colour"][1],
                                                 s=point_info["size"][1],
                                                 marker=point_info["marker"][1], 
                                                 label=point_info["label"][1])
        
def create_cylinder(X,Y,Z):
    
    ax = plt.gca()
    
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