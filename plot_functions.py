import numpy as np
import matplotlib.pyplot as plt
from plot_constants import point_info, res , xy_bound
from physics_functions import z
from numpy.ma import masked_where

# STORING ALL THE FUNCTIONS RELATED TO THE PLOT

# Functions to update the collision markers after a ball/wall collision has occurred
def ball_marker_update(ball_collision_markers):
    
        # Get the collision points
        collisions_x, collisions_y = ball_collision_markers.real, ball_collision_markers.imag
            
        # Create the scatter points on the current axis (can't get them as a parameter due to Python addressing issues)
        return plt.gcf().get_axes()[0].scatter(collisions_x,collisions_y, z(collisions_x, collisions_y),
                                                 c=point_info["marker colour"][0],
                                                 s=point_info["size"][0],
                                                 marker=point_info["marker"][0],
                                                 label=point_info["label"][0])
        
def wall_marker_update(boundary_collision_markers):     
         
        # Get the collision points
        collision_x,collision_y = boundary_collision_markers.real, boundary_collision_markers.imag
        
        # Create the scatter points on the current axis (can't get them as a parameter due to Python addressing issues)        
        return plt.gcf().get_axes()[0].scatter(collision_x,collision_y,z(collision_x,collision_y), 
                                                 c=point_info["marker colour"][1],
                                                 s=point_info["size"][1],
                                                 marker=point_info["marker"][1], 
                                                 label=point_info["label"][1])
        
# Create the cylinder that extends/wraps around the main funnel suface. 
def create_cylinder(X,Y,Z):
    
        # Get the current axis to place the cylinder around
        ax = plt.gca()

        # Get the bottom and top of the cylinder - make them extend beyond the min and max to make it look more
        # realistic to the shape of the original black hole simulator in the Space Centre
        bottom = Z.min()-2
        top = Z.max() + 0.25

        # I have no idea how this code works, I just copied it from chatGPT
        theta = np.linspace(0, 2 * np.pi, res)
        z_c = np.linspace(bottom, top, 100)
        theta, z_c = np.meshgrid(theta, z_c)

        # Ditto as above
        x_c = xy_bound*np.cos(theta)
        y_c = xy_bound*np.sin(theta)

        # This code I do understand. Create the plane at the bottom of the plot, then only show the points
        # within the cylinder to create the filled circle
        cylinder_bottom = masked_where(X**2 + Y**2 > xy_bound**2 + 0.48 , np.ones_like(X)*bottom )

        # Plot the bottom of the cylinder
        ax.plot_surface(X,Y,cylinder_bottom, color="violet" , alpha=0.45)

        # Plot the surface of the cylinder
        ax.plot_surface(x_c, y_c, z_c, color="violet", alpha=0.45)
        
def prepare_KE_ax(KE_ax, fps, t):
    KE_ax.set_xlim([0, t+1])
    revised_xlabels = np.round( KE_ax.get_xticks() / fps, decimals=2) 
    KE_ax.set_xticks(KE_ax.get_xticks())
    KE_ax.set_xticklabels(revised_xlabels)
    KE_ax.set_title("Ball kinetic energy over time")
    KE_ax.set_xlabel("Time (s)")
    KE_ax.set_ylabel("Kinetic energy (J)")
    KE_ax.set_ylim(0)
    
def prepare_energy_ax(total_energy_ax, fps, t):
    total_energy_ax.set_xlim([0,t+1])
    revised_xlabels = np.round( total_energy_ax.get_xticks() / fps, decimals=2) 
    total_energy_ax.set_xticks(total_energy_ax.get_xticks())
    total_energy_ax.set_xticklabels(revised_xlabels)
    total_energy_ax.set_title("Total system energy over time")
    total_energy_ax.set_xlabel("Time (s)")
    total_energy_ax.set_ylabel("Total energy (J)")
    total_energy_ax.set_ylim(0)