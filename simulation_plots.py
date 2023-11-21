from simulation import simulate, fps, position, velocity, friction, collided_this_frame
from physics_functions import kinetic_energy, gp_energy, calculate_KE, z, z_min, dz_dx, dz_dy
from physics_constants import dt,m, e
from plot_constants import number_of_balls, colours
from plot_functions import prepare_KE_ax, prepare_energy_ax

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

# This program takes the simulation from simulation.py and in addition, provides 2 graphs that measure the kinetic
# and gravitational potential energy of all the balls (in Joules). It builds on simulation.py, so we can just 
# use anything defined there in here.

# Number of seconds to run the tests for
seconds = 30

# Number of frames to run for
frame_limit = seconds*fps

# Generate the x-axis that we will plot our data on
stats_xaxis = np.arange(0, frame_limit)


# Number of ticks to be used on the x-axis
xaxis_ticks = 5

# Calculate initial kinetic energy
KE = np.zeros((frame_limit, number_of_balls) )
GPE = np.zeros((frame_limit, number_of_balls) )
position_history = np.copy(KE).astype("complex128")

# Initialising the totals
KE_total = np.zeros(frame_limit)
GPE_total = np.zeros(frame_limit)
true_total =np.zeros(frame_limit)

# Get the figure generated from the simulation
fig = plt.gcf()

# Create the axes for our kinetic energy plot
KE_ax = fig.add_axes([0.5, 0.55, 0.375, 0.375])
total_energy_ax = fig.add_axes([0.5, 0.1, 0.375, 0.375])

# Initialising everything at t = 0. I could've probably incorporated this into the main plot_simulation() function, but
# I wanted to make it clearer.

# We need to also consider the vertical velocity at the beginning or else we won't get all the kinetic energy.
velocity_z_initial = dz_dx(position.real, position.imag) * velocity.real + dz_dy(position.real, position.imag) * velocity.imag
KE[0] = kinetic_energy(m,[velocity.real, velocity.imag, velocity_z_initial])
KE_total[0] = KE[0].sum()
GPE[0] = gp_energy(z(position.real, position.imag), z_min)
GPE_total[0] = GPE[0].sum()
true_total[0] = KE_total[0] + GPE_total[0]
    
# The animation function 
def plot_simulation(t):
    global KE, KE_total, GPE_total, true_total
    
    position_history[t] = position
    
    # Get the velocity before this tick - useful in calculating collsions
    prior_velocity = np.copy(velocity)
    
    # Perform the simulation itself to get the new updated values we need
    simulate(t)
    
    # GPE is always relative to the bottom 
    GPE[t] = gp_energy(z(position.real, position.imag), z_min)
    GPE_total[t] = GPE[t].sum()
    
    # Calculate the kinetic energy. This is more complicated so I've included it in physics functions
    # KE: This records the KE - we will store the result here and use the previous KE to calculate this KE
    # KE_total is just the sum of the kinetic energies at some time t
    # GPE is analogous to KE
    # velocity and prior velocity - useful for calculating the new vertical velocity
    # Need the position and position history to calculate the distance travelled by the balls, used for
    # calculating energy losses from friction
    # friction = the force of friction
    calculate_KE(KE, KE_total, GPE, 
                 velocity, prior_velocity, 
                 collided_this_frame, 
                 position, position_history, 
                 friction, t)
    
    # The total energy of the system is just the sum, since no other energy sources
    true_total[t] = KE_total[t] + GPE_total[t]
    
    # Clear the previous axes so we can plot the next graph
    # Inefficient since we're only adding 1 point each time, but performance impact minimal compared to other parts
    # e.g drawing the figures themselves and the simulation
    KE_ax.cla()
    total_energy_ax.cla()
    
    # The x-axis always goes from the beginning to the current point in time, since that's all the data we have so far
    time_xaxis = stats_xaxis[0:t+1]
    
    # Plotting the KE of each ball individually
    for ball in range(number_of_balls):
        
        # Get the kinetic energy of the ball up to now
        # Need to access columnwise given the shape of the array
        KE_this_ball = KE[:, ball][0:t+1:]
    
        # Plot it, using the colour of the ball as the identifier
        KE_ax.plot(time_xaxis, KE_this_ball , color=colours[ball])
    
    # Plot the energy lines, I've given them colours that I believe best symbolise the quantities
    # e.g GPE = brown since it's like earth, gold in KE since it's moving and vibrant, purple as it's 
    # the sum of all the energy, like the "emperor" of all the energy
    total_energy_ax.plot(time_xaxis, GPE_total[0:t+1:], color="brown")
    total_energy_ax.plot(time_xaxis, KE_total[0:t+1:] , color="gold")
    total_energy_ax.plot(time_xaxis, true_total[0:t+1:] , color="purple")
    
    # These functions just set the x-axis labels, titles, etc
    prepare_KE_ax(KE_ax,fps,t)
    prepare_energy_ax(total_energy_ax,fps,t)
    
    # Provide the legend to show which colours refer to what
    total_energy_ax.legend(["Total GP energy","Total kinetic energy","Total system energy"], loc="lower left")

       
# Run the simulation
simulation = FuncAnimation(fig, plot_simulation, interval = 1000/fps , frames=frame_limit)

# Adjust so it fits the page better
plt.subplots_adjust(-0.5, 0, 1, 0.8)


simulation.save("perfectlyelastiTEST7.mp4", fps = fps)



