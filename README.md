# 3d_gravity_funnel

This is a 3D simulation I've created using numpy and matplotlib, inspired by the black hole funnel simulation in the National Space Centre at Leicester.
The original concept was to place some balls on the funnel with some initial velocity as they went round, getting closer towards the centre
as they lost horizontal velocity from friction and gained velocity towards it from gravity, with this combination of forces mimicking how a black hole
attracts objects towards it within its event horizon. 

I therefore wanted to make my own model to mimic this idea, with the **overall goal of determining the optimal starting velocity and height above the centre 
with which to keep the ball spinning around the longest**. However, I made a couple of modifications to mine:

1. The funnel has a different, steeper slope so it has a greater impact on the ball velocity, so it's easier to observe without having to wait a longer time.
The equation of the surface is: **Z = ln(A + X^2 + Y^2)** for some constant a, which is set to 0.5.
2. The funnel has a bottom, unlike the Space Centre simulation where balls drop into a pit. I did this so that we could also see more complex energy transfers
in kinetic and gravitational potential energy. The depth / steepness of the funnel is determined by the constant **A** - as a -> 0, it becomes a singularity, becoming
shallower as **A** increases.

There are currently **6 main program files**:

**physics_functions.py**: This stores all of the functions that calculate the changes in velocity, handling acceleration, changes in motion, displacement, collisions, etc.

**physics_constants.py** : stores all of the important constants that modify the overall behaviour of the system: coefficients of friction, restitution, size of the timestep dt, etc.

**plot_functions.py** : stores all of the functions that handle the plot itself - how it's presented - e.g drawing the surface and cylinder, the plot axes, etc.

**plot_constants.py** : stores of the constants used in drawing the plot, e.g colours of the balls, resolution of the plot, figure size...

**simulation.py** This is the centralised program that takes all of the above and puts them together to actually run the simulation. 

**simulation_plots.py** Like above, but also tracks the kinetic, gravitational potential and total system energies and shows them as the simulation progresses.

In addition, I've also kept all the clips I made during testing in a separate file so you can see how the work progressed. There are fewer total commits than I would've liked, but should still 
be an indicator of progress.

All modelling assumptions are stated in *simulation.py*. Because all functions are defined based on the surface function z = f(x,y), you can change the surface function (so long as you also
change the partial derivatives *dz_dx* and *dz_dy* and the simulation should still run well on the new function, so long as it's reasonably nice (_smooth, continuous, no sudden spikes, etc_).

This simulation creates the data - it can then be used to test to solve the problem I specified above of finding the optimal velocity, but I haven't yet made the files to do this. However, 
we can still design some kind of pseudo-experiment for it, using some informal definitions (_since it's my own project, after all_).

# The experiment
Given a cylindrically symmetrical (_same behaviour at any points equidistant from the centre_) surface that depresses towards the centre, a circular boundary and a ball,
what is the best initial height and horizontal velocity to provide the ball so it avoids falling into the centre as long as possible? (_assuming friction, collision and weight exist_) 

**Independent variables:**
The initial height and velocity of the ball. These will likely be considered both together and separately in the experiment to test their relationship.

**Dependent variables:**
The time taken for the ball to fall into the centre, and the time taken for the ball to totally lose its kinetic energy. These are similar/analogous measurements but will be separately considered.

**Control variables:**
The size of the funnel, the mass/size of the ball, the coefficients of friction and restitution, the depth of the funnel bottom and the time step.

My hypothesis is that the best velocity to give should be tangential to the boundary, so that it "orbits" the edge of the funnel but not so fast that it collides with the boundary, as this would
cause it to lose kinetic energy which would make it hit the centre sooner. This idea is motivated by considering how planets orbit stars, always accelerating towards them but never colliding (as no 
friction or other significant energy loss factors).
