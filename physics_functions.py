import numpy as np
from physics_constants import u,g,e,m,dt,radii
from plot_constants import A

# STORING ALL THE PHYSICS BASED FUNCTIONS HERE
# These are the functions responsible for all physical events in the simulation

# This is our 3D function that will plot the actual funnel surface - it's cone shaped, centred on (0,0)
z = lambda x,y : np.log(A + x**2 + y**2)

# Minimum value of z - used for GPE calculations
z_min = z(0,0)

# These are the partial derivatives of z with respect to x and y; necessary for computing slope and forces acting on the balls
dz_dx = lambda x,y : 2*x / (A + x**2 + y**2)
dz_dy = lambda x,y : 2*y / (A + x**2 + y**2)

# norm([a,b,c,d,...]) = sqrt(a^2 + b^2 + ... )

# Formula for kinetic energy
kinetic_energy = lambda m,v : 0.5 * m * np.linalg.norm(v,axis=0)**2

# Formula for gravitational potential energy
gp_energy = lambda h, h_0 : m * g * ( h - h_0 )

# Calculate euclidean distance
euclidean = lambda a,b : np.linalg.norm( [a.real - b.real, a.imag - b.imag, z(a.real,a.imag) - z(b.real, b.imag)], axis=0 )

# This function maps a pair of numbers into a single number in a bijective way, using the base2 logarithm of prime composition
# Needed for checking for numpy array membership - the standard np.isin() only works as intended on 1D arrays
primecomposition = lambda a,b : np.log(b) / np.log(2) + a

# The function for calculating the changes in displacement and velocity for the balls. It uses the modelling 
# assumptions stated in main_simulation.py
def calculate_motion(position, velocity):
    
    # Get the current x,y coordinates of the balls - don't need the z because it can be calculated from x and y
    xs,ys = position.real, position.imag
    
    # Calculate the gradient - the slope of the funnel surface at each x-y coordinate with respect to the x-axis and y-axis
    zx_gradient = dz_dx(xs,ys)
    zy_gradient = dz_dy(xs,ys)
       
    # The gradient vector will be used to resolve the x-parallel, y-parallel and perpendicular components of the balls' weights
    # Because we assumed that weight acts in the direction of steepest descent, and we've calculated ascent, negativise it
    gradient_vector = -np.array([zx_gradient, zy_gradient, np.ones_like(zx_gradient)])
    
    # Divide by this normalisation factor to get only the pure components of the x-y-perpendicular components - no magnitudes
    norm = np.linalg.norm(gradient_vector,axis=0)
    
    # The weight is given by F=mg, divide by the norm to remove any magnitude given by the partial derivatives 
    # This result will yield the x-parallel, y-parallel and perpendicular components of the weight in [0],[1] and [2]
    # respectively
    gradient_vector *= m*g / norm    
    
    # Next we need to calculate the friction - friction always acts opposite to motion (velocity)
    # Hence we need to get the motion directions so we can get the opposite
    velocity_direction_x = np.sign(velocity.real)
    velocity_direction_y = np.sign(velocity.imag)

    # The angle of the slope of the surface at the given points between the xy-plane and the vectors is arctan(slope)
    theta_zx = np.arctan(zx_gradient)
    theta_zy = np.arctan(zy_gradient)
    
    # The forces acting parallel to motion are the first and second components of the gradient vector
    # real part = x-axis direction, imaginary part = y-axis direciton
    parallel_forces = gradient_vector[0] + 1j*gradient_vector[1]
    
    # Calculate the magnitude of friction - used in the plots
    frictional_magnitude = -u * gradient_vector[2]
    
    # Frictional forces are the same in both x-axis and y-axis, friction = u * F
    # Negativised because it acts in the OPPOSITE direction to motion
    frictional_forces = frictional_magnitude * ( velocity_direction_x + 1j*velocity_direction_y )  

    # The resultant force = parallel forces - frictional forces 
    parallel_forces -= frictional_forces  
    
    # We have the resultant PARALLEL to motion forces, but we need the HORIZONTAL components of these forces since 
    # the modelling assumption stated we'd only be moving horizontally - the vertical could be determined by z(x,y)
    # To get only the horizontal component, multiply by the cosine of the angle
    # sin(arctan(x)) and cos(arctan(x)) do have simplified forms, but they require inversions, square roots and squares
    # I figured this would be less computationally expensive
    parallel_forces.real *= np.cos(theta_zx)
    parallel_forces.imag *= np.cos(theta_zy)
    
    # F = ma ==> a = F/m 
    # Didn't need to include mass in this function at all - I just wanted to make the formulae clearer
    a = parallel_forces / m
    
    # a = dv/dt ==> dv = a*dt - the change in velocity is the acceleration * the time step
    velocity += a*dt
    
    # Ditto as above, v = dx/dt ==> dx = v*dt
    position += velocity*dt
    
    return frictional_magnitude
    
# The function to identify collisions between balls. Returns a vertical array containing the position indices
# of the colliding balls. Only pairs of balls may collide at a time; any collision points only consider the first 2 to collide
def find_ball_collisions(position):
    
    # Get the vertical positions of the balls - the position array only stores xy coordinates
    zs = z(position.real, position.imag)
    
    # These meshgrids will allow us to get all pairs of balls (ball A, ball B), their positions and their radii
    ball_position_A, ball_position_B = np.meshgrid(position , position)
    radiusA, radiusB = np.meshgrid(radii, radii)
    
    # Because the vertical component is not included, we need to create a separate meshgrid instance for this
    z_A, z_B = np.meshgrid(zs, zs)
    
    # Two balls, as spheres, are said to have collided if the distance between their centres, D, <= the sum of their radii.
    # D <= r_A + r_B ==> collision between ball A and ball B
    sum_radii = radiusA + radiusB
    
    # Get the differences in their positions to calculate the Euclidean distance
    xy_diff = ball_position_A - ball_position_B
    z_diff = z_A - z_B 
    
    # Norm = sqrt(sum of squares), which gives us the 3D Euclidean distance.
    distances = np.linalg.norm([xy_diff.real,xy_diff.imag,z_diff],axis=0)

    # Get the indices of the balls that have collided using argwhere
    collisions = np.argwhere( distances <= sum_radii )
    
    # Filter for self-collisions, and all collisions must be ordered from smallest-largest to avoid double collisions
    # e.g ([0 1] and [1 0]) - this is a single collision only, but without this it'd be seen as 2 collisions
    collisions = collisions[collisions[:,0] < collisions[:,1]]
    
    # Return the indices of the colliding balls
    return collisions

# Given two colliding balls, ball A and ball B, and their velocities, find their new velocities after collision
# The formulae for determining the resultant velocities can be derived through the conservation of momentum law and 
# Newton's law of restitution. In the function below, I've derived the formulae on paper and implemented them here
# This is where the coefficient of restitution, e, is crucial. 
def collision(velocity, A,B):

    # Get the initial velocity and mass of ball A
    u1 = velocity[A]
    u_1x, u_1y = u1.real, u1.imag
    m1 = m[A]
    
    # Get the initial velocity and mass of ball B
    u2 = velocity[B]
    u_2x, u_2y = u2.real, u2.imag
    m2 = m[B]
    
    # Applying the formulae I derived on paper to determine the resultant velocities
    v_1x = ( u_1x * (m1-e*m2) + m2*u_2x*(1+e) ) / (m1 + m2)
    v_2x = (m1*u_1x + m2*u_2x - m1*v_1x) / m2
    
    # The principle is the same on the y-axis, simply replace the x-velocity with the y-velocity
    v_1y = ( u_1y * (m1-e*m2) + m2*u_2y*(1+e) ) / (m1 + m2)
    v_2y = (m1*u_1y + m2*u_2y - m1*v_1y) / m2

    # These are the final velocities - remember that these are stored as complex numbers
    v_1 = v_1x + v_1y * 1j
    v_2 = v_2x + v_2y * 1j
    
    # Assign the new velocities to the balls
    velocity[A] = v_1
    velocity[B] = v_2 
    
# Identify collisions between balls and enact them ONCE per collision.
# The bulk of this code is intended to prevent balls from colliding multiple times, a major problem for a small time step dt
# With small dt, just after collision the velocities change, but they still have almost the same x-y coordinates 
# Hence another collision will be registered, and another and so on until they are out of range - the smaller dt, the more 
# false collisions are registered.
# The function is called per frame.
def calculate_ball_collisions(position, velocity, collided_this_frame, balls_collided_already, ball_collision_markers):
    
    # Determine whether we need to update our collision markers on our plot - this is only True if new collisions occur
    update_markers = False    
    
    # Find the ball collisions
    collisions = find_ball_collisions(position)

    # If any collisions have occurred:
    if collisions.size:

        # Get the balls which have been registered as having collided from find_ball_collisions()
        # Remember that only pairs of balls can collide - disregard any past [:, 1]
        colliding_object_As = collisions[:,0]
        colliding_object_Bs = collisions[:,1]
    
        # balls_collided_already determines if two balls in collision proximity have already collided
        # If this array is not empty we need to check it and disregard any collisions already in the array
        # to prevent duplicate collisions
        if balls_collided_already.size:
        
            # Because these collisions are pairs of indices which are hard to check, to check for membership we need to 
            # fold the pairs into unique singular numbers using some bijective function between R^2 and R. 
            # Prime composition, a * 2 **b, is the easiest way I could think of doing this. 
            
            collision_ids = primecomposition(colliding_object_As, colliding_object_Bs)
            already_collided_ids = primecomposition(balls_collided_already[:,0], balls_collided_already[:,1])
    
            # The balls colliding for the first time will be in the collisions list, but NOT in the already array
            not_yet_collided = ~np.isin(collision_ids, already_collided_ids)
            
            # Determine the unique collisions using the above predicate
            unique_collisions = collisions[not_yet_collided]
            
            # If any unique collisions exist:
            if unique_collisions.size:
                
                # Perform the collisions for the unique colliding balls
                collision(velocity,unique_collisions[:,0], unique_collisions[:,1])
                    
            # Determine the balls still in collision with each other - they will be in both the collision and already... arrays
            still_colliding = np.isin(already_collided_ids, collision_ids)
            
            # Filter the array - this removes any balls which have ended their collisions
            balls_collided_already = balls_collided_already[still_colliding]
            
        else:
            
            # If there are no balls that have already collided, then unique collisions = collisions
            unique_collisions = collisions
            
            # Perform the collisions 
            collision(velocity,colliding_object_As, colliding_object_Bs)
    
        # Add the unique collisions in this frame to the already collided array so they don't happen again while still collided
        if balls_collided_already.size:
            balls_collided_already = np.append(balls_collided_already, unique_collisions,axis=0)
        else:
            # Need to use this if-else since numpy doesn't like adding to arrays of different shapes - empty shape is (0,0)
            balls_collided_already = unique_collisions
        
        # If any unique collisions HAVE occurred, we'll need to update the markers for the collisions
        if unique_collisions.size:
            
            if len(collisions) == 1:
                colliding_indices = unique_collisions
            else:
                colliding_indices = unique_collisions[0]
                
            not_colliding = np.indices(position.shape)
            not_colliding = not_colliding[~np.isin( not_colliding, colliding_indices)]
            
            collided_this_frame[~colliding_indices] = False
            collided_this_frame[colliding_indices] = True
            
            # Show that we need to update the markers
            update_markers = True
             
            # Get the points of collision. A good approximation is the midpoint of their centre positions
            collision_points = (position[unique_collisions[:,0]] + position[unique_collisions[:,1]])/2
        
            # Add these coordinates to the ball collision markers
            ball_collision_markers = np.append(ball_collision_markers, collision_points)
            
    # If no collisions have occurred this frame:
    else:
        # If no collisions, then clear the collided_already array since no 
        if balls_collided_already.size:
            balls_collided_already = np.array([])
            
        collided_this_frame[collided_this_frame] = False
    
    # Because Python doesn't have pointers I can't perform array assignments to memory addresses
    # This means I have to return these arrays and assign them in the main_simulation code
    return ( balls_collided_already, ball_collision_markers, update_markers )

# Function that calculates the new velocities of balls that have collided with the wall boundary
def wall_collision(collided, position, velocity):
    
        # Get the position and initial velocity of the ball when it collided
        x,y = position[collided].real,position[collided].imag
        u_x,u_y = velocity[collided].real, velocity[collided].imag
        
        # The slope of the wall boundary at the point of collision can be determined by -x/y, as this is the
        # gradient of the tangent line to a circle
        gradient = -x/y
        
        # Therefore the angle between the ball and the circle boundary is given by arctan(-x/y)
        theta = np.arctan(gradient)
        
        # The goal is to rotate the velocity component by theta clockwise to get the exact parallel and perpendicular
        # components of the ball, then multiply the perpendicular by -e as given by my A level further mechanics knowledge,
        # then rotate them back to their initial velocities to get the new velocities
        angle = 2*np.pi - theta
        
        # Use the rotation matrix to rotate the velocity to get perpendicular and parallel components
        rotated_u_x, rotated_u_y = [u_x * np.cos(angle) - u_y * np.sin(angle),
                                    u_x * np.sin(angle) + u_y * np.cos(angle) ]
        
        # Multiply by -e as required by the formula
        rotated_u_y *= -e
        
        # Rotate them back
        v_x, v_y = [rotated_u_x * np.cos(theta) - rotated_u_y * np.sin(theta),
                    rotated_u_x * np.sin(theta) + rotated_u_y * np.cos(theta) ]
                
        # Return the new velocity after collision - velocity is stored as a complex number
        return v_x + 1j * v_y

# Function to calculate collisions between balls and the boundary - very similar to the above function
def calculate_wall_collisions(position, velocity, collidedwithwallalready, boundary_collision_markers,xy_bound):

    # This will always be false unless a collision has occurred between the boundary and the baall
    update_wall_markers = False
    
    # A ball has collided with the wall boundary if its position magnitude > the xy_bound
    # As that is the equation for the complement of a circle and the boundary is circular
    collided_with_wall = np.absolute(position) >= xy_bound

    # For the array of balls which have already collided, we can make this an array of booleans since we don't have
    # to consider every possible pair of balls - only each ball by itself and its position.

    # Any ball which is in the collidedwithwallalready array and does NOT appear on collided_with_wall is clearly no longer
    # colliding with the wall, so we can set them to False.
    collidedwithwallalready[collidedwithwallalready & ~collided_with_wall] = False
    
    # The balls which have just collided (equivalent to unique_collisions in ball collisions) are the ones we need
    # to perform the collision code for - we shouldn't do this for the balls already in wall collision because 
    # they've already collided - no duplicate collisions before they leave the boundary
    just_collided = collided_with_wall & ~collidedwithwallalready
    if np.any(just_collided):
         
        # Get the x-y coordinates of the colliding balls
        x,y = position[just_collided].real,position[just_collided].imag
        
        # Perform the collision code to calculate the resultant velocity of the colliding balls
        velocity[just_collided] = wall_collision(just_collided, position, velocity)
        
        # Since a new collision has occurred, need to update the collision markers
        boundary_collision_markers = np.append(boundary_collision_markers,x+1j*y )
        
        # Inform the main program that the markers must be updated
        update_wall_markers = True
        
    # Now since the collision has occurred, we need to set them to True so we don't trigger the collision code more than
    # once per collision
    collidedwithwallalready[just_collided] = True
    
    # Again, due to python not having pointers, we have to return these values - no assignment allowed outside main program
    return [collidedwithwallalready, boundary_collision_markers, update_wall_markers]


# Function to calculate kinetic energy of the balls at some time t, given all the required parameters
# Cannot just use the vertical velocity because it spikes when at the bottom of the funnel
def calculate_KE(KE, KE_total, GPE, velocity, prior_velocity, collided, position, position_history, friction, t):
    
    # The KE for this time instance is based on the KE for the previous time instance - a recurrence
    # t = 0 is therefore the base case
    if t > 0:
        
        # Physics formula: Calculate the work done against friction
        # work done = force * distance
        GPE_diff = GPE[t] - GPE[t-1]
        distance_travelled = euclidean(position, position_history[t])
        work_friction = friction * distance_travelled
        
        # This is the formula - gain in KE = loss of GPE, minus the work done against friction
        KE[t][~collided] = ( KE[t-1] - GPE_diff - work_friction )[~collided]
        
        # There was a bug where it's possible that only 1 object "collided" in a frame
        # I don't know if it's still here so I just kept it in
        if np.any(collided) and collided[collided].size > 1:

            # Calculate the z-velocity magnitude just before the collision in the last frame
            zvel_mag = np.sqrt(np.maximum(2 * KE[t-1] / m - prior_velocity.real**2 - prior_velocity.imag**2,0)) 

            # ONLY consider the first 2 colliding objects for simplicity - if multiple collisions,
            # then they'll be handled in the next frame
            first_2 = np.where(collided)[0][:2]
            
            # Refer to the colliding objects as A and B
            indA, indB = first_2[0], first_2[1]
            
            # Initial velocities (u)
            uz_A, uz_B = zvel_mag[first_2]
            
            # Masses
            m1,m2 = m[first_2]
            
            # Calculate the final velocities using conservation of momentum + coeff. of restitution
            vz_A = ( uz_A * (m1-e*m2) + m2*uz_B*(1+e) ) / (m1 + m2)
            vz_B = (m1*uz_A + m2*uz_B - m1*vz_A) / m2
             
            # Calculate the new kinetic energies
            ke_objA = kinetic_energy(m[indA] , [velocity[indA].real, velocity[indA].imag, vz_A])
            ke_objB = kinetic_energy(m[indB] , [velocity[indB].real, velocity[indB].imag, vz_B])
            
            # Get the kinetic energies prior to the collision
            a1,b1 = kinetic_energy(m[indA] , [prior_velocity[indA].real, prior_velocity[indA].imag, zvel_mag[indA]]), kinetic_energy(m[indB] , [prior_velocity[indB].real, prior_velocity[indB].imag, zvel_mag[indB]])

            # There was a bug with kinetic energy being retained at the bottom of the funnel,
            # we can fix this by taking the minimum of the kinetic energy before and after the collision
            # This trades a small amount of accuracy for fixing the issue
            KE[t][indA] = np.minimum(a1,ke_objA) - GPE_diff[indA] - work_friction[indA]
            KE[t][indB] = np.minimum(b1,ke_objB) - GPE_diff[indB] - work_friction[indB]
            
    # The sum of the KE is just the sum of the KEs of the individual balls
    KE_total[t] = KE[t].sum()