import numpy as np
from physics_constants import u,g,e,m,dt,radii
from plot_constants import a, point_info


# This is our 3D function that will plot the actual funnel surface - it's cone shaped.
z = lambda x,y : np.log(a + x**2 + y**2)

# These are the partial derivatives of z with respect to x and y; necessary for computing slope and forces acting on the balls
dz_dx = lambda x,y : 2*x / (a + x**2 + y**2)
dz_dy = lambda x,y : 2*y / (a + x**2 + y**2)

# This function maps a pair of numbers into a single number in a bijective way, using the base2 logarithm of prime composition
# Needed for checking for numpy array membership - the standard np.isin() only works as intended on 1D arrays
primecomposition = lambda a,b : np.log(b) / np.log(2) + a

def calculate_motion(position, velocity):
    
    xs,ys = position.real, position.imag
    
    zx_gradient = dz_dx(xs,ys)
    zy_gradient = dz_dy(xs,ys)
       
    gradient_vector = np.array([zx_gradient, zy_gradient, np.ones_like(zx_gradient)])
    norm = np.linalg.norm(gradient_vector,axis=0)
    
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
    
def find_ball_collisions(position):
    
    zs = z(position.real, position.imag)
    
    object_position_A, object_position_B = np.meshgrid(position , position)
    radiusA, radiusB = np.meshgrid(radii, radii)
    z_A, z_B = np.meshgrid(zs, zs)
    
    sum_radii = radiusA + radiusB
    xy_diff = object_position_A - object_position_B
    z_diff = z_A - z_B 
    
    distances = np.linalg.norm([xy_diff.real,xy_diff.imag,z_diff],axis=0)

    collisions = np.argwhere( distances <= sum_radii )
    collisions = collisions[collisions[:,0] < collisions[:,1]]
    
    return collisions

def collision(velocity, A,B):

    u1 = velocity[A]
    u_1x, u_1y = u1.real, u1.imag
    m1 = m[A]
    
    u2 = velocity[B]
    u_2x, u_2y = u2.real, u2.imag
    m2 = m[B]
    
    v_1x = ( u_1x * (m1-e*m2) + m2*u_2x*(1+e) ) / (m1 + m2)
    v_2x = (m1*u_1x + m2*u_2x - m1*v_1x) / m2
    
    v_1y = ( u_1y * (m1-e*m2) + m2*u_2y*(1+e) ) / (m1 + m2)
    v_2y = (m1*u_1y + m2*u_2y - m1*v_1y) / m2

    v_1 = v_1x + v_1y * 1j
    v_2 = v_2x + v_2y * 1j
    
    velocity[A] = v_1
    velocity[B] = v_2 
    
def calculate_collisions(position, velocity, balls_collided_already, ball_collision_markers):
    
    update_markers = False    
    
    collisions = find_ball_collisions(position)

    if collisions.size:
        
        colliding_object_As = collisions[:,0]
        colliding_object_Bs = collisions[:,1]
    
        if balls_collided_already.size:
        
            collision_ids = primecomposition(colliding_object_As, colliding_object_Bs)
            already_collided_ids = primecomposition(balls_collided_already[:,0], balls_collided_already[:,1])
    
            not_yet_collided = ~np.isin(collision_ids, already_collided_ids)
            
            unique_collisions = collisions[not_yet_collided]
            
            if unique_collisions.size:
            
                collision(velocity,unique_collisions[:,0], unique_collisions[:,1])
                    
            still_colliding = np.isin(already_collided_ids, collision_ids)
            
            balls_collided_already = balls_collided_already[still_colliding]
            
        else:
            
            unique_collisions = collisions
            collision(velocity,colliding_object_As, colliding_object_Bs)
    
        
        if balls_collided_already.size:
            balls_collided_already = np.append(balls_collided_already, unique_collisions,axis=0)
        else:
            balls_collided_already = unique_collisions
        
        if unique_collisions.size:
            
            update_markers = True
             
            collision_points = (position[unique_collisions[:,0]] + position[unique_collisions[:,1]])/2
        
            ball_collision_markers = np.append(ball_collision_markers, collision_points)
            
            
    else:
        if balls_collided_already.size:
            balls_collided_already = np.array([])
    
    return [balls_collided_already, ball_collision_markers, update_markers]


def calculate_wallphysics(position, velocity, collidedwithwallalready, boundary_collision_markers,xy_bound):

    update_wall_markers = False
    
    u_x,u_y = velocity.real, velocity.imag
    
    collided_with_wall = np.absolute(position) >= xy_bound

    collidedwithwallalready[collidedwithwallalready & ~collided_with_wall] = False
    just_collided = collided_with_wall & ~collidedwithwallalready
    if np.any(just_collided):
         
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
        
        boundary_collision_markers = np.append(boundary_collision_markers,x+1j*y )
        
        update_wall_markers = True
        
    collidedwithwallalready[just_collided] = True
    
    return [collidedwithwallalready, boundary_collision_markers, update_wall_markers]