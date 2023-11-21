# STORING ALL THE CONSTANTS RELATED TO THE PLOT

# Ratio between inches and pixels for a 1920x1080 screen
px = 1/96

# I didn't use 1920x1080 because there'd be too much empty space by the sides, want the viewers to focus on the plot 
figsize = (1920*px, 1080*px)

# This constant determines the depth of the depression in the middle of the funnel - the lower the a value, the deeper (min 0)
A = 0.5

# Number of polygons to render in the surface plot
rccount = 50

# resolution of the plot - there will be res**2 points used to plot the surface, recommend at least res=50
res = 100

# this determines the area of the surface. The surface size will be a square with sides of length xy_bound
xy_bound = 5

# Colours of the balls
colours = ["yellow","green","blue", "black"] # original choice was ["yellow","violet","red"] but needed more contrast

# Number of balls
number_of_balls = len(colours)

# This dict stores information about both the wall and ball collision markers - centralised here for ease of editing
point_info = {
    "marker colour" :  ["red","red"],
    "marker" : ["D","x"],
    "size" : [16,32],
    "label" : ["Ball collision points", "Boundary collision points"]
}

    
