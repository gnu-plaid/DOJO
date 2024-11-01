# config

#--------------------DEFAULT--------------------#
# size of dojo
grid_size = 1000
# sampling fps of ENV
fps = 10


#--------------------ROBOT--------------------#

#-----moving part-----#

# length between wheels (vertically)
agv_a = 45.5
# length between wheels (horizontally)
agv_b = 26.
# maxium speed of motor
max_speed = 120.
max_turning_speed = 85.
# 2 methods to calculate the speed in next timestep
# following method used omega(t+1) = A * omega(t) + B * omega(t-1) + C * omega(t-2) + D * m (t) + E * m(t - 1)
# in which A+B+C+D+E = 1
# A B C D E MUST BE COUPLED WITH fps
A = 0.652386
B = -0.07746
C = 0
D = 0.814724
E = -0.38297

#-----thrusting part-----#

# para of shinai
shinai_length = 52
shinai_tip_length = 6

# coordinate of end of shinai with center of ROBOT as origin
shinai_end_x = -25
shinai_end_y = 5

# shinai thrusting length
shinai_thrusting_length = 50

# shinai thrusting time(s)
thrusting_time = 0.3
holding_time = 0.1
withdrawal_time = 0.1

# setting the valid thrusting time
# means during the following range, the thrusting is considered as valid
valid_thrusting_time_range = [0.2,0.4]
# define the
valid_thrusting_duration_time = 0.1

#-----dashing part-----#

# dashing part
# set the maximum robot dashing time(s)
valid_dash_threshold = 12.
# para for still
valid_still_threshold = 2

#-----inspection part-----#

# define the collision box
robo_collision_box_x = 120
robo_collision_box_y = 100

#--------------------OPPONENT--------------------#

# define the dots presenting defence
# find more details in README.OPPONENT
defence_center_front_dis = 10
defence_center_back_dis = 10
defence_back_length = 36
defence_dots = [[2,0],[4,10],[4,10],[4,10],[4,20],[4,10],[6,10]]

# make a suggestion for hitting others
# the ideal hitting distance should be given out
#####
# ALTERING THIS PARAMETER MAY CAUSE SOMETHING INTERESTING
# TODO

ideal_starting_distance = 130
#ideal_thrusting_distance = 100
ideal_stopping_distance = 100

# define the collision box
oppo_collision_box_x = 50
oppo_collision_box_y = 40

#--------------------RENDER--------------------#

# dojo setting
dojo_color = [150,150,150]

agv_body_width = 76
wheel_width = 6
wheel_diameter = 20

# Moving part setting
body_color = 'grey'
body_contour = 'blue'
wheel_color = 'brown'
wheel_contour = 'black'
aiming_tri_color = 'blue'
aiming_tri_contour = 'black'

# Shinai part setting
shinai_width = 4
shinai_color = 'yellow'
shinai_tip_color = 'white'
shinai_tip_contour = 'orange'
shinai_hilt_end_distance = 13
shinai_hilt_diameter = 5
shinai_hilt_color = 'orange'

# opponent setting
# same as moving