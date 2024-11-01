import random

from sympy.stats.sampling.sample_numpy import numpy

from env_dojo.utils import *
import env_dojo.config as config
'''
the opponent is represent as a fixed *DOBOUGU*(defence in kendo)
for more details of setting the opponent *DEFENCE*, look up in README.OPPONENT section
'''

class OPPONENT:
    def __init__(self):
        # dojo settings
        self.grid_size = config.grid_size
        self.fps = config.fps

        # defence settings
        self.defence_center_front_dis = config.defence_center_front_dis
        self.defence_center_back_dis = config.defence_center_back_dis
        self.defence_back_length = config.defence_back_length
        self.defence_dots = config.defence_dots

        # pos/ang setting
        self.position = np.array([self.grid_size // 2, self.grid_size // 2]).astype(float)
        self.aiming_angle = 0
        self.fixed_y_axis = np.array([0., 1.])
        self.fixed_x_axis = np.array([1., 0.])
        self.defence_list = []
        self.detour_point = []

        # set defence ini
        self._set_defence_initial()

        # initialize the defence and body list

        self.defence_list = self.defence_list_origin
        self.body_list = self.body_list_origin

        '''
        for demonstration, calculate hitting suggestions
        every suggestion is stored as tuple:
        [ideal_dashing_point, ideal_thrusting_point, ideal_thrusting_angle]
        '''
        self.ideal_starting_distance = config.ideal_starting_distance
        self.ideal_stopping_distance = config.ideal_stopping_distance

        self.suggestions = self.guidance()
        '''
        get collision box
        '''
        self.oppo_collision_box_x = config.oppo_collision_box_x // 2
        self.oppo_collision_box_y = config.oppo_collision_box_y // 2
        self._box_mat = np.array([[1., 1.],
                                  [-1., 1.],
                                  [-1., -1.],
                                  [1, -1.]])
        self.oppo_collision_box = self.collision_box()

    '''
    initialize the opponent
    '''
    def _set_defence_initial(self):
        defence_dots = self.defence_dots
        defence_list = []
        body_list = []
        rotating_v = self.fixed_x_axis
        pre_dot = self.defence_center_front_dis * self.fixed_y_axis
        for dot in defence_dots:
            rotating_v = rotate_vector(rotating_v, dot[1])
            pre_dot += dot[0] * rotating_v
            defence_list.append(pre_dot.tolist())
            body_list.append(pre_dot.tolist())
        body_list.append([self.defence_back_length // 2, -self.defence_center_back_dis])
        defence_list.reverse()
        body_list.reverse()

        # symmetry
        rotating_v = -1 * self.fixed_x_axis
        pre_dot = self.defence_center_front_dis * self.fixed_y_axis
        for dot_ in defence_dots:
            rotating_v = rotate_vector(rotating_v, -dot_[1])
            pre_dot += dot_[0] * rotating_v
            defence_list.append(pre_dot.tolist())
            body_list.append(pre_dot.tolist())
        body_list.append([-self.defence_back_length // 2, -self.defence_center_back_dis])

        # save the defence_list and body_list located in origin with a initial angle
        self.defence_list_origin = defence_list
        self.body_list_origin = body_list

    '''
    generate a random position for opponent
    the given_position should be in tuple:[position_x, position_y, angle]
    '''
    def reset(self, given=False, given_position=None, grid_gap=0):
        if given:
            pos = np.array(given_position[:2])
            angle = given_position[2]
            self.position = pos
            self.aiming_angle = angle
        else:
            x = random.randint(0 + grid_gap, self.grid_size - grid_gap)
            y = random.randint(0 + grid_gap, self.grid_size - grid_gap)
            self.position = np.array([x, y])
            self.aiming_angle = random.randint(-180, 180)

        # calculate the detailed defence by given position and given angle
        self.set_opponent_defence()

        # get the suggestions
        self.suggestions = self.guidance()

        # get collision_box
        self.oppo_collision_box = self.collision_box()

    # calculating the defence dots by angle and position
    def set_opponent_defence(self):
        defence = self.defence_list_origin.copy()
        body = self.body_list_origin.copy()
        pos = self.position.copy()
        defence_list = []
        body_list = []
        for i in range(len(self.defence_list_origin)):
            defence_list.append((rotate_vector(defence[i], self.aiming_angle) + pos).tolist())

        for j in range(len(self.body_list_origin)):
            body_list.append((rotate_vector(body[j], self.aiming_angle) + pos).tolist())

        self.body_list = body_list.copy()
        self.defence_list = defence_list.copy()

    '''   
    calculate the ideal hitting point for a certain defence
    return should be in a tuple as [start_point, stop_point, ideal_angle]
    '''
    def guidance(self) -> list:
        guidance = []
        for i in range(len(self.defence_list) - 1):
            defence_vector = np.array(self.defence_list[i + 1]) - np.array(self.defence_list[i])
            # get normal vector
            normal_v = rotate_vector(defence_vector, 90)

            # get generalized normal vector
            normal_v_g = normal_v / np.sqrt(np.sum(np.square(normal_v)))

            # add the normal vector to the middle points of every surface of defence
            middle_point = (np.array(self.defence_list[i + 1]) + np.array(self.defence_list[i])) / 2

            # find the start and stop points
            d_start = self.ideal_starting_distance
            d_stop = self.ideal_stopping_distance

            # get ideal point
            ideal_starting_point = middle_point + d_start * normal_v_g
            ideal_stopping_point = middle_point + d_stop * normal_v_g
            ideal_angle = np.rad2deg(np.arctan2(normal_v_g[0], normal_v_g[1])) + 180

            # get guidance
            suggestion = [ideal_starting_point, ideal_stopping_point, ideal_angle]
            guidance.append(suggestion)

        return guidance

    '''
    :return the hitting box
    '''
    def collision_box(self) -> list:
        box = []
        collision_box_o = self._box_mat.dot(
            np.array([[self.oppo_collision_box_x, 0], [0, self.oppo_collision_box_y]]))
        for i in range(4):
            r_v = rotate_vector(collision_box_o[i], self.aiming_angle)
            r_v += self.position
            box.append(r_v)
        return box

    # TODO
    '''
    for multi-agents task
    as a further topic in the future
    '''
    def opponent_step(self, command: [float] = None):
        pass
