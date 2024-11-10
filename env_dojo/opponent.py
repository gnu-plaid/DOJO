import random

import numpy as np
from sympy.stats.sampling.sample_numpy import numpy

from env_dojo.utils import *
import env_dojo.config as config



class Opponent(object):
    def __init__(self):
        # TODO
        # some details should be added
        """
        the opponent is represented as a fixed *DOBOUGU*(defence in kendo)
        for more details of setting the opponent *DEFENCE*
        """
        #---------------------import para---------------------#
        # dojo settings
        self.grid_size = config.grid_size
        self.fps = config.fps

        # defence settings
        self.defence_center_front_dis = config.defence_center_front_dis
        self.defence_center_back_dis = config.defence_center_back_dis
        self.defence_back_length = config.defence_back_length
        self.defence_dots = config.defence_dots

        # for guidance
        self.ideal_starting_distance = config.ideal_starting_distance
        self.ideal_stopping_distance = config.ideal_stopping_distance

        # for collision inspection
        self.oppo_collision_box_x = config.oppo_collision_box_x // 2
        self.oppo_collision_box_y = config.oppo_collision_box_y // 2

        # ---------------------setting ini---------------------#
        # pos/ang setting
        self.position = np.array([self.grid_size // 2, self.grid_size // 2]).astype(float)
        self.aiming_angle = 0
        self.fixed_y_axis = np.array([0., 1.])
        self.fixed_x_axis = np.array([1., 0.])
        self.body_list = []
        self.defence_list = []
        self._box_mat = np.array([[1., 1.],
                                  [-1., 1.],
                                  [-1., -1.],
                                  [1, -1.]])

        # dots used for guidance
        self.suggestions = None
        # collision box used for collision inspection
        self.oppo_collision_box = None
        # ---------------------defence ini---------------------#
        self._set_defence_initial()

        # ---------------------opponent ini---------------------#
        self.reset()

    def _set_defence_initial(self):
        """
        initialize the defence
        """
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

        # save the defence_list and body_list located in origin with initial angle
        self.defence_list_origin = defence_list
        self.body_list_origin = body_list

    def reset(self, given:bool=False, given_position:np.array([int,int])=np.array([0,0,0]), grid_gap:int=0):
        """
        generate a random position for opponent
        the given_position should be in tuple:[position_x, position_y, angle]
        :param given: whether set a target position
        :param given_position: set your target
        :param grid_gap: if the target is designed to have a margin away from the dojo range, add margin here
        """
        if given:
            self.position = np.array(given_position[:2])
            self.aiming_angle = given_position[2]
        else:
            self.position = np.array([random.randint(0 + grid_gap, self.grid_size - grid_gap),
                                      random.randint(0 + grid_gap, self.grid_size - grid_gap)])
            self.aiming_angle = random.randint(-180, 180)

        self.set_opponent_defence()
        self.suggestions = self.guidance()
        self.oppo_collision_box = self.collision_box()

    def set_opponent_defence(self):
        """
        generating opponent
        """
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

    def guidance(self) -> list:
        """
        calculate the ideal hitting point for a certain defence
        return should be in a tuple as [start_point, stop_point, ideal_angle]
        """
        guidance = []
        for i in range(len(self.defence_list) - 1):
            defence_vector = np.array(self.defence_list[i + 1]) - np.array(self.defence_list[i])

            # get generalized normal vector
            normal_v_g = rotate_vector(defence_vector, 90) / np.sqrt(np.sum(np.square(rotate_vector(defence_vector, 90))))

            # add the normal vector to the middle points of every surface of defence
            middle_point = (np.array(self.defence_list[i + 1]) + np.array(self.defence_list[i])) / 2

            # get guidance
            suggestion = [middle_point + self.ideal_starting_distance * normal_v_g,
                          middle_point + self.ideal_stopping_distance * normal_v_g,
                          np.rad2deg(np.arctan2(normal_v_g[0], normal_v_g[1])) + 180]

            guidance.append(suggestion)

        return guidance

    def collision_box(self) -> list:
        """
        :return: hitting box list
        """
        box = []
        collision_box_o = self._box_mat.dot(
            np.array([[self.oppo_collision_box_x, 0], [0, self.oppo_collision_box_y]]))
        for i in range(4):
            r_v = rotate_vector(collision_box_o[i], self.aiming_angle)
            r_v += self.position
            box.append(r_v)
        return box


    # TODO
    def opponent_step(self, command: [float] = None):
        """
        for multi-agents task
        a further topic in the future(nope, have fun kids)
        """
        pass
