import configparser
import sys

import numpy as np
import pygame
import pygame.freetype
import env_dojo.config as config
from env_dojo.env import Dojo
from env_dojo.utils import *


#####
#
#####

class Render(Dojo):
    def __init__(self, scaling_ratio=1.):
        # basic settings and dojo parts
        super(Render, self).__init__()
        self.scaling_ratio = scaling_ratio
        self.grid_size = config.grid_size
        self.fps = config.fps
        self.dojo_color = config.dojo_color
        self.screen = pygame.display.set_mode((self.grid_size * scaling_ratio, self.grid_size * scaling_ratio), flags=0)
        self.clock = pygame.time.Clock()
        self.done = False
        # moving parts
        self.agv_body_width = config.agv_body_width// 2
        self._box_mat = np.array([[1., 1.],
                                  [-1., 1.],
                                  [-1., -1.],
                                  [1, -1.]])
        self.wheel_width = config.wheel_width // 2
        self.wheel_diameter = config.wheel_diameter // 2
        self.body_color = config.body_color
        self.body_contour_color = config.body_contour
        self.wheel_color = config.wheel_color
        self.wheel_contour_color = config.wheel_contour
        self.aiming_tri_color = config.aiming_tri_color
        self.aiming_tri_contour = config.aiming_tri_contour
        # shinai parts
        self.shinai_width = config.shinai_width // 2
        self.rectangle_mat = np.array([
            [1, 0, 1],
            [1, 0, -1],
            [0, 1, -1],
            [0, 1, 1]
        ])
        self.shinai_color = config.shinai_color
        self.shinai_tip_color = config.shinai_tip_color
        self.shinai_tip_contour = config.shinai_tip_contour
        self.shinai_hilt_end_distance = config.shinai_hilt_end_distance
        self.shinai_hilt_diameter = config.shinai_hilt_diameter
        self.shinai_hilt_color = config.shinai_hilt_color

        # dashing_trace
        self.dashing_trace = []

    def display_robot(self):
        # body part
        agv_a = self.robot.agv_a * self.scaling_ratio
        agv_b = self.robot.agv_b * self.scaling_ratio
        width = self.agv_body_width * self.scaling_ratio
        position = self.robot.position * self.scaling_ratio
        # wheel part
        wheel_width = self.wheel_width * self.scaling_ratio
        wheel_diameter = self.wheel_diameter * self.scaling_ratio
        # draw each wheel
        wheel_center_points = self._box_mat.dot(np.array([[agv_a, 0], [0, agv_b]]))
        wheel_contour_points = self._box_mat.dot(np.array([[wheel_width, 0],
                                                           [0, wheel_diameter]]))
        body_contour_points = self._box_mat.dot(np.array([[width, 0], [0, width]]))
        # draw moving part
        body = []
        for _ in range(4):
            center_point = wheel_center_points[_]
            wheel_contour_points_pos = (wheel_contour_points + np.tile(center_point, (4, 1)))
            wheels = []
            for __ in range(4):
                r_V = rotate_vector(wheel_contour_points_pos[__], self.robot.aiming_angle)
                r_V += position
                wheels.append(r_V)
            # draw wheels
            pygame.draw.polygon(self.screen, self.wheel_color, wheels)
            pygame.draw.lines(self.screen, self.wheel_contour_color, True, wheels, width=int(2 * self.scaling_ratio))
            # draw body
            R_V = rotate_vector(body_contour_points[_], self.robot.aiming_angle)
            R_V += position
            body.append(R_V)
        pygame.draw.polygon(self.screen, self.body_color, body)
        pygame.draw.lines(self.screen, self.body_contour_color, True, body, width=int(2 * self.scaling_ratio))

        # draw direction
        direction_vector = self.robot.robot_y_axis
        vec_2 = rotate_vector(direction_vector, 145)
        vec_3 = rotate_vector(direction_vector, -145)
        # set direction length: 14
        direction_length = 10 * self.scaling_ratio
        head = []
        for j in [direction_vector, vec_2, vec_3]:
            head.append(self.robot.position * self.scaling_ratio + j * direction_length)
        pygame.draw.polygon(self.screen, self.aiming_tri_color, head)
        pygame.draw.lines(self.screen, self.aiming_tri_contour, True, head, width=int(2 * self.scaling_ratio))

        # draw thrusting part
        shinai_tip_pos = self.robot.shinai_tip[0]
        shinai_tip_pos_II = self.robot.shinai_tip[1]
        shinai_end_pos = shinai_tip_pos - self.robot.robot_y_axis * self.robot.shinai_length

        shinai_list = np.array(
            [shinai_end_pos, shinai_tip_pos, self.robot.robot_x_axis * self.shinai_width * self.scaling_ratio])
        shinai_contour_list = self.rectangle_mat.dot(shinai_list)
        pygame.draw.polygon(self.screen, self.shinai_color, shinai_contour_list * self.scaling_ratio, 0)
        pygame.draw.lines(self.screen, self.shinai_tip_contour, True, shinai_contour_list * self.scaling_ratio,
                          width=1)
        shinai_tip_list = np.array([shinai_tip_pos, shinai_tip_pos_II,
                                    self.robot.robot_x_axis * 1.5 * self.shinai_width * self.scaling_ratio])
        shinai_tip_contour_list = self.rectangle_mat.dot(shinai_tip_list)
        pygame.draw.polygon(self.screen, self.shinai_tip_color, shinai_tip_contour_list * self.scaling_ratio, 0)

        hilt_pos = shinai_end_pos + self.shinai_hilt_end_distance * self.robot.robot_y_axis
        h_s = (hilt_pos - self.robot.robot_x_axis * self.shinai_hilt_diameter) * self.scaling_ratio
        h_e = (hilt_pos + self.robot.robot_x_axis * self.shinai_hilt_diameter) * self.scaling_ratio

        pygame.draw.lines(self.screen, self.shinai_hilt_color, False, [h_e, h_s], width=int(5 * self.scaling_ratio))

    def display_opponent(self):
        display_defence_list = []
        display_body_list = []
        for i in self.oppo.defence_list:
            display_defence_list.append((np.array(i) * self.scaling_ratio).tolist())
        for j in self.oppo.body_list:
            display_body_list.append((np.array(j) * self.scaling_ratio).tolist())

        pygame.draw.polygon(self.screen, 'blue', display_body_list, width=0)
        pygame.draw.lines(self.screen, 'black', False, display_defence_list, width=int(3 * self.scaling_ratio))

    def display_hitting_suggestions(self):
        # display the suggestions dots
        for i in self.oppo.suggestions:
            pygame.draw.circle(self.screen, 'red', i[0] * self.scaling_ratio, 2)
            pygame.draw.circle(self.screen, 'green', i[1] * self.scaling_ratio, 2)

    def display_collision_box(self):
        robot_collision = []
        for i in self.robot.robot_collision_box:
            robot_collision.append(i * self.scaling_ratio)
        pygame.draw.lines(self.screen, self.body_contour_color, True, robot_collision,
                          width=int(2 * self.scaling_ratio))

        oppo_collision = []
        for i in self.oppo.oppo_collision_box:
            oppo_collision.append(i * self.scaling_ratio)
        pygame.draw.lines(self.screen, self.body_contour_color, True, oppo_collision,
                          width=int(2 * self.scaling_ratio))

    def display_dashing_trace(self):
        if self.robot.dashing_time > 0:
            self.dashing_trace.append(self.robot.position * self.scaling_ratio)
        else:
            self.dashing_trace = []

        if len(self.dashing_trace) > 1:
            pygame.draw.lines(self.screen, 'red', False, self.dashing_trace,
                              width=int(2 * self.scaling_ratio))



    def play(self, command):

        self.clock.tick(self.fps)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        self.screen.fill(self.dojo_color)

        self.display_opponent()
        self.display_robot()
        self.display_dashing_trace()

        # to display the collision box:
        # self.display_collision_box()

        # to display the suggestions:
        # self.display_hitting_suggestions()

        self.step(robot_command=command)
        pygame.display.flip()

    def pause(self, game_done):
        if self.ending:
            a = pygame.freetype.Font('C:/Windows/Fonts/arial.ttf', 50)
            center_pos = np.array([0, 0])
            a.render_to(self.screen, center_pos, str(self.ending), fgcolor='GOLD')
            pygame.display.update()

        if game_done:
            pygame.time.delay(1000)

if __name__ == '__main__':
    # FOR TEST
    game = Render(scaling_ratio=0.7)
    pygame.init()

    for _ in range(10):
        game.done = False
        game.reset(position_required=False, position=[475, 615, 180])
        while not game.done and game.time_since_start < 500:
            if game.time_since_start == 20:
                state_dic,oppo_state = game.save()

            command = game.act_by_rule(loc = 15, scale = 3)
            game.play(command=command)
            game.done, reward, _ = game.check_done()
            game.pause(game.done)

        if state_dic and oppo_state:
            game.done = False
            game.reset()
            game.load(state_dic, oppo_state)
            while not game.done and game.time_since_start < 500:
                command = game.act_by_rule(loc=15, scale=3)
                game.play(command=command)
                game.done, reward, _ = game.check_done()
                game.pause(game.done)
