import random

import numpy as np

from env_dojo.opponent import OPPONENT
from env_dojo.robot import ROBOT
from env_dojo.utils import *
import env_dojo.config as config

'''
# the settings used in ENV:
# 1. thrust_success_reward : reward given when a valid thrust is performed
# 2. thrust_failure_reward : reward given when a thrusting movement is made but miss its target
# 3. collision_reward : given when a collision is inspected
# 4. out_range_reward : given when either is stepped out of the dojo
# 5. reward : a reward given every time step
# 6. time_threshold: maximum time in the env
'''
class Dojo:
    def __init__(self,
                 thrust_success_reward=0,
                 thrust_failure_reward=-2,
                 collision_reward=-2,
                 out_range_reward=-2,
                 reward=-1,
                 time_threshold=500,
                 ):

        # reward setting
        self.thrust_success_reward = thrust_success_reward
        self.thrust_failure_reward = thrust_failure_reward
        self.collision_reward = collision_reward
        self.out_range_reward = out_range_reward
        self.reward = reward

        # setting other parameter
        self.grid_size = config.grid_size
        self.fps = config.fps

        self.robot = ROBOT()
        self.oppo = OPPONENT()

        # set a timer
        self.time_since_start = 0

        # read the valid_thrusting_range
        self.valid_thrusting_range = config.valid_thrusting_time_range
        self.valid_dash_threshold = config.valid_dash_threshold
        self.valid_still_threshold = config.valid_still_threshold

        # make a bool to store the thrust
        self.valid_thrust_duration = config.valid_thrusting_duration_time
        self.valid_thrust_period_list = [False] * (int(self.valid_thrust_duration * self.fps) + 1)

        # add some para for demo
        self.dashing_ready = False
        self.thrusting_ready = False
        self.detour_done = False
        self.guidance_success_rate = None

        # how the ENV done
        self.time_threshold = time_threshold
        self.ending = None

        # initialize first
        self.reset()

        # for the demo
        guidance_num = len(self.oppo.suggestions)
        self.guidance_success_rate = [0] * guidance_num

        # obtain the state space
        self.state_dim = len(self.get_state())
        self.action_dim = len(self.robot.action_bound)

        self.target_index = self.softmax_guidance_rule_select()

    def reset(self, position_required=False, position=None):
        if position is None:
            position = []
        self.robot.reset()
        self.oppo.reset(position_required, position)

        self.avoid_a_bad_start()
        # reset the timer
        self.time_since_start = 0

        # reset some para for demo
        self.thrusting_ready = False

        self.ending = None

        if self.guidance_success_rate:
            self.target_index = self.softmax_guidance_rule_select()

    '''
    load ang save in generating the guidance data
    '''
    def load(self,robo_state_dic,oppo_para):
        self.robot.load_state_dic(robo_state_dic)
        self.oppo.reset(given = True, given_position= oppo_para)

    def save(self):
        robo_state_dic = self.robot.save_state_dic()
        oppo_para = self.oppo.position.tolist()+[self.oppo.aiming_angle]
        return robo_state_dic,oppo_para

    # avoid crash/ out-range/ impossible attacking point
    def avoid_a_bad_start(self):
        bad_ini = False
        collision = self.collision_inspection()
        for i in self.oppo.suggestions:
            out = self.out_range_inspection(i[0])
            if out:
                bad_ini = True
        if collision:
            bad_ini = True

        if bad_ini:
            self.reset()

    # input the action to the simulator
    def step(self,
             robot_command=np.array([0., 0., 0., 0.]),
             opponent_command=np.array([0])  # if the opponent is movable
             ):
        self.robot.robot_step(robot_command)
        # if oppo is movable
        # TODO
        self.oppo.opponent_step(opponent_command)

        # timer
        self.time_since_start += 1

    # TODO
    # observe environment, getting the state
    def get_state(self):
        # normalize the position to [-1,1]
        # observe the state under ROBOT COORDINATION
        # relative position at time step t
        relative_pos = (self.robot.position - self.oppo.position) / self.grid_size
        r_relative_pos = rotate_vector(relative_pos, -self.robot.aiming_angle)

        # relative position at time step t-1
        relative_pos_pre = (self.robot.position_t_1 - self.oppo.position) / self.grid_size
        r_relative_pos_t_1 = rotate_vector(relative_pos_pre, -self.robot.aiming_angle_t_1)

        # relative angle at t and t-1
        relative_angle = np.array([angle_between(self.robot.aiming_angle, self.oppo.aiming_angle) / 180],
                                  dtype=float)
        relative_angle_t_1 = np.array([angle_between(self.robot.aiming_angle_t_1, self.oppo.aiming_angle) / 180],
                                      dtype=float)

        distance_t = np.array([distance_between(self.robot.position,self.oppo.position) / self.grid_size])
        distance_t_1 = np.array([distance_between(self.robot.position_t_1,self.oppo.position) / self.grid_size])

        max_t = self.fps * (self.robot.withdrawal_time + self.robot.holding_time + self.robot.thrusting_time)
        # thrust_count
        thrust_count = np.array([self.robot.shinai_thrusting_time / max_t])

        threshold_dash_distance = 60
        # dash distance
        if self.robot.dashing_distance:
            d_d = self.robot.dashing_distance
            dash_distance = np.array([d_d / threshold_dash_distance])
        else:
            dash_distance = np.array([0.])

        # tip_pos
        tip_pos = (self.robot.shinai_tip[0] - self.oppo.position) / self.grid_size
        tip_pos = rotate_vector(tip_pos, -self.robot.aiming_angle)

        tip_pos_t_1 = (self.robot.shinai_tip_t_1[0] - self.oppo.position) / self.grid_size
        tip_pos_t_1 = rotate_vector(tip_pos_t_1, -self.robot.aiming_angle_t_1)

        V_bound = np.array([120.,120.,85.])
        # velocity0
        speed = self.robot.speed / V_bound
        speed_t_1 = self.robot.speed_t_1 / V_bound

        # pos
        pos = self.robot.position / self.grid_size

        # concatenate
        state = np.concatenate(
            (
                r_relative_pos,  # re_position
                r_relative_pos_t_1,  # re_position t-1
                relative_angle,  # re_angle
                relative_angle_t_1,  # re_angle t-1

                distance_t,
                distance_t_1,

                thrust_count,
                dash_distance,
                tip_pos,
                tip_pos_t_1,

                speed,
                speed_t_1,

                pos,

            ), axis=0
        )
        return state

    # detect the robot and opponent have a collision
    def collision_inspection(self) -> bool:
        r_collision_box = self.robot.robot_collision_box
        o_collision_box = self.oppo.oppo_collision_box
        # detect contour and diagonal
        detect = [[0, 1], [1, 2], [2, 3], [3, 0], [0, 2], [1, 3]]

        collision = False
        for i in detect:
            for j in detect:
                v_1 = np.array([r_collision_box[i[0]], r_collision_box[i[1]]])
                v_2 = np.array([o_collision_box[j[0]], o_collision_box[j[1]]])
                collision = collision or if_straddle(v_1, v_2)

        return collision

    # detect a thrust is valid or invalid
    def robo_thrust_inspection(self) -> tuple[bool, int]:
        valid_thrust = False
        hitting_point = -1
        for i in range(len(self.oppo.defence_list) - 1):
            defence_surface = [np.array(self.oppo.defence_list[i + 1]), np.array(self.oppo.defence_list[i])]
            tip = [np.array(self.robot.shinai_tip[0]), np.array(self.robot.shinai_tip[1])]
            straddle = if_straddle(tip, defence_surface)
            #print(straddle)

            if straddle:
                defence_v = np.array(self.oppo.defence_list[i + 1]) - np.array(self.oppo.defence_list[i])
                defence_v_n = rotate_vector(defence_v, 90)
                tip_v = tip[0] - tip[1]
                opposite = is_vector_opposite(defence_v_n, tip_v, 170)
                if opposite:
                    valid_thrust = True
                    hitting_point = i
                    break

        return valid_thrust, hitting_point

    # if dashing distance above a certain threshold then verify as True
    def robo_valid_dash_inspection(self) -> bool:
        valid_dash = True if self.robot.dashing_distance > self.valid_dash_threshold else False
        return valid_dash

    # if the sum of diff_pos and diff_ang -> 0:
    # verify as True
    def stay_still_inspection(self) -> bool:
        diff_dis = distance_between(self.robot.position, self.robot.position_t_1)
        diff_ang = abs(angle_between(self.robot.aiming_angle, self.robot.aiming_angle_t_1))
        diff_all = diff_ang + diff_dis
        still = True if diff_all < self.valid_still_threshold else False
        return still

    # if the given dots is out of grid_size
    def out_range_inspection(self, given_dots) -> bool:
        out = False
        for i in given_dots:
            if i >= self.grid_size or i <= 0:
                out = True
                break
        return out

    def check_done(self) -> [bool, float, int]:
        hitting_point = -1

        # return done and reward
        collision = self.collision_inspection()

        # give the  threshold of thrusting
        thrust_valid_range_start = self.fps * self.valid_thrusting_range[0]
        thrust_valid_range_end = self.fps * self.valid_thrusting_range[1]
        # case1 collision1
        if self.time_since_start >= self.time_threshold:
            done = False
            reward = self.reward
            self.ending = 'time out'
        else:
            done = False
            reward = self.reward
            self.ending = None


        if collision:
            done = False
            reward += self.collision_reward
            self.ending = 'collision'

        # case2 valid thrust
        if thrust_valid_range_end >= self.robot.shinai_thrusting_time >= thrust_valid_range_start:
            valid_thrust_t, hitting_point = self.robo_thrust_inspection()
            self.valid_thrust_period_list.append(valid_thrust_t)
            self.valid_thrust_period_list.pop(0)
            valid_thrust_t_range = True
            for i in self.valid_thrust_period_list:
                valid_thrust_t_range = valid_thrust_t_range and i

            still = self.stay_still_inspection()
            dash_distance = self.robo_valid_dash_inspection()
            # the thrust is valid ONLY when the robot stays still, dashed a certain distance, and making valid thrust

            # print(self.robot.dashing_distance)
            # print(valid_thrust_t_range,still,dash_distance)

            valid_thrust = valid_thrust_t_range and still and dash_distance

            # valid_thrust
            if valid_thrust:
                done = True
                reward = self.thrust_success_reward
                self.ending = 'valid thrusting'
            else:
                done = False
                reward += self.reward

        # case3 invalid thrust
        if self.robot.shinai_thrusting_time > thrust_valid_range_end:
            # even an invalid thrust is made, the episode CANNOT be stopped
            done = False
            reward += self.thrust_failure_reward
            self.ending = 'X'

        # case4 out of range
        if self.out_range_inspection(self.robot.position):
            done = False
            reward += self.out_range_reward
            self.ending = 'out range'


        return done, reward, hitting_point

    # STILL MIGHT BE A BAD SOLUTION TO A HARD-EXPLORING PROJECT
    # guide the robot by a certain rule
    def act_by_rule(self, loc=5, scale=3):

        target = self.oppo.suggestions[self.target_index]

        starting_point = target[0]
        stopping_point = target[1]
        ang_sugg = target[2]

        thrusting_point = 0.6 * target[0] + 0.4 * target[1]

        dis_start = distance_between(self.robot.shinai_origin, starting_point)
        dis_thrust = distance_between(self.robot.shinai_origin, thrusting_point)
        dis_stop = distance_between(self.robot.shinai_origin, stopping_point)

        ang = abs(angle_between(self.robot.aiming_angle, ang_sugg))

        now_point = np.rad2deg(np.arctan2(self.oppo.position[0] - self.robot.position[0],
                                          self.oppo.position[1] - self.robot.position[1])) + 180

        if abs(angle_between(now_point, self.oppo.aiming_angle)) > 100:
            self.detour_done = True
        else:
            self.detour_done = False

        if self.detour_done:
            left = self.oppo.suggestions[-1][0]
            right = self.oppo.suggestions[0][0]
            dis_left = distance_between(left, self.robot.position)
            dis_right = distance_between(right, self.robot.position)
            side_dot = left if dis_left < dis_right else right

            dis = self.robot.shinai_origin - side_dot
            dis = rotate_vector(dis, -self.robot.aiming_angle)
            if side_dot is left:
                diff_angle = angle_between(self.robot.aiming_angle, self.oppo.suggestions[-1][2])
            else:
                diff_angle = angle_between(self.robot.aiming_angle, self.oppo.suggestions[0][2])

        else:
            if dis_start + ang < np.random.normal(loc, scale) and self.robot.dashing_time < 1:
                self.dashing_ready = True

            elif self.robot.shinai_thrusting_time >= self.valid_thrusting_range[1] * self.fps and self.dashing_ready:
                self.dashing_ready = False


            if not self.dashing_ready :
                dis = self.robot.shinai_origin - starting_point
                dis = rotate_vector(dis, -self.robot.aiming_angle)
                diff_angle = angle_between(self.robot.aiming_angle, ang_sugg)

            else:
                dis = self.robot.shinai_origin - stopping_point
                dis = rotate_vector(dis, -self.robot.aiming_angle)
                diff_angle = angle_between(self.robot.aiming_angle, ang_sugg)


        x = max(min(-dis[0] * (np.random.normal(loc, scale)), 120), -120)
        y = max(min(-dis[1] * (np.random.normal(loc, scale)), 120), -120)
        omega = max(min(-diff_angle * (np.random.normal(loc, scale)), 85), -85)

        if self.dashing_ready and dis_thrust < 5:
            thrust = 1
        elif self.dashing_ready and dis_stop < 1 and not self.robot.shinai_thrusting_time>1:
            self.dashing_ready = False
            thrust = 0
        else:
            thrust = 0

        action = np.array([x, y, omega, thrust])/self.robot.action_bound
        return action

    def softmax_guidance_rule_select(self):
        guidance_list = np.array(self.guidance_success_rate)
        x = (np.max(guidance_list, axis=0) - guidance_list) / max(np.max(guidance_list, axis=0), 1)
        max_x = np.exp(x) / np.sum(np.exp(x), axis=0)

        # locate all the maximum point
        # select_guidance = np.random.choice(a=len(self.oppo.suggestions), p=max_x)
        # select_guidance = np.random.choice(a=len(self.oppo.suggestions))

        guidance = [0,len(self.oppo.suggestions)]
        select_guidance = np.random.choice(a=guidance)

        return select_guidance

    def update_success_rate(self, success_point):
        self.guidance_success_rate[success_point] += 1


#
#
#
# FOR TEST ONLY
if __name__ == '__main__':
    env = Dojo()
