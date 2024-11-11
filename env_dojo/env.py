import random
import numpy as np
from env_dojo.opponent import Opponent
from env_dojo.robot import Robot
from env_dojo.utils import *
import env_dojo.config as config


class Dojo:
    def __init__(self,
                 thrust_success_reward=0,
                 thrust_failure_reward=-2,
                 collision_reward=-2,
                 out_range_reward=-2,
                 reward=-1,
                 time_threshold=500,
                 ):
        """
        :param thrust_success_reward: reward given when a valid thrust is performed
        :param thrust_failure_reward: reward given when a thrusting movement is made but miss its target
        :param collision_reward: given when a collision is inspected
        :param out_range_reward: given when either is stepped out of the dojo
        :param reward: a reward given every time step
        :param time_threshold: maximum time in the env
        """
        #--------------------reward setting--------------------#
        self.thrust_success_reward = thrust_success_reward
        self.thrust_failure_reward = thrust_failure_reward
        self.collision_reward = collision_reward
        self.out_range_reward = out_range_reward
        self.reward = reward

        # --------------------env config--------------------#
        self.grid_size = config.grid_size
        self.fps = config.fps

        # --------------------reset players--------------------#
        self.robot = Robot()
        self.oppo = Opponent()

        # --------------------env reset--------------------#
        # set a timer
        self.time_since_start = 0

        # read the valid_thrusting_range
        self.valid_thrusting_range = config.valid_thrusting_time_range
        self.valid_dash_threshold = config.valid_dash_threshold
        self.valid_still_threshold = config.valid_still_threshold

        # make a bool to store the thrust
        self.valid_thrust_duration = config.valid_thrusting_duration_time
        self.valid_thrust_period_list = [False] * (int(self.valid_thrust_duration * self.fps) + 1)

        # add demo para
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
            position = np.array([])
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


    def load(self,robo_state_dic,oppo_para):
        """
        load and save in generating the guidance data
        using vine
        """
        self.robot.load_state_dic(robo_state_dic)
        self.oppo.reset(given = True, given_position= oppo_para)

    def save(self):
        """
        load and save in generating the guidance data
        using vine
        """
        robo_state_dic = self.robot.save_state_dic()
        oppo_para = self.oppo.position.tolist()+[self.oppo.aiming_angle]
        return robo_state_dic,oppo_para


    def avoid_a_bad_start(self):
        """
        avoid crash/ out-range/ impossible attacking point
        """
        if self.collision_inspection():
            self.reset()
        else:
            for suggestion in self.oppo.suggestions:
                if self.out_range_inspection(suggestion[0]):
                    self.reset()
                    break

    def step(self,
             robot_command=np.array([0., 0., 0., 0.]),
             opponent_command=np.array([0])  # if the opponent is movable
             ):
        """
        :param robot_command:  [target_speed_x,target_speed_y,target_rotating_speed,thrust]
        :param opponent_command: TODO
        """
        self.robot.robot_step(robot_command)

        # if oppo is movable
        # TODO
        self.oppo.opponent_step(opponent_command)

        # timer
        self.time_since_start += 1

    def get_state(self):
        """
        get the state in  22 dim
        """
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

        # distance between robot and oppo at time step t and t-1
        distance_t = np.array([distance_between(self.robot.position,self.oppo.position) / self.grid_size])
        distance_t_1 = np.array([distance_between(self.robot.position_t_1,self.oppo.position) / self.grid_size])

        # maximum thrust count
        max_t = self.fps * (self.robot.withdrawal_time + self.robot.holding_time + self.robot.thrusting_time)
        # normalized thrust_count
        thrust_count = np.array([self.robot.shinai_thrusting_time / max_t])

        # maximum dash distance
        MAX_DASH_DIS = 60

        # normalized dash distance
        if self.robot.dashing_distance:
            d_d = self.robot.dashing_distance
            dash_distance = np.array([d_d / MAX_DASH_DIS])
        else:
            dash_distance = np.array([0.])

        # tip pos at time step t
        tip_pos = (self.robot.shinai_tip[0] - self.oppo.position) / self.grid_size
        tip_pos = rotate_vector(tip_pos, -self.robot.aiming_angle)

        # tip pos at time step t-1
        tip_pos_t_1 = (self.robot.shinai_tip_t_1[0] - self.oppo.position) / self.grid_size
        tip_pos_t_1 = rotate_vector(tip_pos_t_1, -self.robot.aiming_angle_t_1)

        V_BOUND = np.array([120.,120.,85.])
        # normalized speed
        speed = self.robot.speed / V_BOUND
        speed_t_1 = self.robot.speed_t_1 / V_BOUND

        # normalized pos
        pos = self.robot.position / self.grid_size

        # concatenate
        # 22dim
        state = np.concatenate(
            (
                r_relative_pos,  # position at opponent coordinate, at time step t and t-1
                r_relative_pos_t_1,
                relative_angle,  # angle at opponent coordinate, at time step t and t-1
                relative_angle_t_1,

                distance_t, # distance between opponent and robot at time step t and t-1
                distance_t_1,

                thrust_count, # thrust progress
                dash_distance, # dash distance

                tip_pos, # tip-pos at opponent coordinate, at time step t and t-1
                tip_pos_t_1,

                speed, # speed at t and t-1
                speed_t_1,

                pos, # position at world coordinate, at time step t
            ), axis=0
        )
        return state

    def collision_inspection(self) -> bool:
        """
        detect the robot and opponent have a collision
        :return: if collision box overpass
        """
        r_collision_box = self.robot.robot_collision_box
        o_collision_box = self.oppo.oppo_collision_box
        detect = [[0, 1], [1, 2], [2, 3], [3, 0], [0, 2], [1, 3]]

        collision = any(
            if_straddle([r_collision_box[i[0]], r_collision_box[i[1]]], [o_collision_box[j[0]], o_collision_box[j[1]]])
            for i in detect for j in detect
        )

        return collision

    def robo_thrust_inspection(self) -> tuple[bool, int]:
        """
        detect whether robot make a valid thrust
        :return: if robot satisfied:
        1) straddle with opponent defence
        2) tip is opposite to defence surface
        """
        valid_thrust = False
        hitting_point = -1

        for i in range(len(self.oppo.defence_list) - 1):
            defence_start, defence_end = map(np.array, (self.oppo.defence_list[i + 1], self.oppo.defence_list[i]))
            tip_start, tip_end = map(np.array, self.robot.shinai_tip)

            if if_straddle([tip_start, tip_end], [defence_start, defence_end]):
                defence_v = defence_start - defence_end
                defence_v_n = rotate_vector(defence_v, 90)

                if is_vector_opposite(defence_v_n, tip_start - tip_end, 170):
                    valid_thrust = True
                    hitting_point = i
                    break

        return valid_thrust, hitting_point

    def robo_valid_dash_inspection(self) -> bool:
        """
        if dashing distance above a certain threshold : verify as True
        """
        return True if self.robot.dashing_distance > self.valid_dash_threshold else False

    def stay_still_inspection(self) -> bool:
        """
        if the sum of diff_pos and diff_ang -> 0 : verify as True
        """
        diff_dis = distance_between(self.robot.position, self.robot.position_t_1)
        diff_ang = abs(angle_between(self.robot.aiming_angle, self.robot.aiming_angle_t_1))
        return True if (diff_ang + diff_dis) < self.valid_still_threshold else False

    def out_range_inspection(self, given_dots) -> bool:
        """
        if position out range : verify as True
        """
        return any(i >= self.grid_size or i <= 0 for i in given_dots)

    def check_done(self) -> [bool, float, int]:
        """
        check the ending of env
        :return: done : bool,reward : float,hitting point : int
        """
        hitting_point = -1
        collision = self.collision_inspection()
        thrust_start, thrust_end = [self.fps * r for r in self.valid_thrusting_range]
        done = False
        reward = self.reward
        self.ending = None

        # Case 1: Timeout
        if self.time_since_start >= self.time_threshold:
            self.ending = 'time out'

        # Case 2: Collision
        elif collision:
            reward += self.collision_reward
            self.ending = 'collision'

        # Case 3: Valid thrust attempt
        elif thrust_start <= self.robot.shinai_thrusting_time <= thrust_end:
            valid_thrust_t, hitting_point = self.robo_thrust_inspection()
            self.valid_thrust_period_list = self.valid_thrust_period_list[1:] + [valid_thrust_t]

            valid_thrust_t_range = all(self.valid_thrust_period_list)
            still = self.stay_still_inspection()
            dash_distance = self.robo_valid_dash_inspection()

            # Check if thrust is valid
            if valid_thrust_t_range and still and dash_distance:
                done = True
                reward = self.thrust_success_reward
                self.ending = 'valid thrusting'
            else:
                reward += self.reward

        # Case 4: Invalid thrust
        elif self.robot.shinai_thrusting_time > thrust_end:
            reward += self.thrust_failure_reward
            self.ending = 'X'

        # Case 5: Out of range
        elif self.out_range_inspection(self.robot.position):
            reward += self.out_range_reward
            self.ending = 'out range'


        return done, reward, hitting_point

    def act_by_rule(self, loc=5, scale=3):
        """
        THE MAIN IDEA TO A HARD-EXPLORING PROJECT
        guide the robot by a rule-based guidance
        :param loc: mu
        :param scale: sigma
        :return: action by guidance
        """
        # load the target
        target = self.oppo.suggestions[self.target_index]
        starting_point,stopping_point,ang_sugg = target

        # define the thrusting point
        thrusting_point = 0.6 * target[0] + 0.4 * target[1]

        dis_start, dis_thrust, dis_stop = (distance_between(self.robot.shinai_origin, point) for point in
                                           (starting_point, thrusting_point, stopping_point))

        ang = abs(angle_between(self.robot.aiming_angle, ang_sugg))
        now_point = np.rad2deg(np.arctan2(self.oppo.position[0] - self.robot.position[0],
                                          self.oppo.position[1] - self.robot.position[1])) + 180

        # detour judgment
        self.detour_done = True if abs(angle_between(now_point, self.oppo.aiming_angle)) > 100 else False

        # if detour
        if self.detour_done:
            left = self.oppo.suggestions[-1][0]
            right = self.oppo.suggestions[0][0]
            side_dot = left if distance_between(left, self.robot.position) < distance_between(right, self.robot.position) else right

            dis = rotate_vector(self.robot.shinai_origin - side_dot, -self.robot.aiming_angle)
            diff_angle = angle_between(self.robot.aiming_angle,
                                       self.oppo.suggestions[0 if side_dot != left else -1][2])

        else:
            if dis_start + ang < np.random.normal(loc, scale) and self.robot.dashing_time < 1:
                self.dashing_ready = True
            elif self.robot.shinai_thrusting_time >= self.valid_thrusting_range[1] * self.fps and self.dashing_ready:
                self.dashing_ready = False

            target_point = starting_point if not self.dashing_ready else stopping_point
            dis = self.robot.shinai_origin - target_point
            dis = rotate_vector(dis, -self.robot.aiming_angle)
            diff_angle = angle_between(self.robot.aiming_angle, ang_sugg)

        x = clamp(-dis[0] * np.random.normal(loc, scale), -self.robot.motor_max_speed, self.robot.motor_max_speed)
        y = clamp(-dis[1] * np.random.normal(loc, scale), -self.robot.motor_max_speed, self.robot.motor_max_speed)
        omega = clamp(-diff_angle * np.random.normal(loc, scale), -self.robot.max_turning_speed, self.robot.max_turning_speed)

        if self.dashing_ready:
            if dis_thrust < 5:
                thrust = 1
            elif dis_stop < 1 and self.robot.shinai_thrusting_time <= 1:
                self.dashing_ready = False
                thrust = 0
            else:
                thrust = 0
        else:
            thrust = 0

        action = np.array([x, y, omega, thrust])/self.robot.action_bound

        return action

    def softmax_guidance_rule_select(self, mode = 'uniform'): # uniform select soft-max
        guidance_list = np.array(self.guidance_success_rate)
        x = (np.max(guidance_list, axis=0) - guidance_list) / max(np.max(guidance_list, axis=0), 1)
        max_x = np.exp(x) / np.sum(np.exp(x), axis=0)

        # locate all the maximum point
        if mode == 'uniform':
            select_guidance = np.random.choice(a=len(self.oppo.suggestions))
        elif mode == 'select':
            guidance = [0, len(self.oppo.suggestions) - 1]
            select_guidance = np.random.choice(a=guidance)
        elif mode == 'soft-max':
            select_guidance = np.random.choice(a=len(self.oppo.suggestions), p=max_x)
        else: # use uniform as DEFAULT
            select_guidance = np.random.choice(a=len(self.oppo.suggestions))
        return select_guidance

    def update_success_rate(self, success_point):
        self.guidance_success_rate[success_point] += 1

# FOR TEST ONLY
if __name__ == '__main__':
    env = Dojo()
