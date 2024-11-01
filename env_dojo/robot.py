import env_dojo.config as config
from env_dojo.utils import *


'''
Define the ROBOT
ROBOT will be initialized in the middle of DOJO, facing the angle of 0
To control the robot, using function *robot_step*
the command should be a tuple of 4 [int,int,int,int]
The dynamics model of agv credits to NAKAGAWA Y.and NIE S.
find more details in paper : #TODO
'''

class ROBOT(object):
    def __init__(self):
        #-----DEFAULT----#

        self.grid_size = config.grid_size
        self.fps = config.fps

        #--------------------moving config--------------------#
        self.agv_a = config.agv_a
        self.agv_b = config.agv_b
        self.motor_max_speed = config.max_speed
        self.max_turning_speed = config.max_turning_speed
        # initialized para
        # set initial position to the center of dojo, and aiming angle at 0
        self.initial_position = np.array([self.grid_size // 2, self.grid_size // 2]).astype(float)
        self.initial_angle = 0
        # fixed axis
        self.fixed_y_axis = np.array([0., 1.])
        self.fixed_x_axis = np.array([1., 0.])
        self.A = config.A
        self.B = config.B
        self.C = config.C
        self.D = config.D
        self.E = config.E
        self.para = np.array([self.A, self.B, self.C, self.D, self.E])

        # --------------------thrusting config--------------------#
        self.shinai_end_x = config.shinai_end_x
        self.shinai_end_y = config.shinai_end_y
        self.shinai_thrusting_length = config.shinai_thrusting_length
        self.thrusting_time = config.thrusting_time
        self.holding_time = config.holding_time
        self.withdrawal_time = config.withdrawal_time
        self.shinai_length = config.shinai_length
        self.shinai_tip_length = config.shinai_tip_length

        #--------------------moving part--------------------#
        self.__move_ini__()

        #--------------------thrusting part--------------------#
        self.__thrust_ini__()

        # --------------------observation part--------------------#
        self.__observation_ini__()

        # --------------------state_dic_ini--------------------#
        self.state_ini = self.__state_dic_ini__()

        #--------------------inspection part--------------------#
        self._box_mat = np.array([[1., 1.],
                                  [-1., 1.],
                                  [-1., -1.],
                                  [1, -1.]])
        # get the collision box after initialize
        self.robot_collision_box_x = config.robo_collision_box_x // 2
        self.robot_collision_box_y = config.robo_collision_box_y // 2

        # get the collision_box
        self.robot_collision_box = self.collision_box()

        #--------------------define the action bound---------------------#
        self.action_bound = np.array([120,120,85,1])


    def __move_ini__(self):
        self.initial_speed = np.array([0., 0., 0.]).astype(float)  # Vx, Vy, VC
        self.initial_omega_t_0 = np.array([0., 0., 0., 0.]).astype(float)  # omega(t)
        self.initial_omega_t_1 = np.array([0., 0., 0., 0.]).astype(float)  # omega(t-1)
        self.initial_omega_t_2 = np.array([0., 0., 0., 0.]).astype(float)  # omega(t-2)
        self.initial_omega_target_t_1 = np.array([0., 0., 0., 0.]).astype(float)  # m(t-1)

        self.position = self.initial_position
        self.aiming_angle = self.initial_angle
        self.speed = self.initial_speed
        self.speed_t_1 = self.initial_speed
        self.robot_x_axis = self.fixed_x_axis
        self.robot_y_axis = self.fixed_y_axis
        self.omega_t_0 = self.initial_omega_t_0
        self.omega_t_1 = self.initial_omega_t_1
        self.omega_t_2 = self.initial_omega_t_2
        self.omega_target_t_1 = self.initial_omega_target_t_1

    def __thrust_ini__(self):
        # for state
        self.shinai_origin = self.position + self.robot_x_axis * self.shinai_end_x + self.robot_y_axis * self.shinai_end_y
        self.shinai_tip = np.array([self.shinai_origin + self.robot_y_axis * self.shinai_length,
                                    self.shinai_origin + self.robot_y_axis * (
                                            self.shinai_length - self.shinai_tip_length)])
        self.shinai_thrusting_time = 0
        self.shinai_tip_t_1 = self.shinai_tip

        self.shoulder = self.position + self.robot_x_axis * self.shinai_end_x
        self.shoulder_t_1 = self.shoulder

        # dashing para
        self.dashing_time = 0
        self.dash_start_point = None
        self.dashing_distance = 0

    def __observation_ini__(self):
        self.position_t_1 = self.initial_position
        self.shinai_origin_t_1 = self.shinai_origin
        self.aiming_angle_t_1 = self.initial_angle

    def __state_dic_ini__(self):
        state_ini = {
            #--------------------moving part--------------------#
            'position':self.initial_position,
            'aiming_angle': self.initial_angle,
            'speed':self.initial_speed,
            'speed_t_1':self.initial_speed,
            'x_axis':self.fixed_x_axis,
            'y_axis':self.fixed_y_axis,
            'omega_t_0':self.initial_omega_t_0,
            'omega_t_1': self.initial_omega_t_1,
            'omega_t_2': self.initial_omega_t_2,
            'omega_target_t_1': self.initial_omega_target_t_1,
            #--------------------thrusting part--------------------#
            'shinai_tip':self.shinai_tip,
            'shinai_origin':self.shinai_origin,
            'shinai_thrusting_time':self.shinai_thrusting_time,
            'shinai_tip_t_1':self.shinai_tip,
            'shoulder':self.shoulder,
            'shoulder_t_1':self.shoulder,
            'dashing_time':self.dashing_time,
            'dashing_dis':self.dashing_distance,
            'dash_start_point':None,
            # --------------------observation part--------------------#
            'position_t_1':self.initial_position,
            'shinai_origin_t_1':self.shinai_origin,
            'aiming_angle_t_1':self.initial_angle,
        }
        return state_ini

    # load state dic
    def load_state_dic(self,state_dic):
        # --------------------moving part--------------------#
        self.position = state_dic['position']
        self.aiming_angle = state_dic['aiming_angle']
        self.speed = state_dic['speed']
        self.speed_t_1 = state_dic['speed_t_1']
        self.robot_x_axis = state_dic['x_axis']
        self.robot_y_axis = state_dic['y_axis']
        self.omega_t_0 = state_dic['omega_t_0']
        self.omega_t_1 = state_dic['omega_t_1']
        self.omega_t_2 = state_dic['omega_t_2']
        self.omega_target_t_1 = state_dic['omega_target_t_1']
        # --------------------thrusting part--------------------#
        self.shinai_tip=state_dic['shinai_tip']
        self.shinai_origin=state_dic['shinai_origin']
        self.shinai_thrusting_time=state_dic['shinai_thrusting_time']
        self.shinai_tip=state_dic['shinai_tip']
        self.shoulder=state_dic['shoulder']
        self.shoulder_t_1 =state_dic['shoulder_t_1']
        self.dashing_time=state_dic['dashing_time']
        self.dashing_distance= state_dic['dashing_dis']
        self.dash_start_point = state_dic['position']
        # --------------------observation part--------------------#
        self.position_t_1 = state_dic['position_t_1']
        self.shinai_origin_t_1 = state_dic['shinai_origin_t_1']
        self.aiming_angle_t_1 = state_dic['aiming_angle_t_1']

    # RESET robot
    def reset(self):
        self.load_state_dic(self.state_ini)
        # get the collision_box
        self.robot_collision_box = self.collision_box()

    # save state dic
    # return in dic
    def save_state_dic(self):
        state_ini = {
            # --------------------moving part--------------------#
            'position':self.position,
            'aiming_angle': self.aiming_angle,
            'speed':self.speed,
            'speed_t_1':self.speed_t_1,
            'x_axis':self.robot_x_axis,
            'y_axis':self.robot_y_axis,
            'omega_t_0':self.omega_t_0,
            'omega_t_1': self.omega_t_1,
            'omega_t_2': self.omega_t_2,
            'omega_target_t_1': self.omega_target_t_1,
            # --------------------thrusting part--------------------#
            'shinai_tip': self.shinai_tip,
            'shinai_origin': self.shinai_origin,
            'shinai_thrusting_time': self.shinai_thrusting_time,
            'shinai_tip_t_1': self.shinai_tip,
            'shoulder': self.shoulder,
            'shoulder_t_1': self.shoulder,
            'dashing_time': self.dashing_time,
            'dashing_dis': self.dashing_distance,
            'dash_start_point': None,
            # --------------------observation part--------------------#
            'position_t_1': self.position_t_1,
            'shinai_origin_t_1': self.shinai_origin_t_1,
            'aiming_angle_t_1': self.aiming_angle_t_1,
        }
        return state_ini

    '''
    calculate the real velocity by target speed
    NOTICE: command of target speed should be in *INT*
    '''
    def calculate_velocity(self, target_speed: np.array([float, float, float])):  # target_speed:[Vx,Vy,VC]
        # renew the observation
        self.speed_t_1 = self.speed
        self.aiming_angle_t_1 = self.aiming_angle
        self.position_t_1 = self.position.copy()
        self.shoulder_t_1 = self.shoulder

        # make target_speed integrate
        target_int = list(map(int, target_speed))

        # Forward Kinematics Matrix
        FKM = np.array([
            [1, -1, -0.01 * (self.agv_a + self.agv_b)],
            [1, 1, 0.01 * (self.agv_a + self.agv_b)],
            [1, 1, -0.01 * (self.agv_a + self.agv_b)],
            [1, -1, 0.01 * (self.agv_a + self.agv_b)]
        ])
        omega_target = (FKM.dot(target_int)).astype(float)
        omega_target = np.clip(omega_target, -self.motor_max_speed, self.motor_max_speed)
        # weird stuff of THIS robot
        for i in range(4):
            if self.omega_t_0[i] * omega_target[i] < -1 and not self.omega_t_1[i] * omega_target[i] < -1:
                omega_target[i] = 0.

        # calculate actual speed of every wheel
        omega_mat = np.vstack((self.omega_t_0, self.omega_t_1, self.omega_t_2, omega_target, self.omega_target_t_1))
        omega_real = self.para.dot(omega_mat).astype(float)

        # parameters iteration
        self.omega_target_t_1 = omega_target
        self.omega_t_2 = self.omega_t_1
        self.omega_t_1 = self.omega_t_0
        self.omega_t_0 = omega_real

        # Backward Kinematics Matrix
        a = 0.01 * (self.agv_a + self.agv_b)
        b = 1 / 4
        BKM = np.array([
            [b, b, b, b],
            [-b, b, b, -b],
            [-b * a, b * a, -b * a, b * a]
        ])
        self.speed = BKM.dot(omega_real.T).astype(float)

        # update angle
        self.aiming_angle += self.speed[2] / self.fps
        self.aiming_angle = angle_between(self.aiming_angle, 0)
        # update axis
        self.robot_x_axis = rotate_vector(self.fixed_x_axis, self.aiming_angle)
        self.robot_y_axis = rotate_vector(self.fixed_y_axis, self.aiming_angle)
        # update position
        self.position += (self.speed[0] * self.robot_x_axis / self.fps) + (self.speed[1] * self.robot_y_axis / self.fps)
        self.position = np.clip(self.position, 0, self.grid_size)

        # update the position of arm
        self.shinai_origin = self.position + self.robot_x_axis * self.shinai_end_x + self.robot_y_axis * self.shinai_end_y
        #
        self.shoulder = self.position + self.robot_x_axis * self.shinai_end_x

    '''
    thrusting is currently modeled as three parts
    1. 0.0s - 0.3s thrust forward
    2. 0.3s - 0.4s hold still
    3. 0.4s - 0.5s withdrawal
    change THIS function if a new thrusting model is proposed
    '''
    def calculate_thrusting_length(self, thrust_command: np.array(int)):

        self.shinai_tip_t_1 = self.shinai_tip
        self.shinai_origin_t_1 = self.shinai_origin

        if thrust_command > 0 or self.shinai_thrusting_time > 0:
            self.shinai_thrusting_time += 1

        # setting the threshold
        thrust_threshold = self.fps * self.thrusting_time
        hold_threshold = self.fps * self.holding_time
        withdrawal_threshold = self.fps * self.withdrawal_time

        # update new length
        self.thrusting_length_t_1 = self.shinai_thrusting_length
        # 3 phases
        threshold_1 = thrust_threshold
        threshold_2 = thrust_threshold + hold_threshold
        threshold_3 = thrust_threshold + hold_threshold + withdrawal_threshold

        # judging which phases and the thrust length of current phase
        if 0 < self.shinai_thrusting_time <= threshold_1:
            thrust_length = self.shinai_thrusting_time * self.shinai_thrusting_length / thrust_threshold
        elif threshold_1 < self.shinai_thrusting_time <= threshold_2:
            thrust_length = self.shinai_thrusting_length
        elif threshold_2 < self.shinai_thrusting_time <= threshold_3:
            thrust_length = (1 - (
                    self.shinai_thrusting_time - threshold_2) / withdrawal_threshold) * self.shinai_thrusting_length
        else:
            thrust_length = 0
            self.shinai_thrusting_time = 0

        # update the end
        shinai_end = self.position + self.robot_x_axis * self.shinai_end_x + self.robot_y_axis * self.shinai_end_y
        self.shinai_tip = np.array([shinai_end + self.robot_y_axis * (self.shinai_length + thrust_length),
                                    shinai_end + self.robot_y_axis * (
                                            self.shinai_length + thrust_length - self.shinai_tip_length)])

        # update state
        self.thrusting_length_t = thrust_length

    '''
    dash is now only a conception for robot to determine when ang where the attack starts
    change THIS function to altering dashing model
    the dashing is model as:
    1. record the point robot decide to dash (dash_time = 1)
    2. calculate the distance on robot y_axis (dashi_time > 1)
    3. reset the start point
    '''
    def calculate_dash(self):
        # renew start point
        if self.dashing_time == 0:
            self.dash_start_point = np.clip(self.position, 0, self.grid_size)

        if self.shinai_thrusting_time>0:
            self.dashing_time += 1
        else:
            self.dashing_time = 0

        # calculating dashing distance
        # reset to zero when not dashing
        # print(self.dashing_time)
        if self.dashing_time > 0:
            self.dashing_distance = (self.position - self.dash_start_point).dot(self.robot_y_axis.T)
        else:
            self.dashing_distance = 0.

    # get robot collision box
    def collision_box(self):
        box = []
        wheel_center_points = self._box_mat.dot(
            np.array([[self.robot_collision_box_x, 0], [0, self.robot_collision_box_y]]))
        for i in range(4):
            r_v = rotate_vector(wheel_center_points[i], self.aiming_angle)
            r_v += self.position
            box.append(r_v)
        return box

    '''
    input the command to control the robot
    command is arranged as: [MOVING COMMAND : 3 , THRUSTING COMMAND : 1, DASHING COMMAND : 1]
    '''
    def robot_step(self, action: np.array([int, int, int, int])):
        # MOVING PART
        action = action * self.action_bound
        moving_command = action[:3]
        self.calculate_velocity(moving_command)

        # THRUSTING PART
        thrusting_command = action[3]
        self.calculate_thrusting_length(thrusting_command)
        self.calculate_dash()

        # update collision box
        self.robot_collision_box = self.collision_box()
