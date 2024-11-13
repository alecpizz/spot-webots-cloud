# from scipy.spatial.transform import Rotation as R
import numpy as np
import copy
import time

from SpotKinematics import SpotModel
from Bezier import BezierGait

from controller import Supervisor #, Robot

NUMBER_OF_JOINTS = 12

motions = {
    'stand': [-0.1, 0.0, 0.0, 0.1, 0.0, 0.0, -0.1, 0.0, 0.0, 0.1, 0.0, 0.0],
    'sit'  : [-0.20, -0.40, -0.19, 0.20, -0.40, -0.19, -0.40, -0.90, 1.18, 0.40, -0.90, 1.18],
    'lie'  : [-0.40, -0.99, 1.59, 0.40, -0.99, 1.59, -0.40, -0.99, 1.59, 0.40, -0.99, 1.59],
}


def inverse_lerp(a, b, v):
    return (v - a) / (b - a)

def lerp(a, b, t):
        return (1.0 - t) * a + b * t

DOGZILLA_FORWARD_TRAVERSAL = 25
DOGZILLA_STRAFE_TRAVERSAL = 18
DOGZILLA_TURN_TRAVERSAL = 100

def remap(i_min, i_max, o_min, o_max, v):
    t = inverse_lerp(i_min, i_max, v)
    return lerp(o_min, o_max, t)

class SpotDriver:
    def __init__(self):
        # self.robot = Robot()
        self.robot = Supervisor()


        self.spot_node = self.robot.getFromDef("Spot")

        self.spot_translation = self.spot_node.getField('translation')
        self.spot_translation_initial = self.spot_translation.getSFVec3f()
        self.spot_rotation = self.spot_node.getField('rotation')
        self.spot_rotation_initial = self.spot_rotation.getSFRotation()

        self.timestep = 32

        ### Init motors
        self.motor_names = [
            "front left shoulder abduction motor",  "front left shoulder rotation motor",  "front left elbow motor",
            "front right shoulder abduction motor", "front right shoulder rotation motor", "front right elbow motor",
            "rear left shoulder abduction motor",   "rear left shoulder rotation motor",   "rear left elbow motor",
            "rear right shoulder abduction motor",  "rear right shoulder rotation motor",  "rear right elbow motor"
        ]
        self.motors = []
        for motor_name in self.motor_names:
            self.motors.append(self.robot.getDevice(motor_name))
        

        ## Positional Sensors
        self.motor_sensor_names = [name.replace('motor', 'sensor') for name in self.motor_names]
        self.motor_sensors = []
        self.motors_pos = []
        for idx, sensor_name in enumerate(self.motor_sensor_names):
            self.motor_sensors.append(self.robot.getDevice(sensor_name))
            self.motor_sensors[idx].enable(self.timestep)
            self.motors_pos.append(0.)

        ## Webots Touch Sensors
        self.touch_fl = self.robot.getDevice("front left touch sensor")
        self.touch_fr = self.robot.getDevice("front right touch sensor")
        self.touch_rl = self.robot.getDevice("rear left touch sensor")
        self.touch_rr = self.robot.getDevice("rear right touch sensor")

        self.touch_fl.enable(self.timestep)
        self.touch_fr.enable(self.timestep)
        self.touch_rl.enable(self.timestep)
        self.touch_rr.enable(self.timestep)

        ## LIDAR
        self.lidar = self.robot.getDevice("SickS300")
        self.lidar.enable(self.timestep)

        ## Spot Control
        self.time_step = self.timestep

        self.spot = SpotModel()
        self.T_bf0 = self.spot.WorldToFoot
        self.T_bf = copy.deepcopy(self.T_bf0)
        self.bzg = BezierGait(dt=0.032)

        # ------------------ Inputs for Bezier Gait control ----------------
        self.xd = 0.0
        self.yd = 0.0
        self.zd = 0.0
        self.rolld = 0.0
        self.pitchd = 0.0
        self.yawd = 0.0
        self.StepLength = 0.0
        self.LateralFraction = 0.0
        self.YawRate = 0.0
        self.StepVelocity = 0.0
        self.ClearanceHeight = 0.0
        self.PenetrationDepth = 0.0
        self.SwingPeriod = 0.0
        self.YawControl = 0.0
        self.YawControlOn = False

        # ------------------ Spot states ----------------
        self.x_inst = 0.
        self.y_inst = 0.
        self.z_inst = 0.
        self.roll_inst = 0.
        self.pitch_inst = 0.
        self.yaw_inst = 0.
        self.search_index = -1

        # ------------------ Outputs of Contact sensors ----------------
        self.front_left_lower_leg_contact = 1
        self.front_right_lower_leg_contact = 1
        self.rear_left_lower_leg_contact = 1
        self.rear_right_lower_leg_contact = 1
        self.chattering_front_left_lower_leg_contact = 0
        self.chattering_front_right_lower_leg_contact = 0
        self.chattering_rear_left_lower_leg_contact = 0
        self.chattering_rear_right_lower_leg_contact = 0
        self.lim_chattering = 4

        # Motion command
        self.fixed_motion = False

        self.n_steps_to_achieve_target = 0
        self.step_difference = [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]
        self.m_target = []
        self.paw = False
        self.paw2 = False
        self.paw_time = 0.
        self.previous_cmd = False

        # UR3e and Gripper
        self.u3re_n_steps_to_achieve_target = 0
        self.ur3e_targets = []
        self.ur3e_target = []
        self.allow_new_target = False
        self.gripper_close = False

    def __model_cb(self):
        spot_rot = self.spot_node.getField("rotation")
        spot_rot_val = spot_rot.getSFRotation()
        self.yaw_inst = spot_rot_val[2]

    def _yaw_control(self):
        """ Yaw body controller"""
        yaw_target = self.YawControl
        thr = np.pi / 2
        if (yaw_target > thr and self.yaw_inst < -thr) or (self.yaw_inst > thr and yaw_target < -thr):
            residual = (yaw_target - self.yaw_inst) * np.sign(yaw_target - self.yaw_inst) - 2 * np.pi
            yawrate_d = 2.0 * np.sqrt(abs(residual)) * np.sign(residual)
        else:
            residual = yaw_target - self.yaw_inst
            yawrate_d = 4.0 * np.sqrt(abs(residual)) * np.sign(residual)
        return yawrate_d


    def _cmd_vel(self, linearx, lineary, angularz):
        # Override motion command
        self.fixed_motion = False

        StepLength = 0.15
        ClearanceHeight = 0.015
        PenetrationDepth = 0.003
        SwingPeriod = 0.3
        YawControl = 0.0
        YawControlOn = 0.0
        StepVelocity = 0.8

        self.xd = 0.
        self.yd = 0.
        # self.zd = 0.
        self.rolld = 0.
        self.pitchd = 0.
        self.yawd = 0.
        self.StepLength = StepLength * linearx
        if self.StepLength == 0.0:
            self.StepLength = StepLength * abs(lineary)
        # Rotation along vertical axis
        self.YawRate = angularz
        if self.YawRate != 0 and self.StepLength == 0:
            self.StepLength = StepLength * 0.1

        # Holonomic motions
        self.LateralFraction = np.arctan2(lineary, linearx)
        # forcefully set 0, as output is never 0 if second term in arctan2 is -ve
        self.LateralFraction = np.arctan2(lineary, abs(linearx))
        if linearx < 0:
            self.LateralFraction *= -1

        self.StepVelocity = StepVelocity
        self.ClearanceHeight = ClearanceHeight
        self.PenetrationDepth = PenetrationDepth
        self.SwingPeriod = SwingPeriod
        self.YawControl = YawControl
        self.YawControlOn = YawControlOn

    def __talker(self, motors_target_pos):
        motor_offsets = [0, 0.52, -1.182]
        for idx, motor in enumerate(self.motors):
            motor.setPosition(motor_offsets[idx % 3] + motors_target_pos[idx])

    def __spot_inverse_control(self):
        self.front_left_lower_leg_contact = self.touch_fl.getValue()
        self.front_right_lower_leg_contact = self.touch_fr.getValue()
        self.rear_left_lower_leg_contact = self.touch_rl.getValue()
        self.rear_right_lower_leg_contact = self.touch_rr.getValue()

        pos = np.array([self.xd, self.yd, self.zd])
        orn = np.array([self.rolld, self.pitchd, self.yawd])

        # yaw controller
        if self.YawControlOn == 1.0:
            YawRate_desired = self._yaw_control()
        else:
            YawRate_desired = self.YawRate

        # Update Swing Period
        self.bzg.Tswing = self.SwingPeriod
        contacts = [self.front_left_lower_leg_contact, self.front_right_lower_leg_contact,
                    self.rear_left_lower_leg_contact,
                    self.rear_right_lower_leg_contact]

        # Get Desired Foot Poses
        T_bf = self.bzg.GenerateTrajectory(self.StepLength, self.LateralFraction, YawRate_desired,
                                           self.StepVelocity, self.T_bf0, self.T_bf,
                                           self.ClearanceHeight, self.PenetrationDepth,
                                           contacts)
        joint_angles = -self.spot.IK(orn, pos, T_bf)

        target = [
            joint_angles[0][0], joint_angles[0][1], joint_angles[0][2],
            joint_angles[1][0], joint_angles[1][1], joint_angles[1][2],
            joint_angles[2][0], joint_angles[2][1], joint_angles[2][2],
            joint_angles[3][0], joint_angles[3][1], joint_angles[3][2],
            ]

        self.__talker(target)

    def __handle_transforms_and_odometry(self):
        for idx, motor_sensor in enumerate(self.motor_sensors):
            self.motors_pos[idx] = motor_sensor.getValue()

        time_stamp = self.__node.get_clock().now().to_msg()

        for idx, ur3e_sensor in enumerate(self.ur3e_sensors):
            self.ur3e_pos[idx] = ur3e_sensor.getValue()

        for idx, gripper_sensor in enumerate(self.gripper_sensors):
            self.gripper_pos[idx] = gripper_sensor.getValue()

        for idx, gripper_sensor in enumerate(self.remaining_gripper_sensors):
            self.remaining_gripper_pos[idx] = gripper_sensor.getValue()

    def __movement_decomposition(self, target, duration):
        """
        Decompose big motion into smaller motions
        """
        self.previous_cmd = True

        self.n_steps_to_achieve_target = duration * 1000 / self.timestep
        self.step_difference = [(target[i] - self.motors_pos[i]) / self.n_steps_to_achieve_target
                                for i in range(NUMBER_OF_JOINTS)]
        self.m_target = []

    def __stand_motion_cb(self, request, response):
        self.fixed_motion = True
        if self.previous_cmd and not request.override:
            response.answer = 'performing previous command, override with bool argument'
            return response

        self.paw = False
        self.paw2 = False
        self.__movement_decomposition(motions['stand'], 1)
        response.answer = 'standing up'
        return response

    def __sit_motion_cb(self, request, response):
        self.fixed_motion = True
        if self.previous_cmd and not request.override:
            response.answer = 'performing previous command, override with bool argument'
            return response

        self.paw = False
        self.paw2 = False
        self.__movement_decomposition(motions['sit'], 1)
        response.answer = 'sitting down'
        return response

    def __lie_motion_cb(self, request, response):
        self.fixed_motion = True
        if self.previous_cmd and not request.override:
            response.answer = 'performing previous command, override with bool argument'
            return response

        self.paw = False
        self.paw2 = False
        self.__movement_decomposition(motions['lie'], 1)
        response.answer = 'lying down'
        return response

    def __shakehand_motion_cb(self, request, response):
        self.fixed_motion = True
        if self.previous_cmd and not request.override:
            response.answer = 'performing previous command, override with bool argument'
            return response

        # Start handshake motion
        self.__movement_decomposition([-0.20, -0.30, 0.05,
                                        0.20, -0.40, -0.19,
                                        -0.40, -0.90, 1.18,
                                        0.49, -0.90, 0.80], 1)
        self.paw = True
        response.answer = 'shaking hands'
        return response

    def __defined_motions(self):
        self.__handle_transforms_and_odometry() # Let the sensor values get updated
        if self.n_steps_to_achieve_target > 0:
            if not self.m_target:
                self.m_target = [self.step_difference[i] + self.motors_pos[i] for i in range(NUMBER_OF_JOINTS)]
            else: # if compared to current motors_positions, the final motion is smaller
                self.m_target = [self.step_difference[i] + self.m_target[i] for i in range(NUMBER_OF_JOINTS)]

            # Increment motor positions by step_difference
            for idx, motor in enumerate(self.motors):
                motor.setPosition(self.m_target[idx])
            
            self.n_steps_to_achieve_target -= 1
        else:
            if self.paw:
                self.paw_time = self.robot.getTime() + 4
                self.paw = False
                self.paw2 = True # Do the shakehands motion
            else:
                self.previous_cmd = False
                
            self.m_target = []

        if self.paw2:
            self.previous_cmd = True
            if self.paw_time > self.robot.getTime():
                self.motors[4].setPosition(0.2 * np.sin(2 * self.robot.getTime()) + 0.6)
                self.motors[5].setPosition(0.4 * np.sin(2 * self.robot.getTime()))
            else:
                self.paw2 = False # Sit back again
                self.__movement_decomposition([-0.20, -0.40, -0.19,
                                              0.20, -0.40, -0.19,
                                             -0.40, -0.90, 1.18,
                                              0.40, -0.90, 1.18], 1)

    def __close_gripper_cb(self, request, response):
        if self.gripper_close:
            response.answer = 'gripper already close'
            return response
        response.answer = 'closing gripper'
        self.gripper_close = True
        return response

    def __open_gripper_cb(self, request, response):
        if not self.gripper_close:
            response.answer = 'gripper already open'
            return response
        response.answer = 'opening gripper'
        self.gripper_close = False
        return response

    def __spot_height_cb(self, request, response):
        if not -0.3 <= request.height <= 0:
            response.answer = 'set height within -0.3 and 0'
            return response
        self.zd = -request.height
        return response

    def __ur3e_defined_motions(self):
        if self.u3re_n_steps_to_achieve_target > 0:
            if not self.ur3e_target:
                self.ur3e_target = [self.ur3e_step_difference[i] + self.ur3e_pos[i] for i in range(len(self.ur3e_motor_names))]
            else: # if compared to current motors_positions, the final motion is smaller
                self.ur3e_target = [self.ur3e_step_difference[i] + self.ur3e_target[i] for i in range(len(self.ur3e_motor_names))]

            # Increment motor positions by step_difference
            for idx, motor in enumerate(self.ur3e_motors):
                motor.setPosition(self.ur3e_target[idx])
            
            self.u3re_n_steps_to_achieve_target -= 1
        elif len(self.ur3e_targets) > 0:
            # Forcefully go to target, moveit2 expect a deviation of max 0.01
            for idx, motor in enumerate(self.ur3e_motors):
                motor.setPosition(self.ur3e_targets[0][idx])
            next_target = True
            for position, target in zip(self.ur3e_pos, self.ur3e_targets[0]):
                if abs(position - target) > 0.04:
                    next_target = False
                    break
            if next_target:
                self.ur3e_targets.pop(0)
        else:
            self.allow_new_target = True

    def __moveit_movement_decomposition(self, duration):
        """
        Decompose big motion into smaller motions
        """
        if len(self.ur3e_targets) > 0 and self.allow_new_target:
            self.allow_new_target = False
            target = self.ur3e_targets[0]
            self.u3re_n_steps_to_achieve_target = duration * 1000 / self.timestep
            self.ur3e_step_difference = [(target[i] - self.ur3e_pos[i]) / self.u3re_n_steps_to_achieve_target
                                    for i in range(7)]


    def move_in_direction_for_duration(self, linear_x: float = 0.0, linear_y: float = 0.0,
                                       angular_z: float = 0.0, move_time: float = 0.0):
        end_time = time.time() + move_time
        while self.robot.step(self.timestep) != -1 and time.time() < end_time:
            self._cmd_vel(linear_x, linear_y, angular_z)
            self.__spot_inverse_control()
            self.__model_cb()



    def forward(self, step):
        self._cmd_vel(remap(0, DOGZILLA_FORWARD_TRAVERSAL, 0, 0.5, step), 0, 0)
        self.__spot_inverse_control()
        self.__model_cb()

    def back(self, step):
        self._cmd_vel(-remap(0, DOGZILLA_FORWARD_TRAVERSAL, 0, 0.5, step), 0, 0)
        self.__spot_inverse_control()
        self.__model_cb()

    def left(self, step):
        self._cmd_vel(0, remap(0, DOGZILLA_STRAFE_TRAVERSAL, 0, 0.5, step), 0)
        self.__spot_inverse_control()
        self.__model_cb()

    def right(self, step):
        self._cmd_vel(0, -remap(0, DOGZILLA_STRAFE_TRAVERSAL, 0, 0.5, step), 0)
        self.__spot_inverse_control()
        self.__model_cb()

    def turnleft(self, step):
        self._cmd_vel(0, 0, remap(0, DOGZILLA_TURN_TRAVERSAL, 0, 0.5, step))
        self.__spot_inverse_control()
        self.__model_cb()

    def turnright(self, step):
        self._cmd_vel(0, 0, -remap(0, DOGZILLA_TURN_TRAVERSAL, 0, 0.5, step))
        self.__spot_inverse_control()
        self.__model_cb()

    def stop(self):
        self._cmd_vel(0, 0, 0)
        self.__spot_inverse_control()
        self.__model_cb()

    def move_forward(self, move_time):
        self.move_in_direction_for_duration(0.5, 0.0, 0.0, move_time)

    def move_backward(self, move_time):
        self.move_in_direction_for_duration(-0.5, 0.0, 0.0, move_time)

    def turn_left(self, move_time):
        self.move_in_direction_for_duration(0.0, 0.0, 0.5, move_time)

    def turn_right(self, move_time):
        self.move_in_direction_for_duration(0.0, 0.0, -0.5, move_time)

    def move_forward_and_right(self, move_time):
        self.move_in_direction_for_duration(0.5, -0.5, 0.0, move_time)

    def move_forward_and_left(self, move_time):
        self.move_in_direction_for_duration(0.5, 0.5, 0.0, move_time)

    def move_backward_and_right(self, move_time):
        self.move_in_direction_for_duration(-0.5, -0.5, 0.0, move_time)

    def move_backward_and_left(self, move_time):
        self.move_in_direction_for_duration(-0.5, 0.5, 0.0, move_time)

    def stop_moving(self, move_time):
        self.move_in_direction_for_duration(0.0, 0.0, 0.0, move_time)

    def get_lidar_image(self):
        return self.lidar.getRangeImage()

    def get_timestep(self):
        return self.timestep

    def step(self, timestep):
       return self.robot.step(timestep)