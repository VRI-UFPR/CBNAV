#!/usr/bin/env python3

import math
import time
from .environment_node_data import NodeData, Mode
from math import sqrt

import numpy as np

import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from scipy.ndimage import gaussian_filter

from astropy.convolution import convolve, Gaussian2DKernel, Tophat2DKernel, AiryDisk2DKernel


DEBUG = False

class FitnessData:
    """
    Class for calculating the fitness function reward from the current simulation state. Possible reward calculation
    are: travelled distance, distance to the target node, angle to the target node etc.
    """

    def __init__(self):
        """
        Constructor of the FitnessData class.
        """
        self._node_data = NodeData()

        self._terminate_at_end = True

        self._robot_x_last = 0.0
        self._robot_y_last = 0.0
        self._robot_orientation_last = 0.0


        self._x_max = 0
        self._y_max = 0

        self._x_last = 0
        self._y_last = 0

        self._distance_avg = 0.0
        self._distance_avg_last = 0.0

        self._reward_matrix = np.array( [ [] , [] ] )
        self._path_matrix = np.array( [ [] , [] ] )

        self._create_matrix = False
        self._create_path_matrix = False

        self._distance_end = 0

        self._gaussian_count = 0

        self.fig, (self.ax1, self.ax2) = plt.subplots(2)

    def get_end_node(self):
        """
        Get the current selected end Node.
        :return:
        """
        return self._node_data.get_node_end()

    def init(self, filename="") -> bool:
        """
        Read the node file and initialized the node data.
        :param filename:
        :return:
        """
        return self._node_data.read_node_file(filename)

    def set_mode(self, mode: Mode, terminate_at_end=True):
        """
        Set the mode for the simulation. More information in environment.py.
        :param mode: Simulation mode.
        :param terminate_at_end: Done when the target node is reached.
        :return:
        """
        self._node_data.set_mode(mode)
        self._terminate_at_end = terminate_at_end

    def _distance_start_to_end(self) -> float:
        """
        Calculate the distance between the start node and end node.
        :return: Distance between start and end node.
        """
        return self._distance(
            self._node_data.get_node_start().x(),
            self._node_data.get_node_start().y(),
            self._node_data.get_node_end().x(),
            self._node_data.get_node_end().y())

    def _distance_robot_to_end(self, robot_x: float, robot_y: float) -> float:
        """
        Calculate the distance between the robot position to the end node.
        :param robot_x: Robot position x.
        :param robot_y: Robot position y.
        :return: Distance between robot and end node.
        """
        return self._distance(
            robot_x,
            robot_y,
            self._node_data.get_node_end().x(),
            self._node_data.get_node_end().y())

    def _distance_between_last_step(self, robot_x: float, robot_y: float) -> float:
        """
        Calculate the distance from the last robot position to the current robot position.
        :param robot_x: Robot position x.
        :param robot_y: Robot position y.
        :return: Distance from last and current robot position.
        """
        return self._distance(
            robot_x,
            robot_y,
            self._robot_x_last,
            self._robot_y_last)

    def _orientation_robot_to_end(self, robot_x: float, robot_y: float) -> float:
        """
        Calculate the orientation from the robot position to the end node.
        :param robot_x: Robot position x.
        :param robot_y: Robot position y.
        :return: Orientation from robot position to end node.
        """
        x = self._node_data.get_node_end().x() - robot_x
        y = self._node_data.get_node_end().y() - robot_y
        return math.atan2(y, x)

    def angle_difference_from_robot_to_end(self, robot_x: float, robot_y: float, robot_orientation: float) -> float:
        """
        Calculate the angle difference from robot to end node.
        :param robot_x: Robot position x.
        :param robot_y: Robot position y.
        :return: Angle difference robot to end node.
        """
        return self._difference_two_angles(robot_orientation, self._orientation_robot_to_end(robot_x, robot_y))

    @staticmethod
    def _difference_two_angles(angle1: float, angle2: float) -> float:
        """
        Calculate the difference between to angle.
        :param angle1: First angle.
        :param angle2: Second angle.
        :return: Difference between to angle.
        """
        diff = (angle1 - angle2) % (math.pi * 2)
        if diff >= math.pi:
            diff -= math.pi * 2
        return diff

    @staticmethod
    def _distance(x1: float, y1: float, x2: float, y2: float) -> float:
        """
        Calculate the euler distance from to point.
        :param x1: First point x.
        :param y1: First point y.
        :param x2: Second point x.
        :param y2: Second point y.
        :return: Euler distnace from to points.
        """
        x = x1 - x2
        y = y1 - y2
        return math.sqrt(x*x + y*y)

    def reset(self):
        """
        Reset call from the environment to select new node.
        :return:
        """
        self._create_path_matrix = False
        self._node_data.new_node_selection()

    def calculate_reward(self, robot_x: float, robot_y: float, 
        robot_orientation: float, env_done: bool, 
        laser_min_distance: float, laser_min_distance_angle: float, 
        laser_max_distance: float, laser_max_distance_angle: float,
        test_num: int,
        n_step: int,
        n_episode: int):
        """
        Calculate the reward from one step.
        :param robot_x: Robot position x.
        :param robot_y: Robot position y.
        :param robot_orientation: Robot orientation angle.
        :param env_done: Done when the robot collided with the robot.
        :return: Reward, done.
        """
        done = False
        reward = 0.0

        #Abstand zwinschen P(t-1) und P(t) --> #1
        distance_between_last_step = self._distance_between_last_step(robot_x, robot_y)

        #Abstand zwinschen P(t) und Endposition -->#2
        distance_robot_to_end = self._distance_robot_to_end(robot_x, robot_y)

        #Abstand zwinschen P(t-1) und Endposition -->#3
        distance_robot_to_end_last = self._distance_robot_to_end(self._robot_x_last, self._robot_y_last)

        # -->#4
        distance_robot_to_end_diff = distance_robot_to_end_last - distance_robot_to_end;
        # -->#5
        distance_robot_to_end_diff_abs = abs(distance_robot_to_end_diff)

        diff_rotation_to_end_last = self.angle_difference_from_robot_to_end(self._robot_x_last, self._robot_y_last,self._robot_orientation_last)

        diff_rotation_to_end = self.angle_difference_from_robot_to_end(robot_x, robot_y,robot_orientation)

        rotations_cos_sum =  math.cos(diff_rotation_to_end) #  [ -1 , 1]

        diff_rotations = math.fabs(math.fabs(diff_rotation_to_end_last) - math.fabs(diff_rotation_to_end))   # [0 , pi]

        if distance_between_last_step != 0:
            distance_robot_to_end_diff_abs = distance_robot_to_end_diff_abs/distance_between_last_step # Normalization to [0 , 1]
        else:
            distance_robot_to_end_diff_abs = 0

        if distance_robot_to_end > sqrt(distance_between_last_step**2 + distance_robot_to_end_last**2):
            distance_robot_to_end_diff_abs *= -6.0 # [-6,0]
        else:
            distance_robot_to_end_diff_abs *= 6.0 #[0, 6]

        if math.fabs(diff_rotation_to_end) > math.fabs(diff_rotation_to_end_last):

            diff_rotations *= -3.0 # [-3xpi,0]
        else:
            diff_rotations *=  2.0 # [0,2xpi]

        if (test_num == 1 or test_num == 2 or test_num == 3 or test_num == 4 or test_num == 5):
            reward += distance_robot_to_end_diff_abs
            reward += (3*rotations_cos_sum)
            reward += diff_rotations

            if env_done:
                reward = -20
                done = True
            elif n_step == 400:
                reward = -20
                done = True
            elif distance_robot_to_end < self._node_data.get_node_end().radius():
                reward = 20
                done = self._handle_terminate_at_end()
       
        elif (test_num == 6 or test_num == 7 or test_num == 8 or test_num == 9 or test_num == 10):   

            PATH_MATRIX_START = 1
            PATH_MATRIX_DECREASE = 1
            REWARD_MATRIX_START = -1
            REWARD_MATRIX_CRITICAL_0 = 0
            REWARD_MATRIX_CRITICAL_1 = 0
            REWARD_MATRIX_CRITICAL_2 = 0
            REWARD_MATRIX_CRITICAL_3 = 0
            REWARD_MATRIX_CRITICAL_4 = 4
            REWARD_MATRIX_CRITICAL_5 = 5
            REWARD_MATRIX_SAFETY = 6

            if self._create_matrix == False:
                x = robot_x * 10
                y = robot_y * 10

                x = int(x)
                y = int(y)

                print ('x: {}'.format(x))
                print ('y: {}'.format(y))

                print ('robot_x: {}'.format(robot_x))
                print ('robot_y: {}'.format(robot_y))

                self._reward_matrix = np.full((x+1, y+1), REWARD_MATRIX_START, dtype=float)
                #self._reward_matrix = np.load('new_results/reward_matrix_'+str(test_num)+'.npy', allow_pickle=True)

                self._x_max = x+1
                self._y_max = y+1

                print ('self._x_max: {}'.format(self._x_max))
                print ('self._y_max: {}'.format(self._y_max))
                self._create_matrix = True

                self._x_last = x
                self._y_last = y

                '''                
                kernel = np.array([np.array([0.077847, 0.123317, 0.077847]), np.array([0.123317, 0.195346, 0.123317]), np.array([0.077847, 0.123317, 0.077847])])

                bias = 1
                m, n = kernel.shape
                y, x = self._reward_matrix.shape
                y = y - m + 1
                x = x - m + 1
                new_image = np.zeros((y,x))
                for i in range(y):
                    for j in range(x):
                        if self._reward_matrix[i,j] != -1:
                            new_image[i][j] = np.sum(self._reward_matrix[i:i+m, j:j+m]*kernel) + bias
                        else:
                            new_image[i][j] = -1

                np.save('new_results/conv_reward_matrix_'+str(test_num)+'.npy', new_image)

                return 0
                '''
                #gauss_kernel = Gaussian2DKernel(2)
                #smoothed_data_gauss = convolve(self._reward_matrix, gauss_kernel)
                #np.save('new_results/conv_reward_matrix_'+str(test_num)+'.npy',  smoothed_data_gauss)

            else:
                x = robot_x * 10
                y = robot_y * 10
                x = int(x)
                y = int(y)

                if (int(robot_x * 10)+1) > self._x_max:
                    x = int(robot_x * 10) + 1
                    y = int(self._y_max)

                    _x = x - int(self._x_max)

                    _reward_matrix_temp = np.full((_x, y), REWARD_MATRIX_START, dtype=float)
                    self._reward_matrix = np.append( self._reward_matrix, _reward_matrix_temp, axis=0)

                    _path_matrix_temp = np.full((_x, y), PATH_MATRIX_START, dtype=float)
                    self._path_matrix = np.append( self._path_matrix, _path_matrix_temp, axis=0)
                    
                    self._x_max = int(robot_x*10) + 1

                    print('self._x_max: {}'.format(self._x_max))

                if (int(robot_y * 10)+1) > self._y_max:
                    x = int(self._x_max)
                    y = int(robot_y * 10) + 1

                    _y = y - int(self._y_max)

                    _reward_matrix_temp = np.full((x, _y), REWARD_MATRIX_START, dtype=float)
                    self._reward_matrix = np.append( self._reward_matrix, _reward_matrix_temp, axis=1)

                    _path_matrix_temp = np.full((x, _y), PATH_MATRIX_START, dtype=float)
                    self._path_matrix = np.append( self._path_matrix, _path_matrix_temp, axis=1)

                    self._y_max = int(robot_y*10) + 1

                    print('self._y_max: {}'.format(self._y_max))

                if n_step == 0:
                    self._x_last = int(robot_x * 10)
                    self._y_last = int(robot_y * 10)

            if self._create_path_matrix == False:
                self._path_matrix = np.full((self._x_max, self._y_max), PATH_MATRIX_START, dtype=float)
                self._create_path_matrix = True


            if env_done:
                self._reward_matrix[self._x_last, self._y_last] = REWARD_MATRIX_CRITICAL_0
                reward = -180
                done = True
                np.save('new_results/path_matrix_'+str(test_num)+'.npy', self._path_matrix)
                np.save('new_results/reward_matrix_'+str(test_num)+'.npy', self._reward_matrix)
            elif n_step == 300:
                reward = -130
                done = True
            elif distance_robot_to_end < self._node_data.get_node_end().radius():
                reward = 180
                print('Wir haben gewonnen')
                done = self._handle_terminate_at_end()
                np.save('new_results/path_matrix_'+str(test_num)+'.npy', self._path_matrix)
                np.save('new_results/reward_matrix_'+str(test_num)+'.npy', self._reward_matrix)

            else:
                if (self._x_last != int(robot_x*10) or self._y_last != int(robot_y*10)):
                    #reward matrix
                    if(self._reward_matrix[self._x_last, self._y_last] != REWARD_MATRIX_CRITICAL_5):
                        if(self._reward_matrix[self._x_last, self._y_last] != REWARD_MATRIX_CRITICAL_4):
                            if(self._reward_matrix[self._x_last, self._y_last] != REWARD_MATRIX_CRITICAL_3):
                                if(self._reward_matrix[self._x_last, self._y_last] != REWARD_MATRIX_CRITICAL_2):
                                    if(self._reward_matrix[self._x_last, self._y_last] != REWARD_MATRIX_CRITICAL_1):
                                        if(self._reward_matrix[self._x_last, self._y_last] != REWARD_MATRIX_CRITICAL_0):
                                            self._reward_matrix[self._x_last, self._y_last] = REWARD_MATRIX_SAFETY
                                        else:
                                            self._reward_matrix[self._x_last, self._y_last] = REWARD_MATRIX_CRITICAL_1
                                    else:
                                        self._reward_matrix[self._x_last, self._y_last] = REWARD_MATRIX_CRITICAL_2
                                else:
                                    self._reward_matrix[self._x_last, self._y_last] = REWARD_MATRIX_CRITICAL_3
                            else:
                                self._reward_matrix[self._x_last, self._y_last] = REWARD_MATRIX_CRITICAL_4
                        else:
                            self._reward_matrix[self._x_last, self._y_last] = REWARD_MATRIX_CRITICAL_5

                #np.save('new_results/path_matrix_'+str(test_num)+'.npy', self._path_matrix)
                #np.save('new_results/reward_matrix_'+str(test_num)+'.npy', self._reward_matrix)

                alpha = 1.5
                #Distance between end point and reference point
                ef = self._distance(self._node_data.get_node_end().x(), self._node_data.get_node_end().y(), self._node_data.get_node_start().x(), self._node_data.get_node_start().y())
                r1 = (ef/(ef + distance_robot_to_end)) #- 0.5
                #r1 = (ef/distance_robot_to_end) - 1
                r1 = r1*alpha #[0 <-> alpha]

                #r1 = self._distance_end - distance_robot_to_end

                theta = 1.0
                x = self._node_data.get_node_end().x() - robot_x
                y = self._node_data.get_node_end().y() - robot_y
                diff = (robot_orientation - math.atan2(y, x)) % (math.pi * 2) #[0 <-> 2pi]
                if diff >= math.pi:
                    diff -= math.pi * 2
                    diff = abs(diff)

                r3 = 1 - (diff/math.pi)   #[0 <-> 1]
                r3 = r3 * theta  #[0 <-> theta]

                if (self._reward_matrix[self._x_last, self._y_last] >= REWARD_MATRIX_CRITICAL_0 and self._reward_matrix[self._x_last, self._y_last] <= REWARD_MATRIX_CRITICAL_1):
                    reward = -10
                elif (self._reward_matrix[self._x_last, self._y_last] > REWARD_MATRIX_CRITICAL_1 and self._reward_matrix[self._x_last, self._y_last] <= REWARD_MATRIX_CRITICAL_2):
                    reward = -8
                elif (self._reward_matrix[self._x_last, self._y_last] > REWARD_MATRIX_CRITICAL_2 and self._reward_matrix[self._x_last, self._y_last] <= REWARD_MATRIX_CRITICAL_3):
                    reward = -7
                elif (self._reward_matrix[self._x_last, self._y_last] > REWARD_MATRIX_CRITICAL_3 and self._reward_matrix[self._x_last, self._y_last] <= REWARD_MATRIX_CRITICAL_4):
                    reward = -6
                elif (self._reward_matrix[self._x_last, self._y_last] > REWARD_MATRIX_CRITICAL_4 and self._reward_matrix[self._x_last, self._y_last] <= REWARD_MATRIX_CRITICAL_5):
                    reward = -5
                elif r1 < 0:
                    reward = (r1 * self._reward_matrix[self._x_last, self._y_last])
                    reward += (r3 * self._reward_matrix[self._x_last, self._y_last] * self._path_matrix[self._x_last, self._y_last])
                else:
                    reward = (r1 * self._reward_matrix[self._x_last, self._y_last] * self._path_matrix[self._x_last, self._y_last])
                    reward += (r3 * self._reward_matrix[self._x_last, self._y_last] * self._path_matrix[self._x_last, self._y_last])

                #pathmatrix
                self._path_matrix[self._x_last, self._y_last] -= PATH_MATRIX_DECREASE

            self._x_last = int(robot_x*10)
            self._y_last = int(robot_y*10) 

            if self._gaussian_count == 20_000:
                gauss_kernel = AiryDisk2DKernel(5)
                smoothed_data_gauss = convolve(self._reward_matrix, gauss_kernel)
                self._reward_matrix = smoothed_data_gauss
                self._gaussian_count = 0
            else:
                self._gaussian_count += 1
 
        self._robot_x_last = robot_x
        self._robot_y_last = robot_y
        self._robot_orientation_last = robot_orientation

        return reward, done

    def _distance_robot_to_start(self, robot_x: float, robot_y: float) -> float:
        """
        Calculate the distance between the robot position to the end node.
        :param robot_x: Robot position x.
        :param robot_y: Robot position y.
        :return: Distance between robot and end node.
        """
        return self._distance(
            robot_x,
            robot_y,
            self._node_data.get_node_start().x(),
            self._node_data.get_node_start().y())

    def get_robot_start(self):
        """
        Get the start node for the start position for the robot.
        :return: Start node.
        """
        return self._node_data.generate_robot_start_position()

    def _handle_terminate_at_end(self):
        """
        Select a new target node when the simulation souldn't terminate when reaching the target node.
        :return:
        """
        if not self._terminate_at_end:
            self._node_data.new_end_node()

        return self._terminate_at_end

