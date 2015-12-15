__author__ = "Egor Gavrilov"
__copyright__ = "Copyright 2015, ITMO University"
__license__ = "MIT"
__version__ = "1.0.0"
__email__ = "egorvlgavr@gmail.com"

import numpy as np
import drawlines as drwlines

# Coordinates of four points
r_a = np.array([0.165, 0.1, 0.04])
r_b = np.array([0.145, 0.25, 0.3])
r_c = np.array([0.145, -0.25, 0.3])
r_d = np.array([0.165, -0.1, 0.04])
# Time
T = np.array([0, 3, 6, 9])
# Constants of robot
L = np.array([0.185, 0.165, 0.15])


# 1. Solve inverse problem of kinematics for four points A,B,C,D
# this function calculate vector of angles
def calulate_inverse_kinematics(r_general, L):
    thetha_1 = np.arctan2(r_general[1], r_general[0])
    D = (pow(r_general[0], 2) + pow(r_general[1], 2) + pow(r_general[2] - L[0], 2) - pow(L[1], 2) - pow(L[2], 2)) / (
        2 * L[1] * L[2])
    thetha_3 = np.arctan2(np.sqrt(1 - pow(D, 2)), D)
    r = np.sqrt(pow(r_general[0], 2) + pow(r_general[1], 2))
    s = r_general[2] - L[0]
    thetha_2 = np.arctan2(s, r) - np.arctan2(L[2] * np.sin(thetha_3), L[1] + L[2] * np.cos(thetha_3))
    return np.array([thetha_1, thetha_2, thetha_3])


q_a = calulate_inverse_kinematics(r_a, L)
q_b = calulate_inverse_kinematics(r_b, L)
q_c = calulate_inverse_kinematics(r_c, L)
q_d = calulate_inverse_kinematics(r_d, L)

# 2. Normalization of time
time_interval = 3;
tau = np.linspace(0.0, time_interval, num=201)
for x in np.nditer(tau, op_flags=['readwrite']):
    x[...] = x / time_interval

# 3. Form matrix equation of 4-3-4 trajectory and joint variables
K = np.matrix([[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
               [-4, -3, -2, -1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
               [-12, -6, -2, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
               [0, 0, 0, 0, 0, -3, -2, -1, 0, 0, 0, 0, 1, 0],
               [0, 0, 0, 0, 0, -6, -2, 0, 0, 0, 0, 2, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 3, 2, 1, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 12, 6, 2, 0, 0]])
K_inv = np.linalg.inv(K)

Q_1 = np.transpose(
    np.matrix([q_a[0], 0, 0, q_b[0], q_b[0], 0, 0, q_c[0], q_c[0], 0, 0, q_d[0], 0, 0]))
Q_2 = np.transpose(
    np.matrix([q_a[1], 0, 0, q_b[1], q_b[1], 0, 0, q_c[1], q_c[1], 0, 0, q_d[1], 0, 0]))
Q_3 = np.transpose(
    np.matrix([q_a[2], 0, 0, q_b[2], q_b[2], 0, 0, q_c[2], q_c[2], 0, 0, q_d[2], 0, 0]))

# 4. Solve matrix equation of 4-3-4 trajectory and joint variables
M1 = K_inv * Q_1
M2 = K_inv * Q_2
M3 = K_inv * Q_3

# 5. Calculate joint variables (angles of rotation)
time_length = tau.size
number_of_movements = 3
q_1 = np.zeros(time_length * number_of_movements)
q_2 = np.zeros(time_length * number_of_movements)
q_3 = np.zeros(time_length * number_of_movements)


# this function calculate vector of joint variables q from 4-3-4 trajectory polynomials
def calculate_join_var_from_trajectory_434(q, i, M, tau, k):
    q[i] = M[0] * pow(tau[i], 4) + M[1] * pow(tau[i], 3) + M[2] * pow(tau[i], 2) + M[3] * tau[i] + M[4]
    q[i + k] = M[5] * pow(tau[i], 3) + M[6] * pow(tau[i], 2) + M[7] * tau[i] + M[8]
    q[i + k * 2] = M[9] * pow(tau[i], 4) + M[10] * pow(tau[i], 3) + M[11] * pow(tau[i], 2) + M[12] * tau[i] + M[13]


for i in range(0, time_length):
    calculate_join_var_from_trajectory_434(q_1, i, M1, tau, time_length)
    calculate_join_var_from_trajectory_434(q_2, i, M2, tau, time_length)
    calculate_join_var_from_trajectory_434(q_3, i, M3, tau, time_length)

# 6. Solve forward problem of kinematics
coordinate_x = np.zeros(time_length * number_of_movements)
coordinate_y = np.zeros(time_length * number_of_movements)
coordinate_z = np.zeros(time_length * number_of_movements)


def calculate_matrix_of_homogeneous_transformations(thetha, a, alpha, d):
    A = np.matrix(
        [[np.cos(thetha), -np.sin(thetha), 0, 0], [np.sin(thetha), np.cos(thetha), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) \
        * np.matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, d], [0, 0, 0, 1]]) \
        * np.matrix([[1, 0, 0, a], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) \
        * np.matrix(
        [[1, 0, 0, 0], [0, np.cos(alpha), -np.sin(alpha), 0], [0, np.sin(alpha), np.cos(alpha), 0], [0, 0, 0, 1]])
    return A


# Denavit–Hartenberg parameters
a = [0, L[1], L[2]]
d = [L[0], 0, 0]
alpha = [np.pi / 2, 0, 0]
for i in range(0, time_length * 3):
    A_1 = calculate_matrix_of_homogeneous_transformations(q_1[i], a[0], alpha[0], d[0])
    A_2 = calculate_matrix_of_homogeneous_transformations(q_2[i], a[1], alpha[1], d[1])
    A_3 = calculate_matrix_of_homogeneous_transformations(q_3[i], a[2], alpha[2], d[2])
    T = A_1 * A_2 * A_3
    coordinate_x[i] = T.item((0, 3))
    coordinate_y[i] = T.item((1, 3))
    coordinate_z[i] = T.item((2, 3))

# Draw results
drwlines.draw_robot_way_3d_with_points(coordinate_x, coordinate_y, coordinate_z, [r_a, r_b, r_c, r_d])
xoy_point_list = [r_a[0:2], r_b[0:2], r_c[0:2], r_d[0:2]]
drwlines.draw_robot_way_2d_with_points(coordinate_x, coordinate_y, ["x,m", "y,m"], "XOY", xoy_point_list)
xoz_point_list = [[r_a[0], r_a[2]], [r_b[0], r_b[2]], [r_c[0], r_c[2]], [r_d[0], r_d[2]]]
drwlines.draw_robot_way_2d_with_points(coordinate_x, coordinate_z, ["x,m", "z,m"], "XOZ", xoz_point_list)
yoz_point_list = [r_a[1:3], r_b[1:3], r_c[1:3], r_d[1:3]]
drwlines.draw_robot_way_2d_with_points(coordinate_y, coordinate_z, ["y,m", "z,m"], "YOZ", yoz_point_list)
