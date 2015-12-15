import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

mpl.rcParams['legend.fontsize'] = 10


def draw_robot_way_3d(x, y, z):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(x, y, z, label='way of manipulator')
    ax.legend()
    plt.show()


def draw_robot_way_3d_with_points(x, y, z, point3d_list):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(x, y, z, label='way of manipulator')
    for point in point3d_list:
        ax.scatter(point[0], point[1], point[2])
    ax.legend()
    ax.set_xlabel('x,m')
    ax.set_ylabel('y,m')
    ax.set_zlabel('z,m')
    plt.show()


def draw_robot_way_2d(data_1, data_2, label_names, plate_name):
    plt.plot(data_1, data_2)
    plt.xlabel(label_names[0])
    plt.ylabel(label_names[1])
    plt.title(plate_name)
    plt.grid(True)
    plt.show()


def draw_robot_way_2d_with_points(data_1, data_2, label_names, plate_name, point2d_list):
    plt.plot(data_1, data_2, )
    plt.xlabel(label_names[0])
    plt.ylabel(label_names[1])
    for point in point2d_list:
        plt.plot(point[0], point[1], 'o')
    plt.title(plate_name)
    plt.grid(True)
    plt.show()
