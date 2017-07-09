import matplotlib.pyplot as plt
import numpy as np
import sympy as sym
from scipy.integrate import odeint


def nonlinear(state, t):
    # unpack the state vector
    x1 = state[0]
    x2 = state[1]

    # compute state derivatives
    x1_d = -x1 + x2 + 0.5 * x1 * x2 + (x2 ** 2)
    x2_d = -0.5 * x1 - 0.5 * (x1 ** 2) - x1 * x2

    # return the state derivatives
    return [x1_d, x2_d]


def find_equilibrium_positions():
    x = sym.Symbol('x')
    y = sym.Symbol('y')
    return sym.solve_poly_system([-x + y + 0.5 * x * y + (y ** 2), -0.5 * x - 0.5 * (x ** 2) - x * y], x, y)


def generate_initial_states(start, stop, step):
    states = []
    for i in np.arange(start, stop, step):
        states.append([i, i])
        states.append([i, -i])
    return states


def build_jackobian(position):
    x = sym.Symbol('x')
    y = sym.Symbol('y')
    F1 = -x + y + 0.5 * x * y + (y ** 2)
    F2 = -0.5 * x - 0.5 * (x ** 2) - x * y
    F1_dx = sym.diff(F1, x)
    F1_dy = sym.diff(F1, y)
    F2_dx = sym.diff(F2, x)
    F2_dy = sym.diff(F2, y)
    replacement = [(x, position[0]), (y, position[1])]
    return np.matrix([[F1_dx.subs(replacement), F1_dy.subs(replacement)],
               [F2_dx.subs(replacement), F2_dy.subs(replacement)]])

def define_point_type(point):
    jacobian = build_jackobian(point)
    #TODO solve problem with sympy to numpy
    print(jacobian)


def define_separatris():
    # TODO sutable API: (w, point) return slope, const
    A = np.matrix([[-1.2, 0.1], [0.09, 0.2]])
    v, w = np.linalg.eig(A)
    eig_vect_1 = w[:,1]
    point = [-0.2, -0.4]
    p1 = eig_vect_1[0]
    p2 = eig_vect_1[1]
    slope = p2 / p1
    constant_term = point[1] - slope * point[0]
    print("slope=", slope)
    print("constant_term=", constant_term)


t = np.linspace(0.0, 20, 1000)
initial_states = generate_initial_states(-1.0, 1.0, 0.2)
outputs = []
for state in initial_states:
    outputs.append(odeint(nonlinear, state, t))

equilibrium_positions = find_equilibrium_positions()

fig = plt.figure()
for output in outputs:
    plt.plot(output[:, 0], output[:, 1], 'r-')

for point in equilibrium_positions:
    plt.plot(point[0], point[1], 'o')

values = np.arange(-0.5, 0.1, 0.01)
#TODO automaticaly calculate params!
separatrice_list = [[-0.06, -0.41], [14.06, 2.41]]
for separatrice in separatrice_list:
    f = lambda x: x*separatrice[0] + separatrice[1]
    plt.plot(values, f(values), "g-")

plt.xlabel('x_1')
plt.ylabel('x_2')
plt.title('phase portrait')
plt.grid(True)
plt.show()

