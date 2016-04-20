import numpy as np
import scipy.signal as sp_signal
import matplotlib.pyplot as plt
import scipy.linalg as sp_linalg

# parameters of Furuta Pendulum from report:
# Chye T.K. and Sang T.C. Rotary Inverted Pendulum
L_0 = 0.1370  # length of the platform in meter
l_1 = 0.1572  # length of the pendulum in meter
m_1 = 0.0319  # mass of the pendulum in kg
J_0 = 0.008591  # inertia of the platform in kg*meter_sq
J_1 = 0.000217  # inertia of the pendulum in kg*meter_sq
C_0 = 0.006408  # friction coefficient of platform in kg*meter_sq/sec
C_1 = 0.000158  # friction coefficient of pendulum in kg*meter_sq/sec
g = 9.8
# coefficients of electric motor
R_a = 0.9
K_t = 0.0706
K_b = 0.0707
K_u = 0.0636

# auxiliary quantities
a = J_0 + m_1 * L_0 ** 2
c = m_1 * L_0 * l_1
d = C_0 + (K_t * K_b) / R_a
e = (K_t * K_u) / R_a
f = J_1 + m_1 * l_1 ** 2
h = m_1 * g * l_1

# Furuta pendulum linear model in upper equilibrium point MIMO representation
A = 1 / (a * f - c ** 2) * np.matrix([[-d * f, c * h, -c * C_1], [0, 0, a * f - c ** 2], [-c * d, a * h, -a * C_1]])
B = 1 / (a * f - c ** 2) * np.matrix([[e * f], [0], [c * e]])
C = np.matrix([[1, 0, 0], [0, 1, 0]])
# pendulum state space model
pendulum_model = sp_signal.StateSpace(A, B, C)

# initial conditions for 5 deg deviation of pendulum
x_0 = [0, 5 * np.pi / 180, 0]
# time
t = np.linspace(0, 5)
# no control
u = np.zeros_like(t)
# simulation
tout, y, x = sp_signal.lsim(pendulum_model, u, t, x_0)
w_alpha_unstable = y[:, 0]
betta_unstable = y[:, 1]


# for pendulum stabilization used linear-quadratic regulator (LQR)
# build closed loop system with LQR regulator
# dx/dt = Ax + Bu => dx/dt = (A-K*B)x + Bu

# this function based on project https://github.com/markwmuller/controlpy
def lqr(A, B, Q, R):
    """
    Solve the continuous time LQR controller for a continuous time system.

    A and B are system matrices, describing the systems dynamics:
     dx/dt = A x + B u

    The controller minimizes the infinite horizon quadratic cost function:
     cost = integral (x.T*Q*x + u.T*R*u) dt

    where Q is a positive semidefinite matrix, and R is positive definite matrix.

    Returns K, X, eigVals:
    Returns gain the optimal gain K, the solution matrix X, and the closed loop system eigenvalues.
    The optimal input is then computed as:
     input: u = -K*x
    """
    # ref Bertsekas, p.151

    # first, try to solve the ricatti equation
    X = np.matrix(sp_linalg.solve_continuous_are(A, B, Q, R))
    # compute the LQR gain
    K = np.matrix(sp_linalg.inv(R) * (B.T * X))
    eigVals, eigVecs = sp_linalg.eig(A - B * K)
    return K, X, eigVals


# optimal control weight matrix
Q = np.diag([1, 1, 1])
R = np.matrix([0.1])
# calculate regulator coefficients matrix
K, X, eig = lqr(A, B, Q, R)
# pendulum with regulator state space model
pendulum_model_with_regulator = sp_signal.StateSpace(A - B * K, B, C)
# simulation
tout, y, x = sp_signal.lsim(pendulum_model_with_regulator, u, t, x_0)
w_alpha_stable = y[:, 0]
betta_stable = y[:, 1]

# plot unstable and stable system output
plt.subplot(2, 2, 1)
plt.plot(t, w_alpha_unstable, 'k')
plt.title('Unstable w_alpha')
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(t, betta_unstable, 'k')
plt.title('Unstable betta')
plt.grid(True)

plt.subplot(2, 2, 3)
plt.plot(t, w_alpha_stable, 'r')
plt.title('Stable w_alpha')
plt.grid(True)

plt.subplot(2, 2, 4)
plt.plot(t, betta_stable, 'r')
plt.title('Stable betta')
plt.grid(True)
plt.show()
