# Program that model stabilization of Furuta pendulum in upper equilibrium point
## Description
This program models behavior of pendulum for 5 deg deviation from upper point.
Model of pendulum is nonlinear but we consider it near upper equilibrium where it is linear.
First behavior of pendulum is modeled for system without regulator, then for system with optimal LQR regulator.
## Information
This program is written on *Python 3.5*. It use **numpy** for matrix operations, **scipy** for state space modeling and linalg functions and **matplotlib** for visualisation of data. 
