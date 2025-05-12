import casadi as cs

# Your dynamics model
# x[0], x[1] are unused

L_F = 0.8
L_R = 1.0
T = 10.
N = 20.
M = 4

###########################
## CasADi initializations #
###########################

# Quadratic cost matrices
Q   = cs.diag(cs.DM([20, 20, 10, 0]))   # state error cost
Qf  = cs.diag(cs.DM([30, 30, 30, 0]))   # state  final error cost
R   = cs.diag(cs.DM([10, 10, 10]))      # input cost
R_  = cs.diag(cs.DM([10, 10, 10]))      # input rate of change cost

# Defining symbols for state
x = cs.SX.sym('x')
y = cs.SX.sym('y')
v = cs.SX.sym('v')
theta = cs.SX.sym('theta')
X = cs.vertcat(x, y, v, theta)

# Defining symbols for reference state
x_ref = cs.SX.sym('x_ref')
y_ref = cs.SX.sym('y_ref')
v_ref = cs.SX.sym('v_ref')
theta_ref = cs.SX.sym('theta_ref')
X_ref = cs.vertcat(x_ref, y_ref, v_ref, theta_ref)

# Defining symbols for controls
a = cs.SX.sym('a')
delta_f = cs.SX.sym('df')
delta_r = cs.SX.sym('dr')
U = cs.vertcat(a, delta_f, delta_r)

######################
# Getting kinematics #
######################

get_sideslip = cs.Function(
    'get_sideslip',
    [delta_f, delta_r],
    [cs.arctan( (L_R * cs.tan(delta_f) + L_F * cs.tan(delta_r)) / (L_R + L_F))],
    ['delta_f', 'delta_r'],
    ['beta']
)

beta = get_sideslip(delta_f=delta_f, delta_r=delta_r)

x_dot = v * cs.cos(beta['beta'] + theta)
y_dot = v * cs.sin(beta['beta'] + theta)
v_dot = a
theta_dot = (v * cs.cos(beta['beta'])) / (L_F + L_R) * (cs.tan(delta_f) - cs.tan(delta_r))
X_dot = cs.vertcat(x_dot, y_dot, v_dot, theta_dot)

breakpoint()

# Right now our loss is calculated as difference between the current trajectory and the reference trajectory.
# If we want to do this in the dynamics model, we'll have to put the reference trajectory in as well.
# We need to find the closest point 
e = X - X_ref
L = (e.T @ Q @ e) + (U.T @ R @ U)

# f takes X and U, and returns the time derivative of the state.
f = cs.Function('f', [X, U, X_ref], [X_dot, L])

x0_int  = cs.MX.sym('x0_int', 4)
u_int   = cs.MX.sym('u_int', 3)
x_accumulated = x0_int
q_accumulated = 0

DT = T/N/M
for j in range(M):
    k1, k1_q = f(x_accumulated, u_int, X_ref)
    k2, k2_q = f(x_accumulated + DT/2 * k1, u_int, X_ref)
    k3, k3_q = f(x_accumulated + DT/2 * k2, u_int, X_ref)
    k4, k4_q = f(x_accumulated + DT * k3, u_int, X_ref)
    x_accumulated += DT/6 * (k1 + 2*k2 + 2*k3 + k4)
    q_accumulated += DT/6 * (k1_q + 2*k2_q + 2*k3_q + k4_q)
F = cs.Function('F', [x0_int, u_int, X_ref], [x_accumulated, q_accumulated], ['x0', 'p', 'xref'], ['xf', 'qf'])

Fk = F(x0=[0.2, 0.2, 1.0, cs.pi/2], p=[0.1, cs.pi/10, cs.pi/10], xref=[0.3, 0.3, 2.0, cs.pi/3])


