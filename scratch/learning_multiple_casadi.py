from casadi import *
import matplotlib.pyplot as plt

# Experiment parameters
T = 10.     # Time Horizon
N = 20      # Control intervals
M = 4       # RK4 steps per interval

X = MX.sym("X", 2)
u = MX.sym("u")

# Dynamics
x_dot_0 = (1 - X[1]**2)*X[0] - X[1] + u
x_dot_1 = X[0]
X_dot = vertcat(x_dot_0, x_dot_1)

# Cost
L = X[0]**2 + X[1]**2 + u**2

# Getting symbolic function for integrating dynamics
f = Function('f', [X, u], [X_dot, L])
DT = T/N/M
x0_int = MX.sym("x0_int", 2)
u_int = MX.sym("u_int")
x_accumulated = x0_int
q_accumulated = 0

# RK4
for j in range(M):
    k1, k1_q = f(x_accumulated, u_int)
    k2, k2_q = f(x_accumulated + DT/2 * k1, u_int)
    k3, k3_q = f(x_accumulated + DT/2 * k2, u_int)
    k4, k4_q = f(x_accumulated + DT * k3, u_int)
    x_accumulated += DT/6 * (k1 + 2*k2 + 2*k3 + k4)
    q_accumulated += DT/6 * (k1_q + 2*k2_q + 2*k3_q + k4_q)
F = Function('F', [x0_int, u_int], [x_accumulated, q_accumulated], ['x0', 'p'], ['xf', 'qf'])

#####################
## BUILDING THE NLP #
#####################

# 1. Decision variables (all states except first and controls)
# 1a. LBW and UBW for the decision variables
# 1b. Starting guess for the decision variables
# 2. States at time step k are not constraints - they are just states.
# 3. Maintain the cost over time
# 4. Add inequality constraints for whatever needs constraints

w = [] # Decision variables
g = [] # Constraints

w0 = [] # Initial guesses
lbw = [] # Lower bound for decision variables
ubw = [] # Upper bound for decision variables
lbg = [] # Lower bound for constraint
ubg = [] # Upper bound for constraint
J = 0

# How we encode initial conditions
Xk = MX.sym('X0', 2)
w += [Xk]
lbw += [0, 1]
ubw += [0, 1]
w0 += [0, 1]

for k in range(N):
    Uk = MX.sym("U{}".format(k))
    w.append(Uk)
    lbw.append(-1.0)
    ubw.append(1.0)
    w0.append(0)

    Fk = F(x0=Xk, p = Uk)
    Xkp1 = Fk['xf']
    J += Fk['qf']

    Xk = MX.sym('X_' + str(k+1), 2)
    w += [Xk]
    lbw += [-0.25, -inf]
    ubw += [inf, inf]
    w0 += [0, 0]

    g += [Xkp1 - Xk]
    lbg += [0, 0]
    ubg += [0, 0]

prob = {'x': vertcat(*w), 'f': J, 'g': vertcat(*g)}
solver = nlpsol('solver', 'ipopt', prob)
sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)

w_opt = sol['x'].full().flatten()
x1_opt = w_opt[0::3]
x2_opt = w_opt[1::3]
u_opt = w_opt[2::3]

tgrid = [T/N*k for k in range(N+1)]
plt.figure(1)
plt.clf()
plt.plot(tgrid, x1_opt, '--')
plt.plot(tgrid, x2_opt, '-')
plt.step(tgrid, vertcat(DM.nan(1), u_opt), '-.')
plt.xlabel('t')
plt.legend(['x1', 'x2', 'u'])
plt.grid()
plt.show()
