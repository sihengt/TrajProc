from casadi import *
import matplotlib.pyplot as plt

# Experiment parameters
T = 10.     # Time horizon
N = 20      # Control intervals
M = 4       # RK4 steps per interval

X = MX.sym('X', 2)

u = MX.sym('u')

# Dynamics
x_dot_0 = (1 - X[1]**2)*X[0] - X[1] + u
x_dot_1 = X[0]
X_dot = vertcat(x_dot_0, x_dot_1)

L = X[0]**2 + X[1]**2 + u**2

# Getting symbolic function for integrating dynamics
f = Function('f', [X, u], [X_dot, L])
DT = T/N/M
x0_int = MX.sym('x0_int', 2)
u_int = MX.sym('u_int')
x_accumulated = x0_int
q_accumulated = 0

# Each interval has M RK4 calls. We perform RK4 M times here and accumulate it into the symbolic expression.
# Note that this RK4 function only works for the given time and control interval.
for j in range(M):
    k1, k1_q = f(x_accumulated, u_int)
    k2, k2_q = f(x_accumulated + DT/2 * k1, u_int)
    k3, k3_q = f(x_accumulated + DT/2 * k2, u_int)
    k4, k4_q = f(x_accumulated + DT * k3, u_int)
    x_accumulated += DT/6 * (k1 + 2*k2 + 2*k3 + k4)
    q_accumulated += DT/6 * (k1_q + 2*k2_q + 2*k3_q + k4_q)
F = Function('F', [x0_int, u_int], [x_accumulated, q_accumulated], ['x0', 'p'], ['xf', 'qf'])

# Evaluate at a test point
Fk = F(x0=[0.2,0.3],p=0.4)
print(Fk['xf'])
print(Fk['qf'])

#####################
## BUILDING THE NLP #
#####################

# 1. Decision variables (in this case just the control since it's single shooting)
# 1a. LBW and UBW for the decision variables
# 1b. Starting guess for the decision variables
# 2. States at time step k are not constraints - they are just states.
# 3. Maintain the cost over time
# 4. Add inequality constraints for whatever needs constraints

# Lists to keep track of variables.
w = []      # Decision variables
g = []      # Constraints

w0 = []     # Guess for decision variables
lbw = []
ubw = []
lbg = []
ubg = []
J = 0

# Initial point as defined by the problem set.
Xk = MX([0, 1])

for k in range(N):
    # Decision variables (in this case only the controls)
    Uk = MX.sym("U{}".format(k))
    w.append(Uk)
    lbw.append(-1.0)
    ubw.append(1.0)

    # Integrate
    Fk = F(x0=Xk, p=Uk)
    Xk = Fk['xf']
    J += Fk['qf']

    # Add inequality constraints
    g += [Xk[0]]
    lbg += [-0.3]
    ubg += [inf]

prob = {'f':J, 'x': vertcat(*w), 'g': vertcat(*g)}
solver = nlpsol('solver', 'ipopt', prob)
sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)

u_opt = sol['x']

# Plot the solution

x_opt = [[0, 1]]
for k in range(N):
    Fk = F(x0=x_opt[-1], p=u_opt[k])
    # Convert into np.ndarray and append to list.
    x_opt.append(Fk['xf'].full())

x1_opt = vcat([r[0] for r in x_opt])
x2_opt = vcat([r[1] for r in x_opt])

# T/N = how often we control the system.
# We create a grid for plotting
tgrid = [T/N*k for k in range(N+1)]

plt.figure(1)
plt.clf()
plt.plot(tgrid, x1_opt, '--')
plt.plot(tgrid, x2_opt, '-')
plt.step(tgrid, vertcat(DM.nan(1), u_opt), '-.')
plt.xlabel('t')
plt.legend(['x1','x2','u'])
plt.grid()
plt.show()