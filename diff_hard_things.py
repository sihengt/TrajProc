import numpy as np
import sympy as sp

v, delta, L, l_r = sp.symbols('v delta L l_r')
theta_dot = (v * sp.tan(delta) * sp.cos(sp.atan(l_r * sp.tan(delta) / L))) / L

# Differentiate with respect to desired variables
d_theta_dot_dv = sp.diff(theta_dot, v)
d_theta_dot_ddelta = sp.diff(theta_dot, delta)

# Optional: Convert to a numerical function
f = sp.lambdify((v, delta, L, l_r), theta_dot, 'numpy')
df_ddelta = sp.lambdify((v, delta, L, l_r), d_theta_dot_ddelta, 'numpy')

breakpoint()