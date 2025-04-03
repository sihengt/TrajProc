import numpy as np

# KBM is weird when we use the CG.
# We get multiple different calculations that give us an expression for sideslip.
# I want to explore if all roads lead to the same answer.

# Road 1: Beta = arcsin(l_R/R)

# wheelbase
L = 2.4
l_r = 0.8
l_f = 1.6

# velocity at cg
v_cg = 0.8

# delta
delta = np.pi / 4

beta_3 = np.arctan((l_r * np.tan(delta)) / L)

R = L/(np.tan(delta) * np.cos(beta_3))
beta_1 = np.arcsin(l_r / R)
beta_2 = np.arccos(L / R * np.tan(delta))

print("Beta 3 = {}".format(beta_3))
print("Beta 1 = {}".format(beta_1))
print("Beta 2 = {}".format(beta_2))
