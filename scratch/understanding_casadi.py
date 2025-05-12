import casadi

x = casadi.MX.sym("x")
y = casadi.MX.sym("y", 5)
z = casadi.MX.sym("z", 4, 2)

f = 3*z + x
print(f)
f = casadi.sqrt(f)
print(f)
print(f.shape)