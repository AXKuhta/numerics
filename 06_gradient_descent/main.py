from sympy import *
import numpy as np

init_printing(use_unicode=False)

# Automatic differentiation
f = sympify("exp(-x^2 - y^2)")

df_dx = f.diff("x")
df_dy = f.diff("y")

# Initial approximation
loc = np.array([1.0, 1.0])
alpha = 0.1

history = [loc.copy()]

for i in range(1000):
	grad_x = float( df_dx.subs({"x": loc[0], "y": loc[1]}) )
	grad_y = float( df_dy.subs({"x": loc[0], "y": loc[1]}) )

	grad = np.array([grad_x, grad_y])

	# This function has no minimum
	# So climb instead of descent
	loc += alpha*grad

	history.append(loc.copy())

	dist = np.hypot( *(history[-2] - history[-1]) )

	if dist < 0.001:
		break

# Desmos compatible point list
for i in range(len(history)):
	history[i] = tuple(history[i].tolist()) + (f.subs({"x": history[i][0], "y": history[i][1]}),)

print(history)
