import numpy as np

#
# Start small
# A function: f(a, b) = 2a + 9b
#

# Values at the diagonal must be large
# Does not converge otherwise
A = np.array([
	[2, 1],
	[3, 5]
])

B = np.array([13, 51])

np.linalg.solve(A, B)

#
# 2u + 1v = 20
# 3u + 5v = 51
#

# u = (20 - 1v) / 2
# v = (51 - 5v) / 3
n = 2
diag = np.diag(A)

B = B / diag
A = A / diag[:, None] # Divide in dim 0
A = A * (1 - np.eye(n))
A = -A

# Initial point
x = np.array([0, 0])

for i in range(30):
	x = A @ x + B
	print(x)
