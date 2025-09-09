import numpy as np

#
# Start small
# A function: f(a, b) = 2a + 9b
#

# Values at the diagonal must be large
# Does not converge otherwise
A = np.array([
	[0.91, 1.04, 0.19],
	[0.04, -.99, -.85],
	[0.10, -.07, -.96]
])

B = np.array([-1.67, -3.73, -2.04])

ans = np.linalg.solve(A, B)
print("Reference answer", ans)

#
# 2u + 1v = 20
# 3u + 5v = 51
#

# u = (20 - 1v) / 2
# v = (51 - 5v) / 3
n = len(A)
diag = np.diag(A)

B = B / diag
A = A / diag[:, None] # Divide in dim 0
A = A * (1 - np.eye(n))
A = -A

# Initial point
x = np.zeros(n)

for i in range(30):
	x = A @ x + B
	print(x)
