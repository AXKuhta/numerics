import numpy as np

# Values at the diagonal must be large
# Does not converge otherwise
A = np.array([
	[0.60, 0.02, 0.99],
	[0.99, 0.01, 0.07],
	[0.06, 0.92, 0.03]
])

B = np.array([-0.98, 0.66, -.82])

ans = np.linalg.solve(A, B)
print("Reference answer", ans)

def rearrange(A):
	for i in range(len(A)):
		idx = np.argmax(A[i:, i])
		A[i], A[i+idx] = A[i+idx].copy(), A[i].copy()
		B[i], B[i+idx] = B[i+idx].copy(), B[i].copy()

rearrange(A)

#
# Solving for unknowns
#
# 2u + 1v = 20
# 3u + 5v = 51
#
# u = (20 - 1v) / 2
# v = (51 - 5v) / 3
#

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
