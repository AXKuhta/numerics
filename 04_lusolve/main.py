import numpy as np

np.set_printoptions(suppress=True)

alpha = np.pi * 17/18

# Rotation matrix
A = np.array([
	[+np.cos(alpha), -np.sin(alpha)],
	[+np.sin(alpha), +np.cos(alpha)]
])

x_ground_truth = np.array([0.2, .99])
B = A @ x_ground_truth

ans = np.linalg.solve(A, B)
print("Reference answer", ans)

#
# Gaussian elimination
#
def gaussian_elim(A):
	n = len(A)
	invA = np.eye(n)

	for i in range(n):
		transform = np.eye(n)
		transform.T[i] = -A.T[i] / A[i, i]
		transform[i, i] = 1/A[i, i]

		A = transform @ A
		invA = transform @ invA

	return A, invA

def find_transformations(A):
	history = []
	n = len(A)

	for i in range(n):
		transform = np.eye(n)
		transform.T[i, i+1:] = -A.T[i, i+1:] / A[i, i]

		A = transform @ A

		history.append(transform)

	# Forward E, inverse E
	fwdE = np.eye(n)
	invE = np.eye(n)

	for step in history:
		fwdE = step @ fwdE

	for step in reversed(history):
		invE = (-step + 2*np.eye(n)) @ invE

	return fwdE, invE

#
# LU
#
E, iE = find_transformations(A)

U = E@A
L = iE
iL = E

#
# LDU
#
D = np.diag(U)
U = U/D[:, None]

# Undo
U = np.diag(D) @ U

# Back substitution
# https://en.wikipedia.org/wiki/Triangular_matrix
x = np.zeros(2)
iLB = iL @ B

for i in reversed(range(len(U))):
	x[i] = (iLB - U@x)[i]/U[i, i]

print("Answer", x)
