import numpy as np

np.set_printoptions(suppress=True)

A = np.array([
	[ 3, 1, 1],
	[-1, 3, 1]
]) + 0.0

x_ground_truth = np.array([0.2, .99, .01])
B = A @ x_ground_truth

# No exact solution
# Use least squares
ans, residuals, rank, s = np.linalg.lstsq(A, B)
print("Least squares answer", ans)

# A = U @ S @ V
#
# U	orthonormal
# S	diagonal
# V	orthonormal
#
# Property:
# 	U @ U.T = I
#

U, S, V = np.linalg.svd(A)

#
# U.T @ A = S @ V
#
# (np.diag(1/S) @ U.T @ A) = V
#

ei, eiv = np.linalg.eig(A@A.T)

ind = np.flip(np.argsort(ei))

U_ = eiv[ind].T
S_ = ei[ind]**0.5
V_ = (np.diag(1/S_) @ U_.T @ A)

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


# Orthogonalization
# https://en.wikipedia.org/wiki/QR_decomposition
def qr(A):
	Q = A.copy()
	k_ = len(A.T)

	def project(a, b):
		if len(b) == 0:
			return np.zeros(k_)

		k = np.dot(b, a) / np.sum(b*b, 1)

		return np.sum(k[:, None]*b, 0)

	for k in range(k_):
		a = Q.T[k]
		b = Q.T[:k]
		a -= project(a, b)
		a /= np.linalg.norm(a)

	R = Q.T @ A

	return Q, R

# QR eigenvector algorithm
# https://en.wikipedia.org/wiki/QR_algorithm
#
# See also:
# https://en.wikipedia.org/wiki/Eigendecomposition_of_a_matrix
# https://en.wikipedia.org/wiki/Eigenvalue_algorithm
# https://en.wikipedia.org/wiki/Power_iteration
#
# Q_	Eigenvectors
# LAM	Eigenvalues
#
# A = Q_ @ LAM @ inv(Q_)
#
LAM = A@A.T
Q_ = np.eye(len(A))

for i in range(5000):
	Q, R = qr(LAM)
	LAM = Q.T @ LAM @ Q
	Q_ = Q @ Q_

assert np.sum(np.abs(np.tril(LAM, -1))) < 0.001, "Failed to converge"

# Sort eigenvalues descending
ind = np.flip(np.argsort(np.diag(LAM)))

U__ = Q_[ind].T
S__ = np.diag(LAM)[ind]**0.5
V__ = (np.diag(1/S__) @ U__.T @ A)

# Pseudosolution
y = U__.T @ B
z = y @ np.diag(1/S__)
x = V__.T @ z

print("SVD answer", x)

B_ = A @ x

residuals = np.abs(B - B_)

print("Residuals", residuals)
