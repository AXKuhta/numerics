import numpy as np

#
# l.0 = (x - x.0)/(x.0 - x.0) # skip this (i == j)
#	(x - x.1)/(x.0 - x.1)
#	(x - x.2)/(x.0 - x.2)
#

def lagrangian(x, y, v):
	matrix_n = v*np.ones_like(x)[:, None] - x
	matrix_d = x[:, None] - x

	ind = np.diag_indices_from(matrix_n)

	matrix_n[ind] = 1
	matrix_d[ind] = 1

	vec_n = np.prod(matrix_n, 1)
	vec_d = np.prod(matrix_d, 1)
	vec = vec_n/vec_d

	return np.dot(y, vec)

def test():
	x = np.array([1.0, 2.0, 3.0])
	y = x**2

	# Lagrangian can fit x**2 perfectly
	print("Ground truth:", 1.1 ** 2)
	print("Lagrangian  :", lagrangian(x, y, 1.1))
