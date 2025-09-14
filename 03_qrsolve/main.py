import numpy as np

def disp_vectors(Q):
	from mpl_toolkits.mplot3d import Axes3D
	import matplotlib.pyplot as plt

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	ax.set_xlim([-1, 1])
	ax.set_ylim([-1, 1])
	ax.set_zlim([-1, 1])

	ax.quiver(0,0,0, *Q[0], arrow_length_ratio=0.1)
	ax.quiver(0,0,0, *Q[1], arrow_length_ratio=0.1)
	ax.quiver(0,0,0, *Q[2], arrow_length_ratio=0.1)

	plt.show()

np.set_printoptions(suppress=True)

A = np.array([
	[0.60, 0.02, 0.99],
	[0.99, 0.01, 0.07],
	[0.06, 0.92, 0.03]
])

B = np.array([-0.98, 0.66, -.82])

ans = np.linalg.solve(A, B)
print("Reference answer", ans)

ans = np.linalg.inv(A) @ B
print("Reference answer", ans)

# A = Q@R
#
# Property:
# 	Q @ Q.T = E
#
# Hence:
#	Q.T = inv(Q)

#
# A @ x = B
#
# 	inv(A) @ A @ x = inv(A) @ B
#	E @ x = inv(A) @ B
#	x = inv(A) @ B
#
# Q @ R @ x = B
#
#	Q.T @ Q @ R @ x = Q.T @ B
#	R @ x = Q.T @ B
#
#	inv(R) @ R @ x = inv(R) @ Q.T @ B
#	x = inv(R) @ Q.T @ B
#

Q, R = np.linalg.qr(A)
ans = np.linalg.inv(R) @ Q.T @ B
print(ans)

# Orthogonalization
# https://en.wikipedia.org/wiki/QR_decomposition
Q = np.zeros_like(A)
R = np.zeros_like(A)
k_ = len(A.T)

def project(a, b):
	if len(b) == 0:
		return np.zeros_like(a)

	k = np.dot(b, a) / np.sum(b*b, 1)

	return np.sum(k[:, None]*b, 0)

for k in range(k_):
	Q.T[k] = A.T[k] - project(A.T[k], Q.T[:k])
	Q.T[k] /= np.linalg.norm(Q.T[k])
	R.T[k][:k+1] = Q.T[:k+1] @ A.T[k]

ans = np.linalg.inv(R) @ Q.T @ B
print(ans)
