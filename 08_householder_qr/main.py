import numpy as np

np.set_printoptions(suppress=True)

A = np.array([
	[0.60, 0.02, 0.99],
	[0.99, 0.01, 0.07],
	[0.06, 0.92, 0.03]
])

#
# https://en.wikipedia.org/wiki/QR_decomposition#Using_Householder_reflections
#

QT = np.eye(3)

for i in range(3):
	x = A.T[i].copy() # Be careful not to ruin A
	x[:i] = 0

	a = np.linalg.norm(x)

	if x[i] > 0:
		a = -a

	e = np.eye(3)[i]
	u = x - a*e
	v = u / np.linalg.norm(u)

	vvT = np.outer(v, v)
	H = np.eye(3) - 2*vvT

	#print(H @ x) # Should be one element
	#print(H @ A.T[i]) # Should be i+1 elements

	A = H@A
	QT = H@QT

Q = QT.T
R = A
