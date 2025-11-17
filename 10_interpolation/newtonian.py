import numpy as np

x = np.array([1.0, 2.0, 3.0])
y = x**2

#
# Conceptually, it is possible
# to add elements as needed
#

def divided_difference(x, y):
	lst = []
	z = np.copy(y)

	# From next subtract current
	for i, _ in enumerate(z):
		if i:
			z /= x[i] - x[0]

		lst.append(np.copy(z))
		z[:-1] = z[1:] - z[:-1]


	# Ignore lower right triangle
	return np.vstack(lst).T

def newtonian_fwd(x, y, v):
	dd = divided_difference(x, y)
	x_ = [1.0]

	for u in x[:-1]:
		x_.append( x_[-1] * (v - u) )

	return np.dot(dd[0], x_)

def newtonian_bwd(x, y, v):
	x = np.array(list(reversed(x)))
	y = np.array(list(reversed(y)))

	dd = divided_difference(x, y)
	x_ = [1.0]

	for u in x[:-1]:
		x_.append( x_[-1] * (v - u) )

	return np.dot(dd[0], x_)


def test():
	x = np.array([1.0, 2.0, 3.0])
	y = x**2

	print("Ground truth:", 1.1 ** 2)
	print("Newtonian  :", newtonian_fwd(x, y, 1.1))
