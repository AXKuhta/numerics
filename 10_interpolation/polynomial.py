import numpy as np

# Returns a function
def polyfit(x, y):
	dataset = np.vstack([x**i for i in range(len(x))]).T
	return np.poly1d( np.linalg.solve(dataset, y)[::-1] )

def test():
	x = np.array([1.0, 2.0, 3.0])
	y = x**2

	fn = polyfit(x, y)

	print("Ground truth:", 1.1 ** 2)
	print("Polynomial  :", fn(1.1))

# Third order polynomial spline
def spline2(x, y):
	dataset_x = np.vstack([x**i for i in range(4)]).T
	dataset_x_a = dataset_x[:3]
	dataset_x_b = dataset_x[2:]
	dataset_y_a = y[:3]
	dataset_y_b = y[2:]

	# Add derivative condition
	extra_a = [i*x[2]**(i-1) for i in range(4)]
	extra_b = [i*x[2]**(i-1) for i in range(4)]

	dataset_x_a = np.vstack([dataset_x_a, extra_a])
	dataset_x_b = np.vstack([dataset_x_b, extra_b])
	dataset_y_a = np.hstack([dataset_y_a, 0.])
	dataset_y_b = np.hstack([dataset_y_b, 0.])

	coefs_a = np.linalg.solve(dataset_x_a, dataset_y_a)[::-1]
	coefs_b = np.linalg.solve(dataset_x_b, dataset_y_b)[::-1]

	return np.poly1d(coefs_a), np.poly1d(coefs_b)

def test_spline2():
	a, b = spline2(np.array([1,2,3,4,5.]), np.array([1,2,3,4,5.]))
	x_a = np.linspace(1, 3, 15)
	x_b = np.linspace(3, 5, 15)

	import matplotlib.pyplot as plt

	plt.plot(x_a, a(x_a))
	plt.plot(x_b, b(x_b))
	plt.show()
