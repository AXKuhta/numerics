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

