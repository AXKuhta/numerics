from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

x = np.array([
	1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007
])

y = np.array([
	192, 176, 177, 190, 185, 175, 178, 183, 186, 189, 193, 177, 180, 191, 192, 197, 199
])

x = np.array([1, 2])
y = np.array([3, 4])

indices = np.arange(len(x))

#
# Reference linear regression
#
model = LinearRegression()
model.fit(x[:, None], y)

prediction = model.predict(x[:, None])

print("R^2:", model.score(x[:, None], y))

#plt.plot(x, y)
#plt.plot(x, prediction)
#plt.show()


#
# SGD Linear regression
#

#
# Model:
# ax.i + b
#
# Loss:
# ( (ax.i + b) - y.i )^2
#
# dL/da = 2( ax.i + b - y.i )( x.i + 0 - 0 )
# dL/db = 2( ax.i + b - y.i )( 0 + 1 - 0 )
#

dl_da = lambda a, b, x, y: 2*(a*x + b - y)*x
dl_db = lambda a, b, x, y: 2*(a*x + b - y)

# Initial approximation
loc = np.array([1.0, 1.0])
alpha = 0.1

history = [loc.copy()]

for i in range(1000):
	grad_acc = np.zeros(2) # Gradient accumulator

	a, b = loc

	for i in np.random.choice(indices, size=2, replace=False):
		grad_a = dl_da(a, b, x[i], y[i])
		grad_b = dl_db(a, b, x[i], y[i])

		print(a, b, x[i], y[i], grad_a)

		grad_acc += np.array([grad_a, grad_b])

	loc -= alpha*grad_acc

	history.append(loc.copy())

	dist = np.hypot( *(history[-2] - history[-1]) )

	if dist < 0.001:
		break
