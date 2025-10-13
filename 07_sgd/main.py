from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import StandardScaler

x = np.array([
	1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007
])

y = np.array([
	192, 176, 177, 190, 185, 175, 178, 183, 186, 189, 193, 177, 180, 191, 192, 197, 199
])

scaler = StandardScaler()
x = scaler.fit_transform(x[:, None])
x = x[:, 0]

indices = np.arange(len(x))

#
# Reference linear regression
#
model = LinearRegression()
model.fit(x[:, None], y)

prediction = model.predict(x[:, None])

x_ = scaler.inverse_transform(x[:, None])[:, 0]

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
# Regularization:
# ( (ax.i + b) - y.i )^2 + tau*l2(w)^2
# ( (ax.i + b) - y.i )^2 + tau*(a^2 + b^2)
#
# dL/da = 2( ax.i + b - y.i )( x.i + 0 - 0 ) + tau*2a
# dL/db = 2( ax.i + b - y.i )( 0 + 1 - 0 ) + tau*2b
#

tau = 0 #.1

dl_da = lambda a, b, x, y: 2*(a*x + b - y)*x + tau*2*a
dl_db = lambda a, b, x, y: 2*(a*x + b - y) + tau*2*b

# Initial approximation
loc = np.array([1.0, 1.0])
alpha = 0.01

history = [loc.copy()]

for i in range(1000):
	grad_acc = np.zeros(2) # Gradient accumulator

	a, b = loc

	for j in np.random.choice(indices, size=4, replace=False):
		grad_a = dl_da(a, b, x[j], y[j])
		grad_b = dl_db(a, b, x[j], y[j])

		grad_acc += np.array([grad_a, grad_b])

	loc -= alpha*grad_acc

	history.append(loc.copy())

	dist = np.hypot( *(history[-2] - history[-1]) )

	if dist < 0.001:
		break

	print(i+1, "/ 1000 iterations")

y_pred_sgd = a*x + b
residuals = y - y_pred_sgd

print("R^2 sklearn:", model.score(x[:, None], y))
print("R^2 SGD:", np.var(y_pred_sgd)/np.var(y))
print("MSE SGD:", np.mean(np.dot(residuals, residuals)))
print("MAD SGD:", np.mean(np.sum(np.abs(residuals))))
print("MAPE SGD:", np.mean(np.abs(residuals)/y))

plt.plot(x_, y, label="ground truth")
plt.plot(x_, prediction, label="sklearn")
plt.plot(x_, y_pred_sgd, label="sgd")
plt.legend()
plt.show()
