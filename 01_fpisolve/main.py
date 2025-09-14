import numpy as np

# Values at the diagonal must be large
# Does not converge otherwise
A = np.array([
	[0.60, 0.02, 0.99],
	[0.99, 0.01, 0.07],
	[0.06, 0.92, 0.03]
])

B = np.array([-0.98, 0.66, -.82])

ans = np.linalg.solve(A, B)
print("Reference answer", ans)

# Ensure values at the diagonal are large
def rearrange(A, B):
	for i in range(len(A)):
		idx = np.argmax(A[i:, i])

		# Swap rows
		A[[i, i+idx]] = A[[i+idx, i]]
		B[[i, i+idx]] = B[[i+idx, i]]

rearrange(A, B)

def iterate(A, B, iters=30, eps=0.0001):
	#
	# Solving for unknowns
	#
	# 2u + 1v = 20
	# 3u + 5v = 51
	#
	# u = (20 - 1v) / 2
	# v = (51 - 5v) / 3
	#

	n = len(A)
	diag = np.diag(A)

	B = B / diag
	A = A / diag[:, None] # Divide in dim 0
	A = A * (1 - np.eye(n))
	A = -A

	alpha = np.linalg.norm(A)
	print("Alpha:", alpha)

	#
	# Early stop threshold
	#
	stop = eps * (1 - alpha)/alpha

	# Initial point
	x = np.zeros(n)

	log = [x]

	for i in range(iters):
		x = A @ x + B
		log.append(x)
		print(x)

		dist = np.linalg.norm(log[-2] - log[-1])

		if dist <= stop:
			print("Complete")
			return

	print("Timeout")

iterate(A, B)
