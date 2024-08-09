import numpy as np

# Sample input array
xs = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [109, 110, 111, 112],
    [209, 210, 211, 212],
])

N, _ = xs.shape

# Initialize the output array
output = np.zeros((N, 2, N, 4))

# Fill in channel 0 with the nth row of the input array repeated along the H dimension
for i in range(N):
    output[i, 0, :, :] = np.tile(xs[i], (N, 1))

# Compute Euclidean distances and sort input array for each batch
for i in range(N):
    distances = np.linalg.norm(xs[:, :2] - xs[i, :2], axis=1)
    sorted_indices = np.argsort(distances)
    sorted_xs = xs[sorted_indices]
    output[i, 1, :, :] = sorted_xs

print("Input array:")
print(xs)
print("\nOutput array:")
print(output)
