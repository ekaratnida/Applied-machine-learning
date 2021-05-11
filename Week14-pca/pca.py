from numpy import array
from numpy import mean
from numpy import cov
from numpy.linalg import eig

# define a small 3×2 matrix
matrix = array([[1, 3], [2, 3], [3, 4], [3, 5], [4, 4], [4, 6], [5, 6], [5, 7], [6, 8], [7, 8]])
print("original Matrix: ")
print(matrix)

# calculate the mean of each column
Mean_col = mean(matrix.T, axis=1)
print("Mean of each column: ")
print(Mean_col)

# center columns by subtracting column means
Centre_col = matrix - Mean_col
print("Covariance Matrix: ")
print(Centre_col)

# calculate covariance matrix of centered matrix
cov_matrix = cov(Centre_col.T)
print(cov_matrix)

# eigendecomposition of covariance matrix
eigen_values, eigen_vectors = eig(cov_matrix)
print("Eigen vectors: ",eigen_vectors)
print("Eigen values: ",eigen_values)

# project data on the new axes
projected_data = eigen_vectors.T.dot(Centre_col.T)
print(projected_data.T)
