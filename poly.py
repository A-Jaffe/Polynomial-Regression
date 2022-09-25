import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Training dataset
# x-axis: Time (minutes)
# y-axis: Cell Growth (number of cells)
time_minutes = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
number_of_cells = [[2000], [4000], [8000], [16000], [32000], [64000],
                   [128000], [256000], [512000], [1024000], [2048000]]

# Testing dataset
# x-axis time_minutes_test
# y-axis number_of_cells_test
time_minutes_test = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
number_of_cells_test = [[3000], [6000], [12000], [24000], [48000], [96000],
                        [192000], [384000], [768000], [1536000], [3072000]]

# Use sklearn LinearRegression object to plot a prediction
regressor = LinearRegression()
regressor.fit(time_minutes, number_of_cells)
xx = np.linspace(0, 10, 2100000)
yy = regressor.predict(xx.reshape(xx.shape[0], 1))
plt.plot(xx, yy)

# Use PolynimialFeatures to project training data into higher dimensional plain
quadratic_featurizer = PolynomialFeatures(degree = 5)

# Transform input data matrix into new matrix that fits quadratic shape
time_minutes_quadratic = quadratic_featurizer.fit_transform(time_minutes)
time_minutes_test_quadratic = quadratic_featurizer.transform(time_minutes_test)

# Train and test the regressor_quadratic using sklearn Linear object
regressor_quadratic = LinearRegression()
regressor_quadratic.fit(time_minutes_quadratic, number_of_cells)
xx_quadratic = quadratic_featurizer.transform(xx.reshape(xx.shape[0], 1))

# Plot the graph
plt.plot(xx, regressor_quadratic.predict(xx_quadratic), c='y', linestyle='--')

# Set Title and axes headings
plt.title('Cell Growth over 10 minutes')
plt.xlabel('Time (minutes)')
plt.ylabel('Number of cells (millions)')

# Set axis lengths and include grid pattern
plt.axis([0, 10, 0, 2100000])
plt.grid(True)

# Scatter training data and show the graph
plt.scatter(time_minutes, number_of_cells)
plt.show()
