import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd

# Generate sample data
x = np.linspace(0, 10, 50)
y = 2 * x**2 - 3 * x + 1 + np.random.normal(0, 1, 50)  # Adding some noise

# Save the data in a .csv file
df_generated = pd.DataFrame({'x': x, 'y': y})
df_generated.to_csv("data.csv", index=False)

# -------------------- Start of curve fitting code --------------------------

# Read the data from the .csv file
df_read = pd.read_csv("data.csv")

x_read = df_read['x']
y_read = df_read['y']

# Define the function you want to fit (example: a quadratic function)
def custom_function(x, a, b, c):
    return a * x**2 + b * x + c

# Perform curve fitting
params, covariance = curve_fit(custom_function, x_read, y_read)

# Extract the fitted parameters
a_fit, b_fit, c_fit = params

# Predict y values using the fitted parameters
y_fit = custom_function(x_read, a_fit, b_fit, c_fit)

# Plot the results
plt.scatter(x_read, y_read, label='Data')
plt.plot(
  x_read, 
  y_fit, 
  color='red', 
  label=f'Fitted Curve: $y = {a_fit:.2f}x^2 + {b_fit:.2f}x + {c_fit:.2f}$', linewidth=2
)
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Curve Fitting Example')
plt.show()

# Print the fitted parameters
print(f'Fitted Parameters (a, b, c): {a_fit}, {b_fit}, {c_fit}')