#Author:   Nabin Ray
import numpy as np

def forward_difference_table(x, y):
    n = len(y)
    table = np.zeros((n, n))
    table[:, 0] = y

    for j in range(1, n):
        for i in range(n - j):
            table[i, j] = table[i + 1, j - 1] - table[i, j - 1]

    return table

def backward_difference_table(x, y):
    n = len(y)
    table = np.zeros((n, n))
    table[:, 0] = y

    for j in range(1, n):
        for i in range(j, n):
            table[i, j] = table[i, j - 1] - table[i - 1, j - 1]

    return table

def newton_forward_interpolation(x, y, value):
    h = x[1] - x[0]
    table = forward_difference_table(x, y)

    u = (value - x[0]) / h
    result = y[0]
    term = 1

    for i in range(1, len(x)):
        term *= (u - (i - 1))
        result += (term * table[0, i]) / np.math.factorial(i)

    return result

def newton_backward_interpolation(x, y, value):
    h = x[1] - x[0]
    table = backward_difference_table(x, y)

    u = (value - x[-1]) / h
    result = y[-1]
    term = 1

    for i in range(1, len(x)):
        term *= (u + (i - 1))
        result += (term * table[-1, i]) / np.math.factorial(i)

    return result

# Input data
x = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
y = np.array([1.0000, 0.9975, 0.9900, 0.9776, 0.8604])

# Value to interpolate
value = 0.25

# Perform interpolation using both methods
forward_result = newton_forward_interpolation(x, y, value)
backward_result = newton_backward_interpolation(x, y, value)

print("Newton's Forward Interpolation result:", forward_result)
print("Newton's Backward Interpolation result:", backward_result)
 69 changes: 69 additions & 0 deletions69  
asignment4/q2.py
Original file line number	Diff line number	Diff line change
@@ -0,0 +1,69 @@
#Author:   Nabin Ray
import numpy as np

def forward_difference_table(x, y):
    n = len(y)
    table = np.zeros((n, n))
    table[:, 0] = y

    for j in range(1, n):
        for i in range(n - j):
            table[i, j] = table[i + 1, j - 1] - table[i, j - 1]

    return table

def backward_difference_table(x, y):
    n = len(y)
    table = np.zeros((n, n))
    table[:, 0] = y

    for j in range(1, n):
        for i in range(j, n):
            table[i, j] = table[i, j - 1] - table[i - 1, j - 1]

    return table

def newton_forward_interpolation(x, y, value):
    h = x[1] - x[0]
    table = forward_difference_table(x, y)

    u = (value - x[0]) / h
    result = y[0]
    term = 1

    for i in range(1, len(x)):
        term *= (u - (i - 1))
        result += (term * table[0, i]) / np.math.factorial(i)

    return result

def newton_backward_interpolation(x, y, value):
    h = x[1] - x[0]
    table = backward_difference_table(x, y)

    u = (value - x[-1]) / h
    result = y[-1]
    term = 1

    for i in range(1, len(x)):
        term *= (u + (i - 1))
        result += (term * table[-1, i]) / np.math.factorial(i)

    return result

# Function definition
f = lambda x: 2 * x**3 - 4 * x + 1

# Generate data
x = np.arange(2, 4.25, 0.25)
y = f(x)

# Value to interpolate
value = 2.25

# Perform interpolation using both methods
forward_result = newton_forward_interpolation(x, y, value)
backward_result = newton_backward_interpolation(x, y, value)

print("Newton's Forward Interpolation result at x=2.25:", forward_result)
print("Newton's Backward Interpolation result at x=2.25:", backward_result)
 37 changes: 37 additions & 0 deletions37  
asignment4/q3.py
Original file line number	Diff line number	Diff line change
@@ -0,0 +1,37 @@
#Author:   Nabin Ray
import numpy as np

def divided_difference_table(x, y):
    n = len(y)
    table = np.zeros((n, n))
    table[:, 0] = y

    for j in range(1, n):
        for i in range(n - j):
            table[i, j] = (table[i + 1, j - 1] - table[i, j - 1]) / (x[i + j] - x[i])

    return table

def newton_divided_difference(x, y, value):
    table = divided_difference_table(x, y)
    n = len(x)
    result = y[0]
    term = 1

    for i in range(1, n):
        term *= (value - x[i - 1])
        result += term * table[0, i]

    return result

# Input data
x = np.array([2, 4, 9, 10])
y = np.array([4, 56, 711, 980])

# Value to interpolate
value = 5

# Perform interpolation
result = newton_divided_difference(x, y, value)

print("Newton's Divided Difference result at x=5:", result)
 91 changes: 91 additions & 0 deletions91  
asignment4/q4.py
Original file line number	Diff line number	Diff line change
@@ -0,0 +1,91 @@
#Author:  Nabin Ray
import numpy as np

def divided_difference_table(x, y):
    n = len(y)
    table = np.zeros((n, n))
    table[:, 0] = y

    for j in range(1, n):
        for i in range(n - j):
            table[i, j] = (table[i + 1, j - 1] - table[i, j - 1]) / (x[i + j] - x[i])

    return table

def newton_divided_difference(x, y, value):
    table = divided_difference_table(x, y)
    n = len(x)
    result = y[0]
    term = 1

    for i in range(1, n):
        term *= (value - x[i - 1])
        result += term * table[0, i]

    return result

def simpsons_one_third_rule(x, y):
    n = len(x) - 1
    if n % 2 != 0:
        raise ValueError("Simpson's 1/3 rule requires an even number of intervals.")

    h = (x[-1] - x[0]) / n
    result = y[0] + y[-1]

    for i in range(1, n):
        if i % 2 == 0:
            result += 2 * y[i]
        else:
            result += 4 * y[i]

    return (h / 3) * result

def simpsons_three_eighth_rule(x, y):
    n = len(x) - 1
    if n % 3 != 0:
        raise ValueError("Simpson's 3/8 rule requires the number of intervals to be a multiple of 3.")

    h = (x[-1] - x[0]) / n
    result = y[0] + y[-1]

    for i in range(1, n):
        if i % 3 == 0:
            result += 2 * y[i]
        else:
            result += 3 * y[i]

    return (3 * h / 8) * result

def trapezoidal_rule(x, y):
    n = len(x) - 1
    h = (x[-1] - x[0]) / n
    result = y[0] + y[-1]

    for i in range(1, n):
        result += 2 * y[i]

    return (h / 2) * result

# Input data for integration
x = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
y = np.array([1.0000, 0.9975, 0.9900, 0.9776, 0.8604])

# Extracting the range of integration from x = 1.8 to x = 3.4
x_range = np.arange(1.8, 3.5, 0.1)
y_range = np.array([1.0000, 0.9975, 0.9900, 0.9776, 0.8604])  # Example; replace with actual values corresponding to x_range

# Perform integration using all three methods
try:
    simpsons_1_3_result = simpsons_one_third_rule(x_range, y_range)
    print("Simpson's 1/3 Rule result:", simpsons_1_3_result)
except ValueError as e:
    print(e)

try:
    simpsons_3_8_result = simpsons_three_eighth_rule(x_range, y_range)
    print("Simpson's 3/8 Rule result:", simpsons_3_8_result)
except ValueError as e:
    print(e)

trapezoidal_result = trapezoidal_rule(x_range, y_range)
print("Trapezoidal Rule result:", trapezoidal_result)
 78 changes: 78 additions & 0 deletions78  
asignment4/q5.py
Original file line number	Diff line number	Diff line change
@@ -0,0 +1,78 @@
#Author:   Nabin Ray
import numpy as np

def simpsons_one_third_rule(x, y):
    n = len(x) - 1
    if n % 2 != 0:
        raise ValueError("Simpson's 1/3 rule requires an even number of intervals.")

    h = (x[-1] - x[0]) / n
    result = y[0] + y[-1]

    for i in range(1, n):
        if i % 2 == 0:
            result += 2 * y[i]
        else:
            result += 4 * y[i]

    return (h / 3) * result

def simpsons_three_eighth_rule(x, y):
    n = len(x) - 1
    if n % 3 != 0:
        raise ValueError("Simpson's 3/8 rule requires the number of intervals to be a multiple of 3.")

    h = (x[-1] - x[0]) / n
    result = y[0] + y[-1]

    for i in range(1, n):
        if i % 3 == 0:
            result += 2 * y[i]
        else:
            result += 3 * y[i]

    return (3 * h / 8) * result

def trapezoidal_rule(x, y):
    n = len(x) - 1
    h = (x[-1] - x[0]) / n
    result = y[0] + y[-1]

    for i in range(1, n):
        result += 2 * y[i]

    return (h / 2) * result

# Define the function to integrate
f = lambda x: x**3 + 2

# Function to compute integration using different methods and segment counts
def integrate_using_methods(a, b, segments):
    x = np.linspace(a, b, segments + 1)
    y = f(x)

    try:
        trap_result = trapezoidal_rule(x, y)
        print(f"Trapezoidal Rule with {segments} segments: {trap_result:.6f}")
    except Exception as e:
        print(e)

    try:
        simp_1_3_result = simpsons_one_third_rule(x, y)
        print(f"Simpson's 1/3 Rule with {segments} segments: {simp_1_3_result:.6f}")
    except Exception as e:
        print(e)

    try:
        simp_3_8_result = simpsons_three_eighth_rule(x, y)
        print(f"Simpson's 3/8 Rule with {segments} segments: {simp_3_8_result:.6f}")
    except Exception as e:
        print(e)

# Integration range
a, b = 2, 4

# Perform integration for different segment counts
for segments in [2, 4, 6]:
    print(f"\nUsing {segments} segments:")
    integrate_using_methods(a, b, segments)
 78 changes: 78 additions & 0 deletions78  
asignment4/q6.py
Original file line number	Diff line number	Diff line change
@@ -0,0 +1,78 @@
#Author:   Nabin Ray
import numpy as np

def simpsons_one_third_rule(x, y):
    n = len(x) - 1
    if n % 2 != 0:
        raise ValueError("Simpson's 1/3 rule requires an even number of intervals.")

    h = (x[-1] - x[0]) / n
    result = y[0] + y[-1]

    for i in range(1, n):
        if i % 2 == 0:
            result += 2 * y[i]
        else:
            result += 4 * y[i]

    return (h / 3) * result

def simpsons_three_eighth_rule(x, y):
    n = len(x) - 1
    if n % 3 != 0:
        raise ValueError("Simpson's 3/8 rule requires the number of intervals to be a multiple of 3.")

    h = (x[-1] - x[0]) / n
    result = y[0] + y[-1]

    for i in range(1, n):
        if i % 3 == 0:
            result += 2 * y[i]
        else:
            result += 3 * y[i]

    return (3 * h / 8) * result

def trapezoidal_rule(x, y):
    n = len(x) - 1
    h = (x[-1] - x[0]) / n
    result = y[0] + y[-1]

    for i in range(1, n):
        result += 2 * y[i]

    return (h / 2) * result

# Define the function to integrate
f = lambda x: np.sin(x) / x

# Function to compute integration using different methods and segment counts
def integrate_using_methods(a, b, segments):
    x = np.linspace(a, b, segments + 1)
    y = np.where(x == 0, 1, f(x))  # Handle singularity at x = 0

    try:
        trap_result = trapezoidal_rule(x, y)
        print(f"Trapezoidal Rule with {segments} segments: {trap_result:.6f}")
    except Exception as e:
        print(e)

    try:
        simp_1_3_result = simpsons_one_third_rule(x, y)
        print(f"Simpson's 1/3 Rule with {segments} segments: {simp_1_3_result:.6f}")
    except Exception as e:
        print(e)

    try:
        simp_3_8_result = simpsons_three_eighth_rule(x, y)
        print(f"Simpson's 3/8 Rule with {segments} segments: {simp_3_8_result:.6f}")
    except Exception as e:
        print(e)

# Integration range
a, b = 0, 1

# Perform integration for different segment counts
for segments in [2, 4, 6]:
    print(f"\nUsing {segments} segments:")
    integrate_using_methods(a, b, segments)
 130 changes: 130 additions & 0 deletions130  
asignment4/qn10.py
Original file line number	Diff line number	Diff line change
@@ -0,0 +1,130 @@
#Author:   Nabin Ray
import numpy as np

def simpsons_one_third_rule(x, y):
    n = len(x) - 1
    if n % 2 != 0:
        raise ValueError("Simpson's 1/3 rule requires an even number of intervals.")

    h = (x[-1] - x[0]) / n
    result = y[0] + y[-1]

    for i in range(1, n):
        if i % 2 == 0:
            result += 2 * y[i]
        else:
            result += 4 * y[i]

    return (h / 3) * result

def simpsons_three_eighth_rule(x, y):
    n = len(x) - 1
    if n % 3 != 0:
        raise ValueError("Simpson's 3/8 rule requires the number of intervals to be a multiple of 3.")

    h = (x[-1] - x[0]) / n
    result = y[0] + y[-1]

    for i in range(1, n):
        if i % 3 == 0:
            result += 2 * y[i]
        else:
            result += 3 * y[i]

    return (3 * h / 8) * result

def trapezoidal_rule(x, y):
    n = len(x) - 1
    h = (x[-1] - x[0]) / n
    result = y[0] + y[-1]

    for i in range(1, n):
        result += 2 * y[i]

    return (h / 2) * result

def romberg_integration(f, a, b, tol=1e-6):
    R = [[0]]
    n = 1
    h = b - a
    R[0][0] = h * (f(a) + f(b)) / 2

    while True:
        h /= 2
        n *= 2
        trapezoid_sum = sum(f(a + (2 * i - 1) * h) for i in range(1, n // 2 + 1))
        R.append([0] * (len(R[-1]) + 1))
        R[-1][0] = R[-2][0] / 2 + h * trapezoid_sum

        for k in range(1, len(R[-1])):
            R[-1][k] = R[-1][k - 1] + (R[-1][k - 1] - R[-2][k - 1]) / (4**k - 1)

        if abs(R[-1][-1] - R[-2][-2]) < tol:
            return R[-1][-1]

def double_integral_trapezoidal(f, x_range, y_range, nx, ny):
    x = np.linspace(x_range[0], x_range[1], nx + 1)
    y = np.linspace(y_range[0], y_range[1], ny + 1)
    hx = (x_range[1] - x_range[0]) / nx
    hy = (y_range[1] - y_range[0]) / ny

    integral = 0
    for i in range(nx + 1):
        for j in range(ny + 1):
            weight = 1
            if i == 0 or i == nx:
                weight /= 2
            if j == 0 or j == ny:
                weight /= 2
            integral += weight * f(x[i], y[j])

    return integral * hx * hy

def double_integral_simpsons_one_third(f, x_range, y_range, nx, ny):
    if nx % 2 != 0 or ny % 2 != 0:
        raise ValueError("Simpson's 1/3 rule requires an even number of intervals in both directions.")

    x = np.linspace(x_range[0], x_range[1], nx + 1)
    y = np.linspace(y_range[0], y_range[1], ny + 1)
    hx = (x_range[1] - x_range[0]) / nx
    hy = (y_range[1] - y_range[0]) / ny

    integral = 0
    for i in range(nx + 1):
        for j in range(ny + 1):
            wx = 1
            wy = 1
            if i == 0 or i == nx:
                wx = 1
            elif i % 2 == 0:
                wx = 2
            else:
                wx = 4

            if j == 0 or j == ny:
                wy = 1
            elif j % 2 == 0:
                wy = 2
            else:
                wy = 4

            integral += wx * wy * f(x[i], y[j])

    return integral * hx * hy / 9

# Define the function to integrate
f = lambda x, y: x * y**2

# Integration ranges
x_range = (0, 1)
y_range = (0, 1)

# Number of segments
nx, ny = 4, 4

# Perform double integration
trapezoidal_result = double_integral_trapezoidal(f, x_range, y_range, nx, ny)
simpsons_result = double_integral_simpsons_one_third(f, x_range, y_range, nx, ny)

print(f"Double Integral using Trapezoidal Rule: {trapezoidal_result:.6f}")
print(f"Double Integral using Simpson's 1/3 Rule: {simpsons_result:.6f}")
 78 changes: 78 additions & 0 deletions78  
asignment4/qn7.py
Original file line number	Diff line number	Diff line change
@@ -0,0 +1,78 @@
#Author:   Nabin Ray
import numpy as np

def simpsons_one_third_rule(x, y):
    n = len(x) - 1
    if n % 2 != 0:
        raise ValueError("Simpson's 1/3 rule requires an even number of intervals.")

    h = (x[-1] - x[0]) / n
    result = y[0] + y[-1]

    for i in range(1, n):
        if i % 2 == 0:
            result += 2 * y[i]
        else:
            result += 4 * y[i]

    return (h / 3) * result

def simpsons_three_eighth_rule(x, y):
    n = len(x) - 1
    if n % 3 != 0:
        raise ValueError("Simpson's 3/8 rule requires the number of intervals to be a multiple of 3.")

    h = (x[-1] - x[0]) / n
    result = y[0] + y[-1]

    for i in range(1, n):
        if i % 3 == 0:
            result += 2 * y[i]
        else:
            result += 3 * y[i]

    return (3 * h / 8) * result

def trapezoidal_rule(x, y):
    n = len(x) - 1
    h = (x[-1] - x[0]) / n
    result = y[0] + y[-1]

    for i in range(1, n):
        result += 2 * y[i]

    return (h / 2) * result

# Define the function to integrate
f = lambda x: 1 / (1 + x**2)

# Function to compute integration using different methods and segment counts
def integrate_using_methods(a, b, segments):
    x = np.linspace(a, b, segments + 1)
    y = f(x)

    try:
        trap_result = trapezoidal_rule(x, y)
        print(f"Trapezoidal Rule with {segments} segments: {trap_result:.6f}")
    except Exception as e:
        print(e)

    try:
        simp_1_3_result = simpsons_one_third_rule(x, y)
        print(f"Simpson's 1/3 Rule with {segments} segments: {simp_1_3_result:.6f}")
    except Exception as e:
        print(e)

    try:
        simp_3_8_result = simpsons_three_eighth_rule(x, y)
        print(f"Simpson's 3/8 Rule with {segments} segments: {simp_3_8_result:.6f}")
    except Exception as e:
        print(e)

# Integration range
a, b = 0, 1

# Perform integration for 8 segments
segments = 8
print(f"\nUsing {segments} segments:")
integrate_using_methods(a, b, segments)
 60 changes: 60 additions & 0 deletions60  
asignment4/qn8.py
Original file line number	Diff line number	Diff line change
@@ -0,0 +1,60 @@
#Author:   Nabin Ray
import numpy as np

def romberg_integration(f, a, b, n):
    """
    Perform Romberg integration to approximate the integral of f(x) from a to b.
    
    Parameters:
    f: callable
        Function to integrate.
    a: float
        Lower limit of integration.
    b: float
        Upper limit of integration.
    n: int
        Number of levels of Romberg integration.
        
    Returns:
    np.ndarray
        Romberg integration table.
    """
    R = np.zeros((n, n))

    # Trapezoidal rule for R[0, 0]
    R[0, 0] = (b - a) * (f(a) + f(b)) / 2.0

    # Fill the Romberg table
    for i in range(1, n):
        # Step size for the trapezoidal rule at this level
        h = (b - a) / (2 ** i)
        # Trapezoidal approximation
        sum_trap = sum(f(a + (2 * k - 1) * h) for k in range(1, 2 ** (i - 1) + 1))
        R[i, 0] = 0.5 * R[i - 1, 0] + h * sum_trap

        # Richardson extrapolation
        for j in range(1, i + 1):
            R[i, j] = R[i, j - 1] + (R[i, j - 1] - R[i - 1, j - 1]) / (4 ** j - 1)

    return R

# Define the function to integrate
def integrand(x):
    if x == 0:
        return 0  # Define value at x=0 to avoid division by zero
    return (np.sin(x) ** 2) / x

# Set parameters
a, b = 0, 1  # Integration bounds
n = 5        # Levels of Romberg integration

# Compute the Romberg integration table
romberg_table = romberg_integration(integrand, a, b, n)

# Print the Romberg table
print("Romberg Integration Table:")
print(romberg_table)

# Most accurate result (last cell of the last row)
result = romberg_table[-1, -1]
print(f"\nApproximate value of the integral: {result}")
 100 changes: 100 additions & 0 deletions100  
asignment4/qn9.py
Original file line number	Diff line number	Diff line change
@@ -0,0 +1,100 @@
#Author:   Sagar Thakur
import numpy as np

def simpsons_one_third_rule(x, y):
    n = len(x) - 1
    if n % 2 != 0:
        raise ValueError("Simpson's 1/3 rule requires an even number of intervals.")

    h = (x[-1] - x[0]) / n
    result = y[0] + y[-1]

    for i in range(1, n):
        if i % 2 == 0:
            result += 2 * y[i]
        else:
            result += 4 * y[i]

    return (h / 3) * result

def simpsons_three_eighth_rule(x, y):
    n = len(x) - 1
    if n % 3 != 0:
        raise ValueError("Simpson's 3/8 rule requires the number of intervals to be a multiple of 3.")

    h = (x[-1] - x[0]) / n
    result = y[0] + y[-1]

    for i in range(1, n):
        if i % 3 == 0:
            result += 2 * y[i]
        else:
            result += 3 * y[i]

    return (3 * h / 8) * result

def trapezoidal_rule(x, y):
    n = len(x) - 1
    h = (x[-1] - x[0]) / n
    result = y[0] + y[-1]

    for i in range(1, n):
        result += 2 * y[i]

    return (h / 2) * result

def romberg_integration(f, a, b, tol=1e-6):
    R = [[0]]
    n = 1
    h = b - a
    R[0][0] = h * (f(a) + f(b)) / 2

    while True:
        h /= 2
        n *= 2
        trapezoid_sum = sum(f(a + (2 * i - 1) * h) for i in range(1, n // 2 + 1))
        R.append([0] * (len(R[-1]) + 1))
        R[-1][0] = R[-2][0] / 2 + h * trapezoid_sum

        for k in range(1, len(R[-1])):
            R[-1][k] = R[-1][k - 1] + (R[-1][k - 1] - R[-2][k - 1]) / (4**k - 1)

        if abs(R[-1][-1] - R[-2][-2]) < tol:
            return R[-1][-1]

# Define the function to integrate
f = lambda x: 1 + x**3

# Integration range
a, b = 0, 1

# Perform Romberg integration
romberg_result = romberg_integration(f, a, b)
print(f"Romberg Integration Result: {romberg_result:.6f}")

# Perform integration for 8 segments using other methods
segments = 8
print(f"\nUsing {segments} segments:")
def integrate_using_methods(a, b, segments):
    x = np.linspace(a, b, segments + 1)
    y = f(x)

    try:
        trap_result = trapezoidal_rule(x, y)
        print(f"Trapezoidal Rule with {segments} segments: {trap_result:.6f}")
    except Exception as e:
        print(e)

    try:
        simp_1_3_result = simpsons_one_third_rule(x, y)
        print(f"Simpson's 1/3 Rule with {segments} segments: {simp_1_3_result:.6f}")
    except Exception as e:
        print(e)

    try:
        simp_3_8_result = simpsons_three_eighth_rule(x, y)
        print(f"Simpson's 3/8 Rule with {segments} segments: {simp_3_8_result:.6f}")
    except Exception as e:
        print(e)

integrate_using_methods(a, b, segments)