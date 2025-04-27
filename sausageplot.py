import numpy as np
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization, acquisition

# Set random seed for reproducibility
np.random.seed(42)

# Select test case: 0 = Gaussian, 1 = Sine, 2 = Polynomial
test_case = 0

# Global dictionary to store measurement data
measurement_data = {}

def generate_measurements():
    """Generates synthetic experimental data with noise based on the selected test case."""
    global measurement_data
    if test_case == 0:
        # Gaussian peak
        xc, a, w = 5.0, 10.0, 1.0
        x = np.linspace(-10, 10, 200)
        y = a / np.sqrt(2 * np.pi * w) * np.exp(-(x - xc) ** 2 / (2 * w ** 2))
        dy = np.full_like(y, 0.05)
        y += np.random.normal(scale=0.05, size=y.shape)
    elif test_case == 1:
        # Sine wave
        a, nu, ph = 2.0, 0.72, 1.5
        x = np.linspace(0, 24, 240)
        noise = np.random.randn(len(x)) * 0.1
        y = a * np.sin(nu * x + ph) + noise
        dy = np.full_like(y, 0.1)
    elif test_case == 2:
        # Polynomial curve
        x = np.linspace(0, 9, 100)
        y = 0.4 * x**2 - 0.4 * x + 3.0 + np.random.normal(scale=1.0, size=x.shape)
        dy = np.ones_like(y)
    
    measurement_data = {'x': x, 'y': y, 'dy': dy}

# Generate measurement data
generate_measurements()

def model_function(**params):
    """Theoretical model function depending on the test case."""
    x = measurement_data['x']
    if test_case == 0:
        return params['a'] / np.sqrt(2 * np.pi * params['w']) * np.exp(-(x - params['xc']) ** 2 / (2 * params['w'] ** 2))
    elif test_case == 1:
        return params['a'] * np.sin(params['nu'] * x + params['ph'])
    elif test_case == 2:
        return params['a'] * x**2 + params['b'] * x + params['c']

def build_target_function(fixed_parameters):
    """Constructs a target function for optimization (negative sum of squared errors)."""
    def target(**kwargs):
        params = {**fixed_parameters, **kwargs}
        y_measured = measurement_data['y']
        y_model = model_function(**params)
        return -np.sum((y_measured - y_model) ** 2)
    return target

# Define parameter bounds and fixed parameters based on test case
if test_case == 0:
    param_bounds = {"xc": (0, 10), "a": (5, 15), "w": (0.1, 5)}
    fixed_params = {}
elif test_case == 1:
    param_bounds = {"a": (0.5, 5), "nu": (0.1, 2), "ph": (0, np.pi)}
    fixed_params = {}
elif test_case == 2:
    param_bounds = {"a": (-2, 2), "b": (-2, 2)}
    fixed_params = {"c": 3.0}

# Create target function
target_function = build_target_function(fixed_params)

def plot_loss_function(optimizer):
    """Plots the objective function, Gaussian Process prediction, and uncertainty."""
    keys = list(param_bounds.keys())
    scan_key = keys[0]  # Choose first parameter for scanning
    scan_range = param_bounds[scan_key]
    
    if scan_key not in optimizer.max["params"]:
        return  # No valid max point yet

    # Create fine grid for the scan
    x_values = np.linspace(*scan_range, 1000).reshape(-1, 1)

    # Evaluate true objective function
    def scan_function(val):
        params = optimizer.max["params"].copy()
        params[scan_key] = val
        return target_function(**params)

    y_true = np.array([scan_function(val[0]) for val in x_values])

    # Predict with GP
    X_pred = []
    for val in x_values:
        row = []
        for k in param_bounds:
            row.append(val[0] if k == scan_key else optimizer.max["params"][k])
        X_pred.append(row)
    X_pred = np.array(X_pred)

    mean, sigma = optimizer._gp.predict(X_pred, return_std=True)

    # Plotting
    plt.clf()
    plt.plot(x_values, y_true, label=f"True objective ({scan_key} varied)", color="blue")
    plt.plot(x_values, mean, label="GP prediction", color="green")
    plt.fill_between(x_values.ravel(), mean - sigma, mean + sigma, alpha=0.2, color="green", label="Uncertainty")
    plt.scatter([res["params"][scan_key] for res in optimizer.res if scan_key in res["params"]],
                [res["target"] for res in optimizer.res], color="red", label="Sampled points")
    plt.xlabel(scan_key)
    plt.ylabel("Objective")
    plt.title(f"Objective vs. '{scan_key}'")
    plt.legend()
    plt.grid()
    plt.pause(0.1)

def plot_best_fit(optimizer):
    """Plots measured data against the model with the best found parameters."""
    x = measurement_data['x']
    y_measured = measurement_data['y']
    
    plt.figure(figsize=(10, 5))
    plt.errorbar(x, y_measured, yerr=measurement_data['dy'], fmt='o', label='Measured Data', alpha=0.7)

    best_parameters = {**optimizer.max["params"], **fixed_params}
    y_best_fit = model_function(**best_parameters)
    label = " + ".join([f"{k}={v:.3f}" for k, v in optimizer.max["params"].items()])
    plt.plot(x, y_best_fit, label=f"Model Fit: {label}", color='red', linewidth=2)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Measured Data vs. Best Model Fit")
    plt.legend()
    plt.grid()
    plt.show()

def interactive_optimization(num_iterations, kappa_value):
    """Runs Bayesian optimization interactively with plotting after each iteration and allows dynamic kappa adjustment."""
    plt.ion()

    # Initialize optimizer with the specified kappa value
    acquisition_function = acquisition.UpperConfidenceBound(kappa=kappa_value)
    optimizer = BayesianOptimization(
        f=target_function,
        pbounds=param_bounds,
        verbose=2,
        random_state=42,
        acquisition_function=acquisition_function
    )
    
    for iteration in range(num_iterations):
        print(f"\nIteration {iteration + 1} of {num_iterations}")
        
        # Maximize with the new kappa value
        optimizer.maximize(init_points=0, n_iter=1)
        plot_loss_function(optimizer)
        
        input("Press Enter to continue to the next iteration...")
    plt.ioff()
    plot_best_fit(optimizer)

# Start the interactive optimization process with a dynamic kappa value
interactive_optimization(num_iterations=40, kappa_value=50)
