import numpy as np
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization
from bayes_opt import acquisition

np.random.seed(42)

# Einfach hier den Testfall wählen: 0 (Gauss), 1 (Sinus), 2 (Polynom)
test = 0

# Messdaten
meas_data = {}

def meas():
    """Generates experimental data with noise."""
    global meas_data
    if test == 0:
        xc, a, w = 5.0, 10.0, 1.0
        x = np.linspace(-10, 10, 200)
        y = a / np.sqrt(2 * np.pi * w) * np.exp(-(x - xc) ** 2 / (2 * w ** 2))
        dy = np.full_like(y, 0.05)
        y += np.random.normal(scale=0.05, size=y.shape)
    elif test == 1:
        a, nu, ph = 2.0, 0.72, 1.5
        x = np.linspace(0, 24, 240)
        noise = np.random.randn(len(x)) * 0.1
        y = a * np.sin(nu * x + ph) + noise
        dy = np.full_like(y, 0.1)
    elif test == 2:
        x = np.linspace(0, 9, 100)
        y = 0.4 * x ** 2 - 0.4 * x + 3.0 + np.random.normal(scale=1, size=x.shape)
        dy = np.ones_like(y)
    meas_data = {'x': x, 'y': y, 'dy': dy}

meas()

# Modellfunktion
def theo(**pars):
    x = meas_data['x']
    if test == 0:
        return pars['a'] / np.sqrt(2 * np.pi * pars['w']) * np.exp(-(x - pars['xc']) ** 2 / (2 * pars['w'] ** 2))
    elif test == 1:
        return pars['a'] * np.sin(pars['nu'] * x + pars['ph'])
    elif test == 2:
        return pars['a'] * x ** 2 + pars['b'] * x + pars['c']

# Ziel-Funktion (neg. Fehlerquadratsumme)
def build_target_function(fixed_params):
    def f(**kwargs):
        pars = {**fixed_params, **kwargs}
        y_meas = meas_data['y']
        y_theo = theo(**pars)
        return -np.sum((y_meas - y_theo) ** 2)
    return f

# Parametergrenzen je nach Testfall
if test == 0:
    pbounds = {"xc": (0, 10), "a": (5, 15), "w": (0.1, 5)}
    fixed = {}
elif test == 1:
    pbounds = {"a": (0.5, 5), "nu": (0.1, 2), "ph": (0, 3.14)}
    fixed = {}
elif test == 2:
    pbounds = {"a": (-2, 2), "b": (-2, 2)}
    fixed = {"c": 3.0}

# BO Setup
f = build_target_function(fixed)
acq_func = acquisition.UpperConfidenceBound(kappa=5)

bo = BayesianOptimization(
    f=f,
    pbounds=pbounds,
    verbose=2,
    random_state=987234,
    acquisition_function=acq_func
)

# Universelle Plot-Funktion: scannt über einen Parameter, fixiert alle anderen
def plot_loss_function(bo):
    keys = list(pbounds.keys())
    scan_key = keys[0]  # Erstes zu scannendes Parameter
    scan_bounds = pbounds[scan_key]
    
    # Verwende beste bisher bekannte Werte für Fixierung
    fixed_params = bo.max["params"].copy()
    if scan_key not in fixed_params:
        return  # Noch keine sinnvolle Iteration
    
    x = np.linspace(*scan_bounds, 1000).reshape(-1, 1)
    
    def f_scan(val):
        params = fixed_params.copy()
        params[scan_key] = val
        return f(**params)

    y = np.array([f_scan(val[0]) for val in x])

    # Für GP-Vorhersage alle Parameter übergeben
    X_pred = []
    for val in x:
        row = []
        for k in pbounds:
            if k == scan_key:
                row.append(val[0])
            else:
                row.append(fixed_params[k])
        X_pred.append(row)
    X_pred = np.array(X_pred)

    mean, sigma = bo._gp.predict(X_pred, return_std=True)

    # Plot
    plt.clf()
    plt.plot(x, y, label=f"True objective ({scan_key} variiert)", color="blue")
    plt.plot(x, mean, label="GP mean", color="green")
    plt.fill_between(x.ravel(), mean - sigma, mean + sigma, alpha=0.2, color="green", label="GP std dev")
    plt.scatter([res["params"][scan_key] for res in bo.res if scan_key in res["params"]],
                [res["target"] for res in bo.res], color="red", label="Samples")
    plt.xlabel(scan_key)
    plt.ylabel("Loss")
    plt.title(f"Loss-Funktion bei Variation von '{scan_key}'")
    plt.legend()
    plt.grid()
    plt.pause(0.1)

# Endplot: Messdaten + bestes Modell
def plot_best_fit(bo):
    x = meas_data['x']
    y_meas = meas_data['y']
    plt.figure(figsize=(10, 5))
    plt.errorbar(x, y_meas, yerr=meas_data['dy'], fmt='o', label='Messdaten', alpha=0.7)

    best_params = {**bo.max["params"], **fixed}
    y_best = theo(**best_params)
    label = " + ".join([f"{k}={v:.3f}" for k, v in bo.max["params"].items()])
    plt.plot(x, y_best, label=f"Theorie: {label}", color='red', linewidth=2)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Messdaten vs. Modell (beste Parameter)")
    plt.legend()
    plt.grid()
    plt.show()

# Interaktive Optimierung
def interactive_optimization(bo, num_iterations):
    plt.ion()
    for i in range(num_iterations):
        print(f"\nIteration {i + 1} von {num_iterations}")
        bo.maximize(n_iter=1, init_points=0)
        plot_loss_function(bo)
        input("Drücke Enter für die nächste Iteration...")
    plt.ioff()
    plot_best_fit(bo)

# Start
interactive_optimization(bo, num_iterations=40)
