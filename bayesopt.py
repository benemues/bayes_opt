import sys
print(sys.executable)

from bayes_opt import BayesianOptimization
import numpy as np
import matplotlib.pyplot as plt
import corner
from bayes_opt import acquisition

import logging
logging.basicConfig(filename='log_susi.log', level=logging.DEBUG, format='%(asctime)s %(message)s')

class SusiBO:
    def __init__(self, test, init_points: int, n_iter: int):
        """
        Initialize the Bayesian optimizer with test selection, initial random points, and number of iterations.
        """
        self.allit = 0
        self.test = test
        self.n_iter = n_iter
        self.init_points = init_points
        self._meas = None  # Cached measurement data
        self.init_params()

        acquisition_function = acquisition.UpperConfidenceBound(kappa=80)
        self.optimizer = BayesianOptimization(
            f=self.loss,
            pbounds=self.bounds,
            random_state=42,
            acquisition_function=acquisition_function
        )

    def init_params(self):
        """
        Set initial parameter bounds depending on the selected test case.
        """
        if self.test == 0:
            self.bounds = {'xc': (0, 10), 'a': (5, 15), 'w': (0.5, 2)}
        elif self.test == 1:
            self.bounds = {'a': (1, 3), 'nu': (0.5, 1), 'ph': (0, 3)}
        elif self.test == 2:
            self.bounds = {'a': (0, 1), 'b': (-1, 1), 'c': (0, 5)}

    def meas(self):
        """
        Generate synthetic measurement data including random noise, if not already generated.
        """
        if self._meas is None:
            if self.test == 0:
                xc, a, w = 5.0, 10.0, 1.0
                x = np.linspace(-10, 10, 200)
                y = a / np.sqrt(2 * np.pi * w) * np.exp(-(x - xc) ** 2 / (2 * w ** 2))
                dy = np.full_like(y, 0.05)
                y += np.random.normal(scale=0.05, size=y.shape)
            elif self.test == 1:
                a, nu, ph = 2.0, 0.72, 1.5
                x = np.linspace(0, 24, 240)
                noise = np.random.randn(len(x)) * 0.1
                y = a * np.sin(nu * x + ph) + noise
                dy = np.full_like(y, 0.1)
            elif self.test == 2:
                x = np.linspace(0, 9, 100)
                y = 0.4 * x ** 2 - 0.4 * x + 3.0 + np.random.normal(scale=1, size=x.shape)
                dy = np.ones_like(y)

            dy = gauss_random_multiply(dy)
            self._meas = {'x': x, 'y': y, 'dy': dy}

        return self._meas

    def theo(self, **pars):
        """
        Calculate the theoretical model output based on input parameters.
        """
        x = self.meas()['x']
        if self.test == 0:
            y = pars['a'] / np.sqrt(2 * np.pi * pars['w']) * np.exp(-(x - pars['xc']) ** 2 / (2 * pars['w'] ** 2))
        elif self.test == 1:
            y = pars['a'] * np.sin(pars['nu'] * x + pars['ph'])
        elif self.test == 2:
            y = pars['a'] * x ** 2 + pars['b'] * x + pars['c']
        return y

    def loss(self, **pars):
        """
        Compute negative chi-square loss (BayesianOptimization maximizes).
        """
        y_meas = self.meas()['y']
        y_theo = self.theo(**pars)
        dy = self.meas()['dy']
        return -(np.sum(((y_meas - y_theo) ** 2) / (dy ** 2)))

    def run(self):
        """
        Start the Bayesian optimization process with the given number of iterations.
        """
        self.best_fits = []
        self.allit += self.n_iter
        self.init_plots()
        self._optimize_loop()
        self.after_optimization()

    def continue_fitting(self):
        """
        Continue the optimization process after user input.
        """
        self.allit += self.n_iter
        self.optimizer.maximize(init_points=self.init_points, n_iter=0)
        self._optimize_loop()
        self.after_optimization()

    def _optimize_loop(self):
        """
        Internal method to perform the optimization loop for current iterations.
        """
        for self.i in range(self.n_iter):
            self.optimizer.maximize(init_points=0, n_iter=1)
            self.best_params = self.optimizer.max['params']
            self.update_plots()

    def after_optimization(self):
        """
        After finishing an optimization block: plot, visualize, and ask user to continue.
        """
        self.plot_param_evolution()
        self.plott_corner()
        self.want_to_continue()

    def want_to_continue(self):
        """
        Prompt the user whether to continue optimization or stop.
        """
        match input("Continue? y/n "):
            case "y":
                self.n_iter = int(input("How many additional iterations? "))
                self.continue_fitting()
            case "n":
                print("Optimization finished.")
                print(f"Best parameters found: {self.best_params}")
            case _:
                print("Invalid input! Please type 'y' or 'n'.")
                self.want_to_continue()

    def get_current_params(self):
        """
        Return the parameters of the most recent evaluation.
        """
        return self.optimizer.res[-1]['params']

    def plot_param_evolution(self):
        """
        Plot how each parameter evolves during the optimization.
        """
        plt.figure()
        params = self.optimizer.res
        x = list(range(len(params)))
        param_names = list(params[0]['params'].keys())

        for name in param_names:
            y = [res["params"][name] for res in params]
            plt.plot(x, y, label=f"{name}")

        plt.xlabel("Iteration")
        plt.ylabel("Parameter Value")
        plt.title("Parameter Evolution during Bayesian Optimization")
        plt.legend()
        plt.show()

    def plott_corner(self):
        """
        Create a corner plot to visualize parameter correlations.
        """
        params = self.optimizer.res
        param_names = list(params[0]['params'].keys())
        samples = np.array([[res['params'][name] for name in param_names] for res in params])
        corner.corner(samples, labels=param_names, show_titles=True, title_kwargs={"fontsize": 12})

    def init_plots(self):
        """
        Initialize interactive plots for live updating.
        """
        plt.ion()
        self.fig, self.ax = plt.subplots(1, 2, figsize=(10, 4))
        self.adjust_plot1()
        self.adjust_plot2()

    def adjust_plot1(self):
        """
        Setup the first plot for fits.
        """
        self.best_theo, = self.ax[0].plot([], [], 'r-', label='Best Fit', zorder=4, linewidth=2)
        self.current_plot, = self.ax[0].plot([], [], 'grey', label='Current Fit', zorder=2)
        self.errorbar = self.ax[0].errorbar([], [], yerr=[], fmt='o', color='blue', label='Measurements', linewidth=0.1)

        data = self.meas()
        self.ax[0].set_xlim(min(data['x']), max(data['x']))
        self.ax[0].set_ylim(min(data['y']), max(data['y']))
        self.ax[0].legend()

    def adjust_plot2(self):
        """
        Setup the second plot for loss values.
        """
        self.loss_graph, = self.ax[1].plot([], [], 'r-', label='Loss')
        self.ax[1].set_xlim(0, self.allit)
        self.ax[1].legend()

    def update_plots(self):
        """
        Update both interactive plots during optimization.
        """
        self.update_plot_1()
        self.update_plot_2()
        self.ax[0].relim()
        self.ax[1].relim()
        self.ax[0].autoscale_view()
        self.ax[1].autoscale_view()

    def update_plot_1(self):
        """
        Update fit comparison plot.
        """
        data = self.meas()
        xdata = data['x']
        ydata1 = data['y']
        dy = data['dy']

        besttheo_y = self.theo(**self.best_params)
        self.best_theo.set_xdata(xdata)
        self.best_theo.set_ydata(besttheo_y)

        self.errorbar = self.ax[0].errorbar(xdata, ydata1, yerr=dy, fmt='o', markersize=1, color='blue', zorder=3, linewidth=0.5)

        current_params = self.get_current_params()
        current_theo = self.theo(**current_params)
        self.current_plot.set_xdata(xdata)
        self.current_plot.set_ydata(current_theo)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def update_plot_2(self):
        """
        Update live loss tracking plot.
        """
        best_fit = self.loss(**self.best_params)
        self.best_fits.append(-best_fit)
        self.loss_graph.set_ydata(self.best_fits)
        self.iter = list(range(self.allit - self.n_iter + self.i + 1))
        self.loss_graph.set_xdata(self.iter)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def plot_loss_(self):
        """
        Plot the loss function by varying one parameter while fixing others.
        """
        plt.figure()

        param_names = list(self.bounds.keys())
        vary_param = param_names[0]  # Choose first parameter to vary
        fixed_params = {name: np.mean(bounds) for name, bounds in self.bounds.items() if name != vary_param}
        vary_bounds = self.bounds[vary_param]

        vary_range = np.linspace(vary_bounds[0], vary_bounds[1], 500)

        y = []
        for val in vary_range:
            params = fixed_params.copy()
            params[vary_param] = val
            y.append(self.loss(**params))

        plt.plot(vary_range, y, label=f"Loss vs {vary_param}")

        tried_vals = [res['params'][vary_param] for res in self.optimizer.res]
        tried_losses = [self.loss(**res['params']) for res in self.optimizer.res]
        plt.scatter(tried_vals, tried_losses, color="red", zorder=5, label="Evaluated Points")

        plt.xlabel(vary_param)
        plt.ylabel("Loss")
        plt.title(f"Loss Function w.r.t {vary_param}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_bo(self):
        """
        Plot the Bayesian optimization model prediction vs. real loss function.
        """
        param_names = list(self.bounds.keys())
        vary_param = param_names[2]
        fixed_params = {name: np.mean(bounds) for name, bounds in self.bounds.items() if name != vary_param}
        vary_bounds = self.bounds[vary_param]

        vary_range = np.linspace(vary_bounds[0], vary_bounds[1], 500)
        X = []
        for val in vary_range:
            params = fixed_params.copy()
            params[vary_param] = val
            X.append([params[name] for name in param_names])

        X = np.array(X)
        mean, sigma = self.optimizer._gp.predict(X, return_std=True)

        y_loss = [-self.loss(**{**fixed_params, vary_param: val}) for val in vary_range]

        tried_vals = [res['params'][vary_param] for res in self.optimizer.res]
        tried_losses = [-res['target'] for res in self.optimizer.res]

        plt.figure(figsize=(12, 6))
        plt.plot(vary_range, y_loss, label="True Loss", color='black', linestyle='--')
        plt.plot(vary_range, mean, label="Prediction", color='blue')
        plt.fill_between(vary_range, mean - sigma, mean + sigma, color='blue', alpha=0.2, label='Uncertainty')
        plt.scatter(tried_vals, tried_losses, color="red", s=30, zorder=5, label="Evaluated Points")

        plt.xlabel(vary_param)
        plt.ylabel("Loss")
        plt.title(f"BO Model Prediction w.r.t '{vary_param}'")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

def gauss_random_multiply(arr):
    """
    Randomly modify uncertainties by multiplying with Gaussian noise.
    """
    mean = np.mean(arr)
    sigma = 1
    ga_mu = np.random.normal(loc=mean, scale=sigma, size=arr.shape)
    ga_mu = np.abs(ga_mu)
    return arr * ga_mu

if __name__ == '__main__':
    TEST = 0
    susi = SusiBO(test=TEST, init_points=50, n_iter=70)
    susi.run()
