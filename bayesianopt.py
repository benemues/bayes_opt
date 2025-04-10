from bayes_opt import BayesianOptimization
import numpy as np
import matplotlib.pyplot as plt
import pickle
import random


import logging
logging.basicConfig(filename='log_susi.log', level=logging.DEBUG, format='%(asctime)s %(message)s')

class SusiBO:
    def __init__(self, test, init_points:int, n_iter: int):
        self.test = test
        self.n_iter = n_iter
        self.init_points = init_points
        self._meas = None  # Cache for experimental data
        self.init_params()
        self.optimizer = BayesianOptimization(f=self.loss, pbounds=self.bounds, random_state=42)


    def init_params(self):
        """Sets initial parameter bounds based on TEST value."""
        if self.test == 0:
            self.bounds = {'xc': (0, 10), 'a': (5, 15), 'w': (0.5, 2)}
        elif self.test == 1:
            self.bounds = {'a': (1, 3), 'nu': (0.5, 1), 'ph': (0, 3)}
        elif self.test == 2:
            self.bounds = {'a': (0, 1), 'b': (-1, 1), 'c': (0, 5)}

    def meas(self):
        """Generates experimental data with noise."""
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
            #print(f"typedy {type(dy)}")
            dy = gauss_random_multiply(dy)
            self._meas = {'x': x, 'y': y, 'dy': dy}
        return self._meas
    
    def theo(self, **pars):
        """Computes the theoretical model based on current parameters."""
        x = self.meas()['x']
        if self.test == 0:
            y = pars['a'] / np.sqrt(2 * np.pi * pars['w']) * np.exp(-(x - pars['xc']) ** 2 / (2 * pars['w'] ** 2))
        elif self.test == 1:
            y = pars['a'] * np.sin(pars['nu'] * x + pars['ph'])
        elif self.test == 2:
            y = pars['a'] * x ** 2 + pars['b'] * x + pars['c']
        return y

    def loss(self, **pars):
        """Computes the error between measured data and model prediction."""
        y_meas = self.meas()['y']
        y_theo = self.theo(**pars)
        dy = self.meas() ["dy"]
        return -(np.sum(((y_meas - y_theo) **2) / (dy**2)))  # Negative sum since bayes_opt maximizes
        
    def plot_param_evolution(self, params):
        x = list(range(len(params)))
        param_names = list(params[0]['params'].keys())

        for name in param_names:
            y = [res["params"][name] for res in params]
            plt.plot(x, y, label=f"{name}")

        plt.xlabel("Iteration")
        plt.ylabel("Parameterwert")
        plt.title("Parameterentwicklung während der Bayesian Optimization")
        plt.legend()
        plt.show()

    def want_to_continue(self):
        match input("Continue? y/n "):
            case "y":
                n_iter = int(input("How many iterations? "))
                self.continue_fitting(n_iter)
                
            case "n":
                print("Ok")
                print(f"best_params: {self.best_params}")
            case _:
                print("Invalid input!")
                self.want_to_continue()

    def run(self):
        """Runs Bayesian Optimization."""
        self.optimizer.maximize(init_points=self.init_points, n_iter=self.n_iter)
        #self.sausage_plot()
        self.best_params = self.optimizer.max['params']
        with open("optimizer.pkl", "wb") as f:
            pickle.dump(self.optimizer, f)
            
        self.plot_param_evolution(self.optimizer.res)
        
        self.plot_results()

        self.want_to_continue()
            
    def sausage_plot(self):
        gp = self.optimizer._gp
        X_obs = self.optimizer._space.params
        X_obs = np.array([[x ["x"]]for x in X_obs])
        y_obs = np.array(self.optimizer._space.target)
        
        x_ = np.linspace(0, 2, 400).reshape(-1, 1)
        y_mean, y_std = gp.predict(x_, return_std=True)
        
        plt.figure(figsize=(10, 5))
        plt.plot(x_, y_mean, "b-", label="GP mean prediction")
        plt.fill_between(
            x_.ravel(),
            y_mean - y_std,
            y_mean + y_std,
            alpha = 0.3,
            color="blue",
            label="uncertainty"
        )
     
        plt.scatter(X_obs, y_obs, c="red", s=50, zorder=10, label="Observations")
        plt.title("Uncertainty plot")
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.legend()
        plt.grid(True),
        plt.show()

    def plot_results(self):
        """Plots the measured data and the best model fit."""
        plt.figure()
        data = self.meas()
        plt.errorbar(data['x'], data['y'], yerr=data['dy'], fmt='b.', alpha=0.5, label='Measured data')
        best_fit = self.theo(**self.best_params)
        plt.plot(data['x'], best_fit, 'r-', label='Best fit')
        plt.legend()
        plt.show()
        
    def continue_fitting(self, n_iter):
        with open("optimizer.pkl", "rb") as f:
            self.optimizer = pickle.load(f)
        
        self.optimizer.maximize(n_iter=n_iter)
        self.best_params = self.optimizer.max['params']
        
        self.plot_param_evolution(self.optimizer.res)
        
        self.plot_results()

        with open("optimizer.pkl", "wb") as f:
            pickle.dump(self.optimizer, f)
            
        self.want_to_continue()
    
def gauss_random_multiply(arr):
    mean = np.mean(arr)
    sigma = 0.01
    ga_mu = np.random.normal(loc=mean, scale=sigma, size=arr.shape)
    ga_mu = np.abs(ga_mu)
    return arr*ga_mu           
      
if __name__ == '__main__':
    TEST = 1  # Choose the model
    susi = SusiBO(test=TEST, init_points=0, n_iter=4)    
    susi.run()
    