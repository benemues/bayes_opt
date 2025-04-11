from bayes_opt import BayesianOptimization
import numpy as np
import matplotlib.pyplot as plt
import corner

import logging
logging.basicConfig(filename='log_susi.log', level=logging.DEBUG, format='%(asctime)s %(message)s')

class SusiBO:
    def __init__(self, test, init_points:int, n_iter: int):
        self.allit = 0
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
        
    def plot_param_evolution(self):
        plt.figure() 
        params = self.optimizer.res
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
        
    def plott_corner(self):
        params = self.optimizer.res
        param_names = list(params[0]['params'].keys())
        samples = np.array([[res['params'][name] for name in param_names] for res in params])
        fig = corner.corner(samples, labels=param_names, show_titles=True, title_kwargs={"fontsize": 12})
        
    def continue_fitting(self):
        self.allit += self.n_iter
        self.init_plots()
        for self.i in range(self.n_iter):
            self.optimizer.maximize(init_points=0, n_iter=1)
            self.best_params = self.optimizer.max['params']
            self.update_plots()
        self.plot_param_evolution()
        self.plott_corner()
        self.want_to_continue()

    def want_to_continue(self):
        match input("Continue? y/n "):
            case "y":
                self.n_iter = int(input("How many iterations? "))
                self.continue_fitting()
                self.continue_ = True
            case "n":
                print("Ok")
                print(f"best_params: {self.best_params}")
            case _:
                print("Invalid input!")
                self.want_to_continue()
        
    def get_current_params(self):
        return (self.optimizer.res[-1]['params'])
    
    
    def plot_results(self):
        """Plots the measured data and the best model fit."""
        plt.figure()
        data = self.meas()
        plt.errorbar(data['x'], data['y'], yerr=data['dy'], fmt='b.', alpha=0.5, label='Measured data')
        best_fit = self.theo(**self.best_params)
        plt.plot(data['x'], best_fit, 'r-', label='Best fit')
        plt.legend()
        plt.show()
        
    def run(self):
        """Runs Bayesian Optimization."""
        self.best_fits = []
        self.allit += self.n_iter
        self.init_plots()
        for self.i in range(self.n_iter):
            self.optimizer.maximize(init_points=0, n_iter=1)
            self.best_params = self.optimizer.max['params']
            self.update_plots()
        self.plot_param_evolution()
        self.plott_corner()
        self.want_to_continue()
        
    def init_plots(self):
        #Init the plot
        plt.ion()
        
        #Init the different graphs
        self.fig, self.ax = plt.subplots(1, 2, figsize=(10, 4))
        
        #Adjust the plots
        self.adjust_plot1()
        self.adjust_plot2()   
    
    def update_plots(self):
        self.update_plot_1()
        self.update_plot_2()
        self.ax[0].relim()
        self.ax[1].relim()
        self.ax[1].autoscale_view()
        self.ax[1].autoscale_view()     

    def adjust_plot1(self):
        #Make the three graphs
        self.best_theo, = self.ax[0].plot([], [], 'r-', label='best fit', zorder=4, linewidth = 2)
        self.current_plot, = self.ax[0].plot([], [], 'grey', label='current fit', zorder=2)
        self.errorbar = self.ax[0].errorbar([], [], yerr=[], fmt='o', color='blue', label='Measurements', linewidth=0.1)
        
        #Adjust the looks of the plot
        data = self.meas()
        self.ax[0].set_xlim(min(data ['x']), max(data['x']))
        self.ax[0].set_ylim(min(data ['y']), max(data['y']))
        self.ax[0].legend()
        
    def update_plot_1(self):
        #Get the data
        data = self.meas()
        xdata = data['x']
        ydata1 = data ['y']
        dy = data ['dy']

        #Get the best fit
        besttheo_y = self.theo(**self.best_params)  
        self.best_theo.set_xdata(xdata)
        self.best_theo.set_ydata(besttheo_y) 
        
        #Get the errorbar
        self.errorbar = self.ax[0].errorbar(xdata, ydata1, yerr=dy, fmt='o', markersize=1, color='blue', zorder=3, linewidth=0.5)
        
        #Create the current graph
        current_params = self.get_current_params()
        current_theo = self.theo(**current_params)
        self.current_plot.set_xdata(xdata)
        self.current_plot.set_ydata(current_theo)
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def adjust_plot2(self):
        #Make a graph
        self.loss_graph, = self.ax[1].plot([], [], 'r-', label='loss')
        
        #Adjust the plot
        self.ax[1].set_xlim(0, self.allit)
        self.ax[1].legend()
        
    def update_plot_2(self):
        best_fit = self.loss(**self.best_params)
        self.best_fits.append(-best_fit)
        self.loss_graph.set_ydata(self.best_fits)
        self.iter = list(range(self.allit - self.n_iter + self.i +1 ))
        self.loss_graph.set_xdata(self.iter)
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
def gauss_random_multiply(arr):
    mean = np.mean(arr)
    sigma = 1
    ga_mu = np.random.normal(loc=mean, scale=sigma, size=arr.shape)
    ga_mu = np.abs(ga_mu)
    return arr * ga_mu           
      
if __name__ == '__main__':
    TEST = 1  # Choose the model
    susi = SusiBO(test=TEST, init_points=50, n_iter=200)    
    susi.run()