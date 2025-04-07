#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import fabada

import logging
logging.basicConfig(filename='log_susi.log', level=logging.DEBUG, format='%(asctime)s %(message)s')
import numpy as np

class Susi():
    """
    This is a simple version of Susi, using simple functions as "data" and
    "theory" instead of MD simulations and the corresponding scattering data.
    """
    def __init__(self,
                 pars={},
                 sampler=fabada.Fabada):
        self.pars = pars # always contains laTEST parameter values (updated after a complete iteration)
        self._meas = None # cache for the experimental data
        self.sampler = sampler(pars=self.pars, meas=self.meas, theo=self.theo)
        self.log = self.sampler.log
        self.trace = self.sampler.trace
    
    def meas(self):
        """
        read the experimental data with error bars
        """
        if self._meas is None: # only calculate once, cache result
            if TEST == 0:
                # recreating example from doi:10.1088/1742-6596/325/1/012006
                xc = 5.0
                a = 10.0
                w = 1.0
                x = np.arange(-10, 10, 0.01)
                y = a / np.sqrt(2.*np.pi*w) * np.exp(-(x-xc)**2/(2.*w**2))
                err = 0.05
                y += np.random.normal(scale=err, size=y.shape)
                dy = np.ones_like(y) * err
            if TEST == 1:
                a = 2.0
                nu = 0.72
                ph = 1.5
                x = np.arange(0, 24, 0.1)
                noise = np.random.randn(len(x))
                dy = np.ones(len(x)) * np.std(noise)
                y = a*np.sin(nu*x+ph)
                y+= noise
            if TEST == 2:
                N = 100
                sigma = 1.
                x = np.linspace(0., 9., N)
                a = 0.4
                b = -0.4
                c = 3.0
                np.random.seed(716742)
                y = a*x**2 + b*x + c
                y += sigma * np.random.randn(N)
                dy = sigma * np.ones_like(y)

            self._meas = {'func': {'x': x,
                                   'y': y,
                                   'dy': dy}
                          }
        return self._meas

    def theo(self, pars=None):
        """
        calculate fit function using present parameter values

        """
        if pars is None:
            pars = self.pars
            
        if TEST == 0:
            x = self.meas()['func']['x']
            xc = pars['xc']
            a = pars['a']
            w = pars['w']
            y = a / np.sqrt(2.*np.pi*w) * np.exp(-(x-xc)**2/(2.*w**2))
        if TEST == 1:
            x = self.meas()['func']['x']
            a = pars['a']
            nu = pars['nu']
            ph = pars['ph']
            y = a*np.sin(nu*x+ph)
        if TEST == 2:
            x = self.meas()['func']['x']
            a = pars['a']
            b = pars['b']
            c = pars['c']
            y = a*x**2 + b*x + c
            
        return {'func': {'x': x,
                         'y': y}
                }
    
    def run(self):
        """
        iterate
        """
        iterations = 10**4
        percentage = 1
        for i in range(iterations):
            onepercent = iterations // 100
            if i % onepercent == 0:
                print('progress:', percentage, '/ 100')
                percentage += 1
            self.pars = self.sampler.walk(dump=False)
        #self.sampler.dump()
        
if __name__ == '__main__':
    TEST = 1
    #if TEST == 0: s = Susi(pars={'xc': 5.0, 'a': 10.0, 'w': 1.0})
    if TEST == 0: s = Susi(pars={'xc': 4.0, 'a': 8.0, 'w': 0.8})
    if TEST == 1: s = Susi(pars={'a': 2.0, 'nu': 0.72, 'ph': 1.5})
    if TEST == 2: s = Susi(pars={'a': 0.4, 'b': -0.4, 'c': 3.0})
    s.n_tune = 100
    s.run()
    
    myburn = 0

    # generate corner plot
    from corner import corner    
    tr = s.trace(burn=myburn)
    parameternames = list(tr[0]['pars'])
    parameters = np.empty((len(tr), len(parameternames)))
    for ti,tv in enumerate(tr):
        for pi,pv in enumerate(parameternames):
            parameters[ti,pi] = tv['pars'][pv]
    corner(parameters, labels=parameternames)
    
    # find best fit
    import numpy as np
    chisq = []
    for tv in tr:
        chisq.append(sum(tv['chis'].values()))
    chisq = np.array(chisq)
    best_iterations = np.where(chisq==chisq.min())[0]
    for i in best_iterations:
        print(tr[i])
    
    # plot curve and fits
    import matplotlib.pyplot as plt
    plt.figure()
    # measured data points
    for m_i in s.meas().values():
        plt.errorbar(m_i['x'], m_i['y'], yerr=m_i['dy'], fmt='b.', alpha=0.5)
    # best fits
    for i in best_iterations:
        par_i = tr[i]['pars']
        for t_i in s.theo(pars=par_i).values():
            plt.plot(t_i['x'], t_i['y'], '-', color='red', zorder=3)
    # some more fits in the background
    n = 100 # how many curves to plot in the background
    every = len(tr)//n
    for i in range(every, len(tr)+1, every):
        par_i = tr[i]['pars']
        for t_i in s.theo(pars=par_i).values():
            plt.plot(t_i['x'], t_i['y'], '-', color='orange', zorder=-1)
    plt.show()