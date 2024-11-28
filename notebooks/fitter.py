import numpy as np
from numpy import dot
from numpy.linalg import eigh, eig, norm, inv
import pandas as pd
import plotly.express as px
from plotly import graph_objects as go
from scipy.optimize import curve_fit
import rampy

def lorentz(X, scale, mean, sigma, offset):
    return scale / (np.pi * sigma * (1 + ((X - mean) / sigma) ** 2)) + offset

def normalize(vector):
    return vector / np.linalg.norm(vector)

class Fitter:
    def __init__(self, src):
        data = pd.read_csv(src, delimiter='\t', names=['x', 'y', 'RamanShift', 'Intensity'])
        self.raman_shift = data[(data['x'] == 0) & (data['y'] == 0)]['RamanShift'].values
        self.shape = (len(data['x'].unique()), len(data['y'].unique()))
        self.size = (data['x'].max() - data['x'].min(), data['y'].max() - data['y'].min())
        self.raw_spectra = data['Intensity'].values.reshape((self.shape[0] * self.shape[1], -1))
        
        self.spectra = self.raw_spectra.copy()

    def reset_spectra(self):
        self.spectra = self.raw_spectra.copy()

    def remove_baseline(self, roi, method, **kwargs):
        for spectrum in self.spectra:
            ycalc, base = rampy.baseline(self.raman_shift, spectrum, roi, baseline)
            spectrum = ycalc
        
        if 'plot' in kwargs:
            fig = go.Figure()
            fig.add_scatter(x=self.raman_shift, y=self.raw_spectra[kwargs['plot']], mode='lines', name='Raw')
            fig.add_scatter(x=self.raman_shift, y=base, mode='lines', name='Baseline')
            fig.update_layout(hovermode="x")
            return fig

    def smooth(self, **kwargs):
        for spectrum in self.spectra:
            spectrum = rampy.smooth(self.raman_shift, spectrum, **kwargs)

        if 'plot' in kwargs:
            fig = go.Figure()
            fig.add_scatter(x=self.raman_shift, y=self.raw_spectra[kwargs['plot']], name='Raw')
            fig.add_scatter(x=self.raman_shift, y=self.spectra[kwargs['plot']], mode='lines', name='Smoothed')
            fig.update_layout(hovermode="x")
            return fig

    def compare_spectra(self, id, **kwargs):
        fig = go.Figure()
        fig.add_scatter(x=self.raman_shift, y=self.raw_spectra[id], name='Raw', **kwargs)
        fig.add_scatter(x=self.raman_shift, y=self.spectra[id], mode='lines', name='Processed', **kwargs)
        fig.update_layout(hovermode="x")
        return fig

    def plotSpectra(self, *args, **kwargs):
        fig = go.Figure()
        for id in args:
            fig.add_scatter(x=self.raman_shift, y=self.spectra[id], mode='lines', name='ID: ' + str(id), **kwargs)
        fig.update_layout(hovermode="x")
        return fig

    def posToId(self, x, y):
        return self.shape[0] * y + x

    # def locToId(self, x, y):
    #     return np.argmin(list(map(lambda e: np.linalg.norm([x, y] - e), self.map)))

    # def draw_curve(self, curve, params, a, b, **draw):
    #     fig = go.Figure()
    #     if 'spectrum' in draw:
    #         fig.add_scatter(x=self.raman_shift, y=self.data_matrix[draw['spectrum']], mode='lines')
    #     if 'base_vector' in draw:
    #         fig.add_scatter(x=self.raman_shift, y=self.base_vectors[draw['base_vector']], mode='lines')
    #     data_slice = self.data[(self.data['RamanShift'] >= a) & (self.data['RamanShift'] <= b)]['RamanShift']
    #     fig.add_scatter(mode='lines', x=data_slice, y=curve(data_slice, *params))
    #     fig.update_layout(hovermode="x")
    #     return fig
    #
    # def fit_curve(self, curve, p0, a, b, seek, **plot):
    #     y = 0
    #     if 'spectrum' in plot:
    #         y = self.data_matrix[plot['spectrum'], (self.data['RamanShift'] >= a) & (self.data['RamanShift'] <= b)]
    #     if 'base_vector' in plot:
    #         y = self.base_vectors[plot['base_vector'], (self.data['RamanShift'] >= a) & (self.data['RamanShift'] <= b)]
    #     data_slice = self.data[(self.data['RamanShift'] >= a) & (self.data['RamanShift'] <= b)]['RamanShift']
    #
    #     bounds = 0
    #     if 'bounds' in plot and plot['bounds'] == 'inf':
    #         bounds = ((-np.inf, -np.inf, -np.inf, -np.inf), (np.inf, np.inf, np.inf, np.inf))
    #     else:
    #         bounds = ((p0[0] - 2, -np.inf, -np.inf, -np.inf), (p0[0] + 2, np.inf, np.inf, np.inf))
    #
    #     param, cov = curve_fit(curve, data_slice, y, p0=p0, bounds=bounds)
    #     print(param)
    #     if 'draw' in plot and plot['draw'] == True:
    #         self.draw_curve(curve, param, a, b, **plot).show()
    #     return param[seek]
