import numpy as np
from numpy import dot
from numpy.linalg import eigh, eig
import pandas as pd
import plotly.express as px
from plotly import graph_objects as go
from scipy.optimize import curve_fit

def lorentz(X, mean, sigma, scale, offset):
    return scale / (np.pi * sigma * (1 + ((X - mean) / sigma) ** 2)) + offset


def lorentz_diff1(X, mean, sigma, scale, offset):
    return -(2 * sigma * (-mean + X)) / (
                np.pi * (mean ** 2 + sigma ** 2 - 2 * mean * X + np.power(X, 2)) ** 2) * scale + offset


def lorentz_diff2(X, mean, sigma, scale, offset):
    return -(2 * sigma * (sigma ** 2 - 3 * np.power(mean - X, 2))) / (
                np.pi * np.power(sigma ** 2 + np.power(mean - X, 2), 3)) * scale + offset


def lorentz_diff3(X, mean, sigma, scale, offset):
    return (-(48 * np.power(-mean + X, 3)) / (
                sigma ** 7 * np.pi * np.power(1 + np.power(-mean + X, 2) / sigma ** 2, 4)) + (24 * (-mean + X)) / (
                        sigma ** 5 * np.pi * np.power(1 + np.power(-mean + X, 2) / sigma ** 2, 3))) * scale + offset


def lorentz_spread(X, mean, sigma, scale, offset):
    return (mean ** 2 - 2 * mean * X + np.power(X, 2) - sigma ** 2) / (
                np.pi * (mean ** 2 - 2 * mean * X + np.power(X, 2) + sigma ** 2) ** 2) * scale + offset


def line(X, offset):
    return np.ones(len(X)) * offset


# def both_lorentz(X, mean1, sigma1, scale1, offset1, mean2, sigma2, scale2, offset2):
#     return lorentz(X, mean1, sigma1, scale1, offset1) + lorentz_shift(X, mean2, sigma2, scale2, offset2)

class Map:
    def __init__(self, src, x, y, **kwargs):
        raw_data = np.loadtxt(src, delimiter='\t')
        if 'begin' in kwargs and 'end' in kwargs:
            raw_data = raw_data[np.r_[0:1, (kwargs['begin'] + 2):(kwargs['end'] + 2)]]
        self.shape = (x, y)
        self.raman_shift = raw_data.T[0][2:]
        self.data_matrix = raw_data[2:].T[1:]
        self.data = pd.DataFrame(raw_data[2:], columns=['RamanShift'] + list(np.arange(0, x * y, 1)))
        self.map = np.array([raw_data[0][1:], raw_data[1][1:]]).T

        self.cov_matrix = np.cov(self.data_matrix)

        eigenvalues, eigenvectors = eigh(self.cov_matrix)
        idx = np.flip(eigenvalues.argsort())
        self.weights = eigenvalues[idx] / sum(eigenvalues)
        self.vectors = eigenvectors.T[idx]

        self.base_vectors = np.dot(self.vectors, self.data_matrix)

    def reconstruct(self, count):
        return np.dot(self.vectors[:count].T, self.base_vectors[:count])

    def plotComponent(self, num):
        fig = px.imshow(np.reshape(self.vectors[num], self.shape), title='Komponent ' + str(num))
        return fig

    def plotPhysicalComponent(self, v):
        img = np.dot(np.array(v), self.vectors[:len(v)])
        fig = px.imshow(np.reshape(img, self.shape), title='Peak')
        return fig

    def plotCovMatrix(self):
        fig = px.imshow(self.cov_matrix, title='Macierz kowariancji')
        return fig

    def plotWeights(self, cum, log):
        if cum:
            fig = px.bar(np.cumsum(self.weights), title='Wagi', log_y=log)
        else:
            fig = px.bar(self.weights, title='Wagi', log_y=log)
        return fig

    def plotSpectrums(self, ids, **kwargs):
        fig = go.Figure()
        if 'base' not in kwargs:
            for num in ids:
                fig.add_scatter(x=self.raman_shift, y=self.data_matrix[num], mode='lines', name='ID: ' + str(num))
        else:
            r = self.reconstruct(kwargs['base'])
            for num in ids:
                fig.add_scatter(x=self.raman_shift, y=r[num], mode='lines', name='ID: ' + str(num))
        fig.update_layout(hovermode="x")
        return fig

    def plotSpectrum(self, num, **kwargs):
        return self.plotSpectrums([num], **kwargs)

    def plotBaseVectors(self, count):
        fig = go.Figure()
        for num in range(count):
            fig.add_scatter(x=self.raman_shift, y=self.base_vectors[num], mode='lines', name='Num: ' + str(num))
        fig.update_layout(hovermode="x")
        return fig

    def locToId(self, x, y):
        return np.argmin(list(map(lambda e: np.linalg.norm([x, y] - e), self.map)))

    def posToId(self, x, y):
        return self.shape[0] * y + x

    def draw_curve(self, curve, params, a, b, **draw):
        fig = go.Figure()
        if 'spectrum' in draw:
            fig.add_scatter(x=self.raman_shift, y=self.data_matrix[draw['spectrum']], mode='lines')
        if 'base_vector' in draw:
            fig.add_scatter(x=self.raman_shift, y=self.base_vectors[draw['base_vector']], mode='lines')
        data_slice = self.data[(self.data['RamanShift'] >= a) & (self.data['RamanShift'] <= b)]['RamanShift']
        fig.add_scatter(mode='lines', x=data_slice, y=curve(data_slice, *params))
        fig.update_layout(hovermode="x")
        return fig

    def fit_curve(self, curve, p0, a, b, seek, **plot):
        y = 0
        if 'spectrum' in plot:
            y = self.data_matrix[plot['spectrum'], (self.data['RamanShift'] >= a) & (self.data['RamanShift'] <= b)]
        if 'base_vector' in plot:
            y = self.base_vectors[plot['base_vector'], (self.data['RamanShift'] >= a) & (self.data['RamanShift'] <= b)]
        data_slice = self.data[(self.data['RamanShift'] >= a) & (self.data['RamanShift'] <= b)]['RamanShift']

        bounds = 0
        if 'bounds' in plot and plot['bounds'] == 'inf':
            bounds = ((-np.inf, -np.inf, -np.inf, -np.inf), (np.inf, np.inf, np.inf, np.inf))
        else:
            bounds = ((p0[0] - 2, -np.inf, -np.inf, -np.inf), (p0[0] + 2, np.inf, np.inf, np.inf))

        param, cov = curve_fit(curve, data_slice, y, p0=p0, bounds=bounds)
        print(param)
        if 'draw' in plot and plot['draw'] == True:
            self.draw_curve(curve, param, a, b, **plot).show()
        return param[seek]

