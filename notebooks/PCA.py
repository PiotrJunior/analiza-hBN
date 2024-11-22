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


def normalize(vector):
    return vector / np.linalg.norm(vector)

class Map:
    refit_vectors = 0
    refit_matrices = []

    def __init__(self, src, **kwargs):
        # raw_data = np.loadtxt(src, delimiter='\t')
        # if 'begin' in kwargs and 'end' in kwargs:
        #     raw_data = raw_data[np.r_[0:1, (kwargs['begin'] + 2):(kwargs['end'] + 2)]]
        # self.shape = ( len(np.where( raw_data[0][2:] == 0)), len(np.where( raw_data[1][2:] == 0)) )
        # print(self.shape)
        # self.raman_shift = raw_data.T[0][2:]
        # self.data_matrix = raw_data[2:].T[1:]
        # # self.data = pd.DataFrame(raw_data[2:], columns=['RamanShift'] + list(np.arange(0, x * y, 1)))
        # self.map = np.array([raw_data[0][1:], raw_data[1][1:]]).T

        data = pd.read_csv(src, delimiter='\t', names=['x', 'y', 'RamanShift', 'Intensity'])
        if 'begin' in kwargs and 'end' in kwargs:
            data = data[(data['RamanShift'] >= kwargs['begin']) & (data['RamanShift'] <= kwargs['end'])]

        self.raman_shift = data[(data['x'] == 0) & (data['y'] == 0)]['RamanShift'].values
        self.shape = (len(data['x'].unique()), len(data['y'].unique()))
        self.data_matrix = data['Intensity'].values.reshape((self.shape[0] * self.shape[1], -1))
        if 'baseline' and 'roi' in kwargs:
            for spectrum in self.data_matrix:
                ycalc, base = rampy.baseline(self.raman_shift, spectrum, kwargs['roi'], kwargs['baseline'])
                spectrum -= base.T[0]

        self.cov_matrix = np.cov(self.data_matrix)

        eigenvalues, eigenvectors = eigh(self.cov_matrix)
        idx = np.flip(eigenvalues.argsort())
        self.weights = eigenvalues[idx] / sum(eigenvalues)
        self.vectors = eigenvectors.T[idx]

        self.base_vectors = np.dot(self.vectors, self.data_matrix)

    def reconstruct(self, count):
        return np.dot(self.vectors[:count].T, self.base_vectors[:count])

    def reftit_matrix(self):
        rm = np.identity(self.refit_vectors)
        for refit_matrix in self.refit_matrices:
            rm = np.dot(refit_matrix, rm)
        return rm

    def physical_base_vectors(self):
        return np.dot(self.reftit_matrix(), self.base_vectors[:self.refit_vectors])

    def physical_vectors(self):
        return np.dot(self.vectors[:self.refit_vectors].T, inv(self.reftit_matrix())).T

    def plotComponent(self, num, physical=False, log=False):
        img = self.vectors[num]
        if log:
            img = np.log(img + 1)
        if physical:
            img = self.physical_vectors()[num]
        fig = px.imshow(np.reshape(img, self.shape), title='Component ' + str(num))
        return fig

    def plotBaseVector(self, num, physical=False, normalize_vector=False):
        vector = self.base_vectors[num]
        if physical:
            vector = self.physical_base_vectors()[num]
        if normalize_vector:
            return px.line(x=self.raman_shift, y=normalize(vector), title='Num: ' + str(num))
        else:
            return px.line(x=self.raman_shift, y=vector, title='Num: ' + str(num))

    def plotSpectrums(self, *args, **kwargs):
        fig = go.Figure()
        for num in args:
            fig.add_scatter(x=self.raman_shift, y=self.data_matrix[num], mode='lines', name='ID: ' + str(num), **kwargs)
        fig.update_layout(hovermode="x")
        return fig

    def plotSpectrumWithLorenzian(self, num, amp, cen, wid, off):
        fig = go.Figure()
        fig.add_scatter(x=self.raman_shift, y=self.data_matrix[num], mode='lines', name='ID: ' + str(num))
        fig.add_scatter(x=self.raman_shift, y=lorentz(self.raman_shift, amp, cen, wid, off), mode='lines', name='Lorentzian')
        fig.update_layout(hovermode="x")
        return fig

    def plotBaseVectors(self, count, normalize_vector=False, **kwargs):
        fig = go.Figure()
        for num in range(count):
            if normalize_vector:
                fig.add_scatter(x=self.raman_shift, y=normalize(self.base_vectors[num]),
                                mode='lines', name='Num: ' + str(num), **kwargs)
            else:
                fig.add_scatter(x=self.raman_shift, y=self.base_vectors[num],
                                mode='lines', name='Num: ' + str(num), **kwargs)
        fig.update_layout(hovermode="x")
        return fig

    def addRefitMatrix(self, matrix, normalized=False):
        if matrix.shape[0] != matrix.shape[1]:
            raise Exception('Matrix is not square')
        if matrix.shape[0] != self.refit_vectors:
            raise Exception('Matrix has wrong size')

        if normalized:
            for i, vector in enumerate(matrix):
                for j, value in enumerate(vector):
                    matrix[i][j] *= norm(self.physical_base_vectors()[i]) / norm(self.physical_base_vectors()[j])

        self.refit_matrices.append(matrix)

    def posToId(self, x, y):
        return self.shape[0] * y + x

    def plotCovMatrix(self):
        fig = px.imshow(self.cov_matrix, title='Macierz kowariancji')
        return fig

    def plotWeights(self, cum=False, log=False):
        if cum:
            return px.bar(np.cumsum(self.weights), title='Wagi', log_y=log)
        else:
            return px.bar(self.weights, title='Wagi', log_y=log)

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
