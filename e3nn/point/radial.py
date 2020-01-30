# pylint: disable=arguments-differ, no-member, missing-docstring, invalid-name, line-too-long
import math
from functools import partial

from scipy.special import binom
import torch
import torch.nn.functional as F


class ConstantRadialModel(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(d))

    def forward(self, _radii):
        batch = _radii.size(0)
        return self.weight.view(1, -1).expand(batch, -1)


class FiniteElementModel(torch.nn.Module):
    def __init__(self, out_dim, position, basis, Model):
        '''
        :param out_dim: output dimension
        :param position: tensor [i, ...]
        :param basis: scalar function: tensor [a, ...] -> [a]
        :param Model: Class(d1, d2), trainable model: R^d1 -> R^d2
        '''
        super().__init__()
        self.register_buffer('position', position)
        self.basis = basis
        self.f = Model(len(position), out_dim)

    def forward(self, x):
        """
        :param x: tensor [batch, ...]
        :return: tensor [batch, dim]
        """
        diff = x.unsqueeze(1) - self.position.unsqueeze(0)  # [batch, i, ...]
        batch, n, *rest = diff.size()
        x = self.basis(diff.view(-1, *rest)).view(batch, n)  # [batch, i]
        return self.f(x)


class FC(torch.nn.Module):
    def __init__(self, d1, d2, h, L, act):
        super().__init__()

        weights = []

        hh = d1
        for _ in range(L):
            weights.append(torch.nn.Parameter(torch.randn(h, hh)))
            hh = h

        weights.append(torch.nn.Parameter(torch.randn(d2, hh)))
        self.weights = torch.nn.ParameterList(weights)
        self.act = act

    def forward(self, x):
        L = len(self.weights) - 1

        if L == 0:
            W = self.weights[0]
            h = x.size(1)
            return x @ (W.t() / h ** 0.5)

        for i, W in enumerate(self.weights):
            h = x.size(1)

            if i == 0:
                # note: normalization assumes that the sum of the inputs is 1
                x = self.act(x @ W.t())
            elif i < L:
                x = self.act(x @ (W.t() / h ** 0.5))
            else:
                x = x @ (W.t() / h ** 0.5)

        return x


def FiniteElementFCModel(out_dim, position, basis, h, L, act):
    Model = partial(FC, h=h, L=L, act=act)
    return FiniteElementModel(out_dim, position, basis, Model)


def CosineBasisModel(out_dim, max_radius, number_of_basis, h, L, act):
    radii = torch.linspace(0, max_radius, steps=number_of_basis)
    step = radii[1] - radii[0]
    basis = lambda x: x.div(step).add(1).relu().sub(2).neg().relu().add(1).mul(math.pi / 2).cos().pow(2)
    return FiniteElementFCModel(out_dim, radii, basis, h, L, act)


class GaussianFunctions(torch.nn.Module):
    def __init__(self, num_basis_functions):
        super().__init__()
        self.num_basis_functions = num_basis_functions
        self.register_buffer('center', torch.linspace(1, 0, self.num_basis_functions))
        self.register_buffer('width', torch.tensor(1.0 * self.num_basis_functions))
        self._alpha = torch.nn.Parameter(torch.Tensor(1))
        self.alpha = lambda: F.softplus(self._alpha)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.constant_(self._alpha, softplus_inverse(0.9448630629184640))  # 0.5 in bohr

    def forward(self, r, cutoff_values=None):
        if cutoff_values is None:
            cutoff_values = torch.ones(r.shape).to(r)
        rbf = cutoff_values.view(-1, 1) * torch.exp(-self.width * (torch.exp(-self.alpha() * r.view(-1, 1)) - self.center)**2)
        return rbf


def softplus_inverse(x):
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    return x + torch.log(-torch.expm1(-x))


class BernsteinPolynomials(torch.nn.Module):
    def __init__(self, num_basis_functions):
        super(BernsteinPolynomials, self).__init__()
        self.num_basis_functions = num_basis_functions
        nu = torch.arange(0, self.num_basis_functions)
        self.register_buffer('coeff', torch.tensor(binom(self.num_basis_functions - 1, nu),
                             dtype=torch.double))
        self.register_buffer('pow1', torch.tensor((self.num_basis_functions - 1) - nu,
                             dtype=torch.double))
        self.register_buffer('pow2', torch.tensor(nu, dtype=torch.double))
        self._alpha = torch.nn.Parameter(torch.Tensor(1))
        self.alpha = lambda: F.softplus(self._alpha)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.constant_(self._alpha, softplus_inverse(0.9448630629184640))  # 0.5 in bohr

    def forward(self, r, cutoff_values=None):
        if cutoff_values is None:
            cutoff_values = torch.ones(r.shape).to(r)
        x = torch.exp(-self.alpha() * r.view(-1, 1))
        rbf = cutoff_values.view(-1, 1) * (self.coeff * x**self.pow1 * (1 - x)**self.pow2)
        return rbf


class ExponentialBernsteinPolynomials(torch.nn.Module):
    def __init__(self, num_basis_functions, no_basis_function_at_infinity=False,
                 ini_alpha=0.9448630629184640):  # 0.5/Bohr converted to 1/Angstrom
        super(ExponentialBernsteinPolynomials, self).__init__()
        self.num_basis_functions = num_basis_functions
        self.ini_alpha = ini_alpha
        self.no_basis_function_at_infinity = no_basis_function_at_infinity
        if self.no_basis_function_at_infinity:  # increase number of basis functions by one
            num_basis_functions += 1

        # compute values to initialize buffers
        logfactorial = torch.zeros((num_basis_functions, ))
        for i in range(2, num_basis_functions):
            logfactorial[i] = logfactorial[i - 1] + torch.log(torch.Tensor([i]))
        v = torch.arange(0, num_basis_functions)
        n = (num_basis_functions - 1) - v
        logbinomial = logfactorial[-1] - logfactorial[v] - logfactorial[n]
        if self.no_basis_function_at_infinity:  # remove last basis function at infinity
            v = v[:-1]
            n = n[:-1]
            logbinomial = logbinomial[:-1]
        # register buffers and parameters
        self.register_buffer('logc', torch.tensor(logbinomial, dtype=torch.float64))
        self.register_buffer('n', torch.tensor(n, dtype=torch.float64))
        self.register_buffer('v', torch.tensor(v, dtype=torch.float64))
        self.register_parameter('_alpha', torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float64)))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.constant_(self._alpha, softplus_inverse(self.ini_alpha))

    def forward(self, r, cutoff_values):
        alpha = F.softplus(self._alpha)
        x = -alpha * r.view(-1, 1)
        ones = torch.ones_like(x)
        x_ = torch.where(x == 0, ones, x)  # making sure there are no zeros to avoid NaNs
        x_ = self.logc + self.n * x_ + self.v * torch.log(-torch.expm1(x_))
        x_ = torch.where(x == 0, x, x_)
        rbf = cutoff_values.view(-1, 1) * torch.exp(x_)
        return rbf


class FiniteBasisModel(torch.nn.Module):
    def __init__(self, out_dim, max_radius, num_of_basis, basis, Model):
        '''
        :param out_dim: output dimension
        :param position: tensor [i, ...]
        :param basis: scalar function: tensor [a, ...] -> [a]
        :param Model: Class(d1, d2), trainable model: R^d1 -> R^d2
        '''
        super().__init__()
        self.max_radius = max_radius
        self.basis = basis(num_of_basis)
        self.f = Model(num_of_basis, out_dim)

    def forward(self, x):
        """
        :param x: tensor [batch, ...]
        :return: tensor [batch, dim]
        """

        cutoff_values = x < self.max_radius
        cutoff_values = cutoff_values.to(x)
        x = self.basis(x, cutoff_values)
        return self.f(x)


def GaussianBasisModel(out_dim, max_radius, number_of_basis, h, L, act):
    Model = partial(FC, h=h, L=L, act=act)
    return FiniteBasisModel(out_dim, max_radius, number_of_basis, GaussianFunctions, Model)


def BernsteinBasisModel(out_dim, max_radius, number_of_basis, h, L, act):
    Model = partial(FC, h=h, L=L, act=act)
    return FiniteBasisModel(out_dim, max_radius, number_of_basis, BernsteinPolynomials, Model)


def ExponentialBernsteinBasisModel(out_dim, max_radius, number_of_basis, h, L, act):
    Model = partial(FC, h=h, L=L, act=act)
    return FiniteBasisModel(out_dim, max_radius, number_of_basis, ExponentialBernsteinPolynomials, Model)
