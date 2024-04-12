import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from functools import partial

from jax import grad, jit
from jax import lax
from jax import random
from jax import custom_jvp

import flax

import jax
import jax.numpy as jnp


from typing import NamedTuple



class FiniteDiffParams(NamedTuple):
    # PML factor
    SigmaMax: jnp.float32
    # vector of directions (x,y)
    x: jnp.ndarray
    y: jnp.ndarray
    # grid in x
    X: jnp.ndarray
    # grid in y 
    Y: jnp.ndarray
    # size of simulation
    nxi: jnp.int32
    nyi: jnp.int32
    nx: jnp.int32
    ny: jnp.int32
    npml: jnp.int32
    h: jnp.float32
    # vector of directions int (xi,yi)
    xi: jnp.ndarray
    yi: jnp.ndarray
    # grid on interior x
    Xi: jnp.ndarray
    # grid on interior y
    Yi: jnp.ndarray


def init_params(ax:jnp.float32, ay:jnp.float32, nxi:jnp.int32, nyi:jnp.int32, npml:jnp.int32, SigmaMax:jnp.float32):
    """ function to initialize the parameters
    ax:    length of the domain in the x direction
    ay:    length of the domian in the y direction
    nxi:   number of discretization points in the x direction
    nyi:   number of discretization points in the y direction
    npml:  number of discretization points for PML
    """

    hx = ax/(nxi-1)
    hy = ay/(nyi-1)

    nx = nxi + 2*npml
    ny = nyi + 2*npml

    xi = jnp.linspace(0.,ax-hx, nxi) - ax/2
    yi = jnp.linspace(0.,ay-hy, nyi) - ay/2

    x = jnp.concatenate((xi[0]+jnp.arange(-npml,0)*hx,xi,xi[-1]+jnp.arange(1,npml+1)*hx))
    y = jnp.concatenate((yi[0]+jnp.arange(-npml,0)*hy,yi,yi[-1]+jnp.arange(1,npml+1)*hy))

    [Xi, Yi] = jnp.meshgrid(xi,yi)
    [X, Y] = jnp.meshgrid(x,y)

    return FiniteDiffParams(SigmaMax, x, y, X, Y, nxi, nyi, nx, ny, npml, hx, xi, yi, Xi, Yi)


def _assert(cond, msg):
    if not cond:
        raise ValueError(msg)


@flax.struct.dataclass
class Weights:


    def fd_weights_all(self, x, x0=0, n=1):
        """
        Return finite difference weights for derivatives of all orders up to n.

        Parameters
        ----------
        x : vector, length m
            x-coordinates for grid points
        x0 : scalar
            location where approximations are to be accurate
        n : scalar integer
            highest derivative that we want to find weights for

        Returns
        -------
        weights :  array, shape n+1 x m
            contains coefficients for the j'th derivative in row j (0 <= j <= n)

        Notes
        -----
        The x values can be arbitrarily spaced but must be distinct and len(x) > n.

        The Fornberg algorithm is much more stable numerically than regular
        vandermonde systems for large values of n.

        See also
        --------
        fd_weights

        References
        ----------
        B. Fornberg (1998)
        "Calculation of weights_and_points in finite difference formulas",
        SIAM Review 40, pp. 685-691.

        http://www.scholarpedia.org/article/Finite_difference_method
        """
        m = len(x)
        _assert(n < m, 'len(x) must be larger than n')

        weights = jnp.zeros((m, n + 1))
        weights = self._fd_weights_all(weights, x, x0, n)
        return weights.T


    def _fd_weights_all(self, weights: jax.Array, x, x0, n):
        m = len(x)
        c_1, c_4 = 1, x[0] - x0
        weights = weights.at[0, 0].set(1)
        for i in range(1, m):
            j = jnp.arange(0, jnp.minimum(i, n) + 1)
            c_2, c_5, c_4 = 1, c_4, x[i] - x0
            for v in range(i):
                c_3 = x[i] - x[v]
                c_2, c_6, c_7 = c_2 * c_3, j * weights[v, j - 1], weights[v, j]
                weights = weights.at[v, j].set((c_4 * c_7 - c_6) / c_3)
            weights = weights.at[i, j].set(c_1 * (c_6 - c_5 * c_7) / c_2)
            c_1 = c_2
        return weights

    def __call__(self):
        def fd_weights(x, x0=0, n=1):
            """
            Return finite difference weights for the n'th derivative.

            Parameters
            ----------
            x : vector
                abscissas used for the evaluation for the derivative at x0.
            x0 : scalar
                location where approximations are to be accurate
            n : scalar integer
                order of derivative. Note for n=0 this can be used to evaluate the
                interpolating polynomial itself.

            Examples
            --------
            >>> import numpy as np
            >>> import numdifftools.fornberg as ndf
            >>> x = np.linspace(-1, 1, 5) * 1e-3
            >>> w = ndf.fd_weights(x, x0=0, n=1)
            >>> df = np.dot(w, np.exp(x))
            >>> np.allclose(df, 1)
            True

            See also
            --------
            fd_weights_all
            """
            return self.fd_weights_all(x, x0, n)[-1]
        return fd_weights


def DistribPML(nx:jnp.int32,ny:jnp.int32,nPML:jnp.int32,fac:jnp.int32) -> jax.Array:
    t = jnp.linspace(0,1,nPML,dtype=jnp.float32)

    sigmaX = jnp.zeros((nx,ny))
    sigmaX = sigmaX.at[:,:nPML].set(jnp.tile(fac*jnp.square(t[::-1]),(ny,1)))
    sigmaX = sigmaX.at[:,nx-nPML:].set(jnp.tile(fac*jnp.square(t),(ny,1)))

    sigmaY = jnp.zeros((ny,nx))
    sigmaY = sigmaY.at[:nPML,:].set(jnp.transpose(jnp.tile(fac*jnp.square(t[::-1]),(nx,1))))
    sigmaY = sigmaY.at[ny-nPML:,:].set(jnp.transpose(jnp.tile(fac*jnp.square(t),(nx,1))))

    sigmaXp = jnp.zeros((ny,nx))
    sigmaXp = sigmaXp.at[:,:nPML].set(jnp.tile(-2*fac*t[::-1],(ny,1)))
    sigmaXp = sigmaXp.at[:,nx-nPML:].set(jnp.tile(2*fac*t,(ny,1)))

    sigmaYp = jnp.zeros((ny,nx))
    sigmaYp = sigmaYp.at[:nPML,:].set(jnp.transpose(jnp.tile(-2*fac*t[::-1],(nx,1))))
    sigmaYp = sigmaYp.at[ny-nPML:,:].set(jnp.transpose(jnp.tile(2*fac*t,(nx,1))))

    return sigmaX.T, sigmaY.T, sigmaXp.T, sigmaYp.T


def FirstOrderDifferenceMatrix1d(nx,h,order):
    fdweights = Weights()
    Dx = sp.sparse.spdiags(jnp.tile(fdweights()(np.arange(order+1), order/2,1)/h,(nx,1)).T,
                           jnp.arange(-(order/2),(order/2)+1),(nx,nx)).tolil()
    for i in range(int(order/2)-1):
        weights = fdweights()(jnp.arange(0,order+3),i+1,1)/h
        Dx[i,:order+2] = weights[1:]

        weights = fdweights()(jnp.arange(order+3),order+2-(i+1),1)/h
        Dx[-(i-1),-(order+2):] = weights[:-1]
    return Dx


def SecondOrderDifferenceMatrix1d(nx, h, order):
    fdweights = Weights()
    D_xx = sp.sparse.spdiags(jnp.tile(fdweights()(jnp.arange(order+1), order/2,2)/h**2,(nx,1)).T,
                             jnp.arange(-(order/2),(order/2)+1),(nx,nx)).tolil()
    for i in range(int(order/2)-1):
        weights = fdweights()(jnp.arange(0,order+3),i+1,2)/h**2
        D_xx[i,:order+2]=weights[1:]

        weights = fdweights()(jnp.arange(order+3),order+2-(i+1),2)/h**2
        D_xx[-(i-1),-(order+2):]=weights[:-1]
    return D_xx


def HelmholtzMatrix(m,nx,ny,npml,h,fac,order,omega,form):
    n = nx*ny
    [sx, sy, sxp, syp] = DistribPML(nx,ny,npml,fac)
    Dxx1d = SecondOrderDifferenceMatrix1d(nx,h,order)
    Dyy1d = SecondOrderDifferenceMatrix1d(ny,h,order)
    Dx1d = FirstOrderDifferenceMatrix1d(nx,h,order)
    Dy1d = FirstOrderDifferenceMatrix1d(ny,h,order)  
    Dx = sp.sparse.kron(Dx1d,sp.sparse.eye(ny))
    Dy = sp.sparse.kron(sp.sparse.eye(nx),Dy1d)
    Dxx = sp.sparse.kron(Dxx1d,sp.sparse.eye(ny))
    Dyy = sp.sparse.kron(sp.sparse.eye(nx),Dyy1d)
    M = sp.sparse.spdiags(m.flatten(),0,(n,n))

    if form == 'compact':
        K  = Dxx + Dyy;
        Sx = sp.sparse.spdiags(sx.flatten(),0,(n,n))
        Sy = sp.sparse.spdiags(sy.flatten(),0,(n,n))
        H  = -omega**2*M + 1j*omega*M@(Sx+Sy) + M@Sx@Sy - K - \
             Dx@(sp.sparse.spdiags((sy.flatten()-sx.flatten())/(1j*omega+sx.flatten()),0,(n,n))@Dx)- \
             - Dy@(sp.sparse.spdiags((sx.flatten()-sy.flatten())/(1j*omega+sy.flatten()),0,(n,n))@Dy)
    elif form == 'compact_explicit':
        H = -omega**2*M + sp.sparse.spdiags(-1j/(omega*(npml-1)*h)*sxp.flatten()/jnp.power(1-1j/omega*sx.flatten(),3),0,(n,n))@Dx \
            + sp.sparse.spdiags(-1j/(omega*(npml-1)*h)*syp.flatten()/jnp.power(1-1j/omega*sy.flatten(),3),0,(n,n))@Dy \
            - sp.sparse.spdiags(1./jnp.square(1-1j/omega*sx.flatten()),0,(n,n))@Dxx \
            - sp.sparse.spdiags(1./jnp.square(1-1j/omega*sy.flatten()),0,(n,n))@Dyy
    
    #return H
    return jnp.array(H.todense())


def ExtendModel(m,nxint,nyint,npml):
    m = jnp.reshape(m,(nxint,nyint))
    nx = m.shape[0] + 2*npml
    ny = m.shape[1] + 2*npml

    mnew = jnp.zeros((nx,ny))
    mnew = mnew.at[npml:nx-npml,npml:ny-npml].set(m)
    mnew = mnew.at[:npml,:].set(jnp.tile(mnew[npml,:],(npml,1)))
    mnew = mnew.at[-npml:,:].set(jnp.tile(mnew[-npml-1,:],(npml,1)))
    mnew = mnew.at[:,:npml].set(jnp.tile(mnew[:,npml],(npml,1)).T)
    mnew = mnew.at[:,-npml:].set(jnp.tile(mnew[:,-npml-1],(npml,1)).T)
    mnew = mnew.flatten()
    return mnew
