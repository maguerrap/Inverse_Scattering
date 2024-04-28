import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from functools import partial

from jax import grad, jit, vmap
from jax import lax
from jax import custom_jvp
from jax import custom_vjp

from jax import random
import jax
import jax.numpy as jnp
import jax.scipy as jsp

import flax

from interpax import interp2d

from typing import NamedTuple

from jax_fd import init_params, FiniteDiffParams
from jax_fd import ExtendModel



def u_i(r, omega):
    # this is green's function defined by as the fundamental solution
    # note the sign, i.e., it satisfies,
    # Delta G(x,y) + omega^2 G(x,y) = - delta (x,y)
    return (1j/4)*sp.special.hankel1(0,omega*r)

def u_i_(x, s, omega):
    return jnp.exp(1j*omega*jnp.vdot(x,s))

def vu_i(X, Y, S, omega):
    return jnp.exp(1j*omega*(jnp.outer(X.flatten(),S[:,0])+jnp.outer(Y.flatten(),S[:,1])))

class NearFieldParams(NamedTuple):
    fd_params: FiniteDiffParams
    # incident wave
    u_i: jnp.ndarray


def init_params_near_field(ax:jnp.float32, ay:jnp.float32, nxi:jnp.int32,
                           nyi:jnp.int32, npml:jnp.int32, r:jnp.float32,
                           n_theta:jnp.int32, omega:jnp.float32, SigmaMax:jnp.float32) -> NearFieldParams:
    """ function to initialize the parameters
    ax:    length of the domain in the x direction
    ay:    length of the domian in the y direction
    nxi:   number of discretization points in the x direction
    nyi:   number of discretization points in the y direction
    npml:  number of discretization points for PML
    r:     radius of the observation manifold
    n_theta: number of samples and point sources in the obs manifold
    omega: frequency 
    """

    # initilize params for FD
    params = init_params(ax, ay, nxi, nyi, npml, SigmaMax)

    d_theta = jnp.pi*2/(n_theta)
    theta = jnp.linspace(jnp.pi, 3*jnp.pi-d_theta, n_theta)

    # defining the observation (and sampling) manifold
    X = jnp.concatenate((params.X, params.Y))
    S = r*jnp.concatenate((jnp.cos(theta).reshape((n_theta, 1)),\
                                  jnp.sin(theta).reshape((n_theta, 1))),\
                                  axis = 1)


    # computing the distances to each evaluation point
    #R = jnp.sqrt(  jnp.square(params.X.reshape((-1, 1))
    #                          - S[:,0].reshape((1, -1)))
    #             + jnp.square(params.Y.reshape((-1, 1))
    #                          - S[:,1].reshape((1, -1))))

    # here we are defining u_i function
    #vu_i = vmap(u_i_, in_axes=(0,0,None))  
    U_i = vu_i(params.X,params.Y,S, omega)

    return NearFieldParams(params, U_i)

@flax.struct.dataclass
class NearField:
    H_: jax.Array
    omega: jnp.float32
    params: NearFieldParams
    tol:jnp.float32
    def __call__(self):
        def near_field_map(eta_vect_ext: jax.Array) -> jax.Array:
            """
            function to compute the near field in a circle of radius 1
            """
            Rhs = -(self.omega**2)\
                *eta_vect_ext.reshape((-1,1))*self.params.u_i
            m_ext = 1 + eta_vect_ext
            M = jnp.diag(m_ext.flatten(),0)
            H = self.H_ - self.omega**2*M
            #u_ = sp.sparse.linalg.spsolve(H, Rhs)
            solver = vmap(partial(jsp.sparse.linalg.gmres, tol=self.tol), in_axes=[None, 1])
            u_ = solver(H, Rhs)[0].T

            return u_
        return near_field_map


def smoothing_solution(near_field,
                       params: NearFieldParams,
                       n_theta, r):
    

    dtheta = 2*np.pi/(n_theta)
    theta = dtheta*jnp.arange(n_theta)
    S = r*jnp.concatenate((jnp.cos(theta).reshape((n_theta, 1)),\
                                  jnp.sin(theta).reshape((n_theta, 1))),\
                                  axis = 1)
    points_query = 0.5*S
    near_field_real = jnp.real(near_field)
    near_field_imag = jnp.imag(near_field)

    Lambda_real = jnp.zeros((n_theta, n_theta))
    Lambda_imag = jnp.zeros((n_theta, n_theta))


    for i in range(n_theta):
        Lambda_real = Lambda_real.at[:,i].set(interp2d(points_query[:,0], points_query[:,1],
                                    params.fd_params.x,params.fd_params.y,
                                    near_field_real[:,i].reshape((params.fd_params.nx,params.fd_params.ny)).T))
        Lambda_imag = Lambda_imag.at[:,i].set(interp2d(points_query[:,0], points_query[:,1],
                                    params.fd_params.x,params.fd_params.y,
                                    near_field_imag[:,i].reshape((params.fd_params.nx,params.fd_params.ny)).T))
    
    Lambda = Lambda_real + 1j*Lambda_imag
    
    return Lambda



def get_projection_mat(params, n_theta, r):#
    dtheta = 2*np.pi/(n_theta)
    theta = dtheta*np.arange(n_theta)
    S = r*jnp.concatenate((jnp.cos(theta).reshape((n_theta, 1)),\
                                  jnp.sin(theta).reshape((n_theta, 1))),\
                                  axis = 1)
    points_query = 0.5*S
    Projection_mat = jnp.zeros((n_theta, params.fd_params.nx, params.fd_params.ny))
    for i in range(params.fd_params.nx):
        for j in range(params.fd_params.ny):
            mat_dummy = jnp.zeros((params.fd_params.nx, params.fd_params.ny))
            mat_dummy = mat_dummy.at[j,i].set(1)
            #rbs = sp.interpolate.RectBivariateSpline(params.fd_params.x,params.fd_params.y,
            #                                         mat_dummy)
            #Projection_mat[:,i,j] = rbs.ev(points_query[:,0], points_query[:,1])
            Projection_mat = Projection_mat.at[:,i,j].set(
               interp2d(points_query[:,0], points_query[:,1],params.fd_params.x,params.fd_params.y,
                                                     mat_dummy)
                                                     )
    return Projection_mat.reshape((n_theta, params.fd_params.nx*params.fd_params.ny))


@flax.struct.dataclass
class MisFit(NearField):
    Lambda: jax.Array
    Projection_mat: jax.Array

    def __call__(self):
        def misfit(eta_vect):

            eta_vect_ext = ExtendModel(eta_vect, self.params.fd_params.nxi,
                                       self.params.fd_params.nyi, self.params.fd_params.npml)

            near_field_ = NearField(self.H_, self.omega, self.params, self.tol)
            U = near_field_()(eta_vect_ext)

            scatter = self.Projection_mat@U

            residual = self.Lambda - scatter

            mis = 0.5*jnp.linalg.norm(residual)**2

            U_tot = U + self.params.u_i

            # computing the rhs for the adjoint system
            rhs_adj = self.omega**2*self.Projection_mat.T@residual
            

            m_ext = 1 + eta_vect_ext
            M = jnp.diag(m_ext.flatten(),0)
            H = self.H_ - self.omega**2*M

            # solving the adjoint system
            #B_adj = sp.sparse.linalg.splu(H1.H)
            #W_adj = B_adj.solve(rhs_adj)
            #W_adj = sp.sparse.linalg.spsolve(H1.H, rhs_adj)
            solver = vmap(partial(jsp.sparse.linalg.gmres, tol=self.tol), in_axes=[None, 1])
            W_adj = solver(H.T, rhs_adj)[0].T
            
            # computing the gradient
            grad = jnp.real(jnp.sum(jnp.conj(U_tot)*W_adj, axis=1))

            # reshaping and extrating the gradient
            grad = grad.reshape((self.params.fd_params.nx, self.params.fd_params.ny))
            grad = grad[self.params.fd_params.npml:self.params.fd_params.npml+self.params.fd_params.nxi, 
                        self.params.fd_params.npml:self.params.fd_params.npml+self.params.fd_params.nyi]

            dmis = grad.flatten()

            return mis, dmis
        return misfit



@flax.struct.dataclass
class GradMisFit(MisFit):

    def __call__(self):
        def grad_misfit(eta_vect):

            eta_vect_ext = ExtendModel(eta_vect, self.params.fd_params.nxi,
                                       self.params.fd_params.nyi, self.params.fd_params.npml)

            near_field_ = NearField(self.H_, self.omega, self.params, self.tol)
            U = near_field_()(eta_vect_ext)

            scatter = self.Projection_mat@U

            residual = self.Lambda - scatter

            U_tot = U + self.params.u_i

            # computing the rhs for the adjoint system
            rhs_adj = self.omega**2*self.Projection_mat.T@residual
            
            m_ext = 1 + eta_vect_ext
            M = jnp.diag(m_ext.flatten(),0)
            H = self.H_ - self.omega**2*M

            # solving the adjoint system
            #B_adj = sp.sparse.linalg.splu(H1.H)
            #W_adj = B_adj.solve(rhs_adj)
            #W_adj = sp.sparse.linalg.spsolve(H1.H, rhs_adj)
            solver = vmap(partial(jsp.sparse.linalg.gmres, tol=self.tol), in_axes=[None, 1])
            W_adj = solver(H.T, rhs_adj)[0].T
            
            # computing the gradient
            grad = jnp.real(jnp.sum(jnp.conj(U_tot)*W_adj, axis=1))

            # reshaping and extrating the gradient
            grad = grad.reshape((self.params.fd_params.nx, self.params.fd_params.ny))
            grad = grad[self.params.fd_params.npml:self.params.fd_params.npml+self.params.fd_params.nxi, self.params.fd_params.npml:self.params.fd_params.npml+self.params.fd_params.nyi]

            dmis = grad.flatten()

            return dmis
        return grad_misfit


def DisplayField(u,x,y,npml=None):
    nx = x.shape[0]
    ny = y.shape[0]
    h = x[1]-x[0]
    u_plot = u.reshape((ny,nx))
    if npml is None:
        if jnp.all(np.isreal(u_plot)):
            plt.imshow(u_plot)
            plt.colorbar()
            plt.show()
        else:
            fig, axs = plt.subplots(1,2, figsize=(10,5))
            im = axs[0].imshow(jnp.real(u_plot))
            axs[0].set_title('Real')
            im = axs[1].imshow(jnp.imag(u_plot))
            axs[1].set_title('Imaginary')
            fig.colorbar(im, ax=axs[:], shrink=0.6)
            plt.show()
    else:
        u_plot = u.reshape((ny,nx))
        if jnp.all(np.isreal(u_plot)):
            plt.imshow(u_plot)
            plt.colorbar()
            plt.show()
        else:
            fig, axs = plt.subplots(1,2, figsize=(10,5))
            axs[0].axhline(y=0.5, color='r', linestyle='-') 
            axs[0].axhline(y=-0.5, color='r', linestyle='-')
            axs[0].axvline(x=0.5, color='r', linestyle='-') 
            axs[0].axvline(x=-0.5, color='r', linestyle='-')
            im = axs[0].imshow(jnp.real(u_plot),extent=[x[0],x[-1],y[0],y[-1]])
            axs[0].set_title('Real')
            axs[1].axhline(y=0.5, color='r', linestyle='-') 
            axs[1].axhline(y=-0.5, color='r', linestyle='-')
            axs[1].axvline(x=0.5, color='r', linestyle='-') 
            axs[1].axvline(x=-0.5, color='r', linestyle='-')
            im = axs[1].imshow(jnp.imag(u_plot),extent=[x[0],x[-1],y[0],y[-1]])
            axs[1].set_title('Imaginary')
            fig.colorbar(im, ax=axs[:], shrink=0.6)
            plt.show()
