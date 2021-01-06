import numpy as np
from scipy.integrate import solve_ivp 

from swarm.utils import wiener, wiener_s1

#from numba import jit


class Swarmalator(object):
    
    def __init__(self, N=20, Iatt=None, Irep=None, Fatt=None, Frep=None, H=None, G=None, 
                 J1=1.0, K=0.1, n=2, tol=1e-8, option="vanilla"): 


        self.J1 = J1
        self.option = option
        self.forced = False

        if option == "vanilla":                
            if Iatt is None:
                Iatt = lambda xdiff, lx: 1
            self.Iatt = Iatt
                            
            if Fatt is None:
                Fatt = lambda tdiff: 1 + J1 * np.cos(tdiff)
            self.Fatt = Fatt    
                
            if Irep is None:
                Irep = lambda xdiff, lx: 1 / (tol + lx)
            self.Irep = Irep

            if Frep is None:
                Frep = lambda tdiff: 1.0
            self.Frep = Frep
                
            if H is None:
                H = lambda tdiff: np.sin(tdiff)
            self.H = H
                
            if G is None:
                G = lambda l: 1.0 / (tol + l)
            self.G = G

        elif option == "repulsive":
            if Iatt is None:
                Iatt = lambda xdiff, lx: 1.0
            self.Iatt = Iatt
                            
            if Fatt is None:
                Fatt = lambda tdiff: 1 + np.cos(tdiff)
            self.Fatt = Fatt    
                
            if Irep is None:
                Irep = lambda xdiff, lx: lx / J1 * np.exp(-(lx ** 2) / J1)
            self.Irep = Irep

            if Frep is None:
                Frep = lambda tdiff: 1.0
            self.Frep = Frep
                
            if H is None:
                H = lambda tdiff: np.sin(tdiff)
            self.H = H
                
            if G is None:
                G = lambda l: 1.0 / (tol + l)
            self.G = G

        self.n = n
        self.N = N
        self.K = K
        self.iterations = 0

        
    def simulate(self, tspan=[0, 100], max_iterations=100000, x0=None, theta0=None, thetavar=0.2, v=None, vvar=0.05, omega=None, omegavar=0.05, 
                 noise=False, sigma=0.0, **kwargs):

        if x0 is None:
            x0 = 1.0 * (1 - 2 * np.random.uniform(size=(self.n * self.N,)))
        elif x0 == 0:
            x0 = np.zeros((self.n * self.N,))
            
        if theta0 is None:
            theta0 = thetavar * np.random.uniform(size=(self.N,))
        elif theta0 == 0:
            theta0 = np.zeros((self.N,))
        elif isinstance(theta0, float) or isinstance(omega, int):
            theta0 = theta0 * np.ones((self.N,))
        elif theta0 == "spatial" and self.n == 2:
            theta0 = np.arctan2(x0[::2], x0[1::2])
        
        if v is None:
            v = vvar * (1 - 2 * np.random.uniform(size=(self.n * self.N,)))
        elif v == 0:
            v = np.zeros((self.n * self.N,))
 
        if omega is None:
            omega = omegavar * (1 - 2 * np.random.uniform(size=(self.N,)))
        elif omega == 0:
            omega = np.zeros((self.N,))  
        elif isinstance(omega, float) or isinstance(omega, int):
            omega = omega * np.ones((self.N,))
                        
        z0 = np.concatenate((x0, theta0))
        only_one = np.ones((self.N,))
        only_one[0] = 0.0

        zd = np.zeros((self.n,))
        if noise:
            s_func = wiener_s1(t_start=max(0.0, tspan[0] - 1.0), t_end=tspan[1] + 1.0, dt=0.01, n=self.N)
            x_func = wiener(t_start=max(0.0, tspan[0] - 1.0), t_end=tspan[1] + 1.0, dt=0.01, n=self.n)
        else:
            s_func = lambda t: 0.0
            x_func = lambda t: zd
            sigma = 0.0

        if self.n == 2:
            if noise:
                def dynamics(z, t):
                    x1 = z[:self.N]
                    x2 = z[self.N:2 * self.N]
                    theta = z[2 * self.N:]
                    
                    x1d = x1[:, np.newaxis] - x1 
                    x2d = x2[:, np.newaxis] - x2

                    thetad = theta[:, np.newaxis] - theta

                    norm_sq = x1d ** 2 + x2d ** 2
                    norm = np.sqrt(norm_sq)
                    xn = x_func(t)
                    inv_norm = 1.0 / (1e-5 + np.sqrt((x1 - xn[0]) ** 2 + (x2 - xn[1]) ** 2))

                    x1_rhs = v[:self.N] - 1.0 / self.N * np.sum((1.0 - np.eye(x1d.shape[0])) * \
                        (x1d / (1e-5 + norm) * (self.Iatt(x1d, norm) * self.Fatt(thetad) - self.Irep(x1d, norm) * self.Frep(thetad))), axis=1)
                    x2_rhs = v[self.N:] - 1.0 / self.N * np.sum((1.0 - np.eye(x2d.shape[0])) * \
                        (x2d / (1e-5 + norm) * (self.Iatt(x2d, norm) * self.Fatt(thetad) - self.Irep(x2d, norm) * self.Frep(thetad))), axis=1)

                    theta_rhs = omega + sigma * np.cos(s_func(t) - theta) * inv_norm - \
                        self.K / self.N * np.sum((1 - np.eye(theta.shape[0])) * (self.G(norm)  * self.H(thetad)), axis=1)

                    dz = np.concatenate((x1_rhs, x2_rhs, theta_rhs))
                    self.iterations += 1

                    return dz
            else:
                def dynamics(z, t):
                    x1 = z[:self.N]
                    x2 = z[self.N:2 * self.N]
                    theta = z[2 * self.N:]
                    
                    x1d = x1[:, np.newaxis] - x1 
                    x2d = x2[:, np.newaxis] - x2

                    thetad = theta[:, np.newaxis] - theta

                    norm_sq = x1d ** 2 + x2d ** 2
                    norm = np.sqrt(norm_sq)
                    #xn = x_func(t)
                    #inv_norm = 1.0 / (1e-5 + np.sqrt((x1 - xn[0]) ** 2 + (x2 - xn[1]) ** 2))

                    x1_rhs = v[:self.N] - 1.0 / self.N * np.sum((1.0 - np.eye(x1d.shape[0])) * \
                        (x1d / (1e-5 + norm) * (self.Iatt(x1d, norm) * self.Fatt(thetad) - self.Irep(x1d, norm) * self.Frep(thetad))), axis=1)
                    x2_rhs = v[self.N:] - 1.0 / self.N * np.sum((1.0 - np.eye(x2d.shape[0])) * \
                        (x2d / (1e-5 + norm) * (self.Iatt(x2d, norm) * self.Fatt(thetad) - self.Irep(x2d, norm) * self.Frep(thetad))), axis=1)

                    theta_rhs = omega - self.K / self.N * np.sum((1 - np.eye(theta.shape[0])) * (self.G(norm)  * self.H(thetad)), axis=1)

                    dz = np.concatenate((x1_rhs, x2_rhs, theta_rhs))
                    self.iterations += 1

                    return dz
                

        elif self.n == 3:
            if noise:
                def dynamics(z, t):
                    x1 = z[:self.N]
                    x2 = z[self.N:2 * self.N]
                    x3 = z[2 * self.N:3 * self.N]
                    theta = z[3 * self.N:]

                    x1d = x1[:, np.newaxis] - x1
                    x2d = x2[:, np.newaxis] - x2
                    x3d = x3[:, np.newaxis] - x3

                    thetad = theta[:, np.newaxis] - theta

                    norm_sq = x1d ** 2 + x2d ** 2 + x3d ** 2
                    norm = np.sqrt(norm_sq)
                    xn = x_func(t)
                    inv_norm = 1.0 / (1e-5 + np.sqrt((x1 - xn[0]) ** 2 + (x2 - xn[1]) ** 2))

                    x1_rhs = v[:self.N] - 1.0 / self.N * np.sum((1.0 - np.eye(x1d.shape[0])) * \
                        (x1d / (1e-5 + norm) * (self.Iatt(x1d, norm) * self.Fatt(thetad) - self.Irep(x1d, norm) * self.Frep(thetad))), axis=1)
                    x2_rhs = v[self.N:2*self.N] - 1.0 / self.N * np.sum((1.0 - np.eye(x2d.shape[0])) * \
                        (x2d / (1e-5 + norm) * (self.Iatt(x2d, norm) * self.Fatt(thetad) - self.Irep(x2d, norm) * self.Frep(thetad))), axis=1)
                    x3_rhs = v[2*self.N:] - 1.0 / self.N * np.sum((1.0 - np.eye(x3d.shape[0])) * \
                        (x3d / (1e-5 + norm) * (self.Iatt(x3d, norm) * self.Fatt(thetad) - self.Irep(x2d, norm) * self.Frep(thetad))), axis=1)

                    theta_rhs = omega + sigma * np.cos(s_func(t) - theta) * inv_norm - \
                        self.K / self.N * np.sum((1 - np.eye(theta.shape[0])) * (self.G(norm) * self.H(thetad)), axis=1)

                    dz = np.concatenate((x1_rhs, x2_rhs, x3_rhs, theta_rhs))
                    self.iterations += 1
                    return dz
            else:
                def dynamics(z, t):
                    x1 = z[:self.N]
                    x2 = z[self.N:2 * self.N]
                    x3 = z[2 * self.N:3 * self.N]
                    theta = z[3 * self.N:]

                    x1d = x1[:, np.newaxis] - x1
                    x2d = x2[:, np.newaxis] - x2
                    x3d = x3[:, np.newaxis] - x3

                    thetad = theta[:, np.newaxis] - theta

                    norm_sq = x1d ** 2 + x2d ** 2 + x3d ** 2
                    norm = np.sqrt(norm_sq)
                    #xn = x_func(t)
                    #inv_norm = 1.0 / (1e-5 + np.sqrt((x1 - xn[0]) ** 2 + (x2 - xn[1]) ** 2))

                    x1_rhs = v[:self.N] - 1.0 / self.N * np.sum((1.0 - np.eye(x1d.shape[0])) * \
                        (x1d / (1e-5 + norm) * (self.Iatt(x1d, norm) * self.Fatt(thetad) - self.Irep(x1d, norm) * self.Frep(thetad))), axis=1)
                    x2_rhs = v[self.N:2*self.N] - 1.0 / self.N * np.sum((1.0 - np.eye(x2d.shape[0])) * \
                        (x2d / (1e-5 + norm) * (self.Iatt(x2d, norm) * self.Fatt(thetad) - self.Irep(x2d, norm) * self.Frep(thetad))), axis=1)
                    x3_rhs = v[2*self.N:] - 1.0 / self.N * np.sum((1.0 - np.eye(x3d.shape[0])) * \
                        (x3d / (1e-5 + norm) * (self.Iatt(x3d, norm) * self.Fatt(thetad) - self.Irep(x2d, norm) * self.Frep(thetad))), axis=1)

                    theta_rhs = omega - self.K / self.N * np.sum((1 - np.eye(theta.shape[0])) * (self.G(norm) * self.H(thetad)), axis=1)

                    dz = np.concatenate((x1_rhs, x2_rhs, x3_rhs, theta_rhs))
                    self.iterations += 1
                    return dz


        #def event(t, y):
        #    return -1.0 + 1.0 * (self.iterations > max_iterations)

        #event.terminal = True

        return solve_ivp(lambda t, z: dynamics(z, t), tspan, z0, **kwargs)