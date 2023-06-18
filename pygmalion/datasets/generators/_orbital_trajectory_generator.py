import numpy as np
import pandas as pd
from typing import Callable


class OrbitalTrajectoryGenerator:

    def __init__(self, n_batches: int, batch_size: int, T: float=5.0,
                 dt: float=1.0E-2, dt_min: float=1.0E-6, tol: float=1.0E-6, verbose: bool=False):
        """
        Parameters
        ----------
        n_batches : int
            number of batches to yield in an interation
        batch_size : int
            number of objects orbital trajectory to generate each batch
        T : float
            time duration of the generated orbits
        dt : float
            time steps of the trajectory snapshots
        dt_min : float
            minimum time step during integration
        tol : float
            tolerance per second of integration of velocity along x and y
        verbose : bool
            if True display integration progress
        """
        self.n_batches = n_batches
        self.batch_size = batch_size
        self.T = T
        self.dt = dt
        self.dt_min = dt_min
        self.tol = tol
        self.verbose = verbose

    def __iter__(self):
        for _ in self.n_batches:
            yield self.generate_batch(self.batch_size, self.T, self.dt, self.dt_min, self.tol, self.verbose)

    @staticmethod
    def generate_batch(batch_size: int, T: float=5.0, dt: float=1.0E-2,
                       dt_min: float=1.0E-6, tol=1.0E-6, verbose: bool=False):
        """
        Generates a single batch of trajectories
        """
        X = np.random.uniform(-1, 1, size=(batch_size, 2))
        theta = np.random.uniform(0, 2*np.pi, size=batch_size)
        rot = np.stack([np.cos(theta), -np.sin(theta)], axis=1)
        r = np.linalg.norm(X, axis=-1)[:, None]
        GM = 1.0
        escape_velocity = (GM/r)**0.5
        V = np.random.uniform(0.5, 1.2, size=(batch_size, 1)) * escape_velocity * rot
        y0 = np.concatenate([X, V], axis=-1)
        df = OrbitalTrajectoryGenerator.runge_kutta_fehlberg(y0, T, dt, dt_min, tol, verbose)
        df.loc[(df["x"] < -3) | (df["x"] > 3) | (df["y"] < -3) | (df["y"] > 3),
               ["x", "y", "u", "v"]] = float("nan")
        return df

    @staticmethod
    def derivatives(data: np.ndarray) -> np.ndarray:
        """
        returns the derivative of (x, y, u, v) with regards to time

        Parameters
        ----------
        data : np.ndarray
            array of shape (batch_size, 4)
        
        Returns
        -------
        np.ndarray :
            array of shape (batch_size, 4) of derivatives of (x, y, u, v) wrt time
        """
        X, V = data[:, :2], data[:, 2:]
        r = np.linalg.norm(X, ord=2, axis=1)[:, None]
        GM = 1.0
        eps = 1.0E-20
        ur = X/r
        return np.concatenate([V, GM/(r**2 + eps) * -ur], axis=-1)
    
    @staticmethod
    def runge_kutta_fehlberg(y0: np.ndarray, T: float, dt: float, dt_min: float, tol: float, verbose: bool) -> pd.DataFrame:
        """
        Perform dynamic time step integration of the problem using runge kutta fehlberg method.
        This is to avoid adding scipy.integrate.ode as a dependency.

        https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods
        https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta%E2%80%93Fehlberg_method
        https://maths.cnam.fr/IMG/pdf/RungeKuttaFehlbergProof.pdf

        Parameters
        ----------
        y0 : np.ndarray
            initial conditions of (x, y, u, v)
            array of floats of shape (n_objects, 4)
        T : float
            integration time
        dt : float
            maximum time step, and final interpolated time steps
        dt_min : float
            minimum time step
        tol : float
            maximum tolerated error accumulation per second of integration, on u and v
        
        Returns
        -------
        pd.DataFrame :
            dataframe containing the (x, y, u, v) at each time 't', for each object 'obj'
        """
        dydt = OrbitalTrajectoryGenerator.derivatives
        y, t, h = y0, 0., dt
        parameters = [y]
        times = [t]
        while t < T:
            k1 = dydt(y)
            k2 = dydt(y + h/4 * k1)
            k3 = dydt(y + h*3/32 * k1 + h*9/32 * k2)
            k4 = dydt(y + h*1932/2197 * k1 + h*-7200/2197 * k2 + h*7296/2197 * k3)
            k5 = dydt(y + h*439/216 * k1 + h*-8 * k2 + h*3680/513 * k3 + h*-845/4104 * k4)
            k6 = dydt(y + h*-8/27 * k1 + h*2 * k2 + h*-3544/2565 * k3 + h*1859/4104 * k4 + h*-11/40 * k5)
            RK5 = (16/135*k1 + 6656/12825*k3 + 28561/56430*k4 - 9/50*k5 + 2/55*k6)
            RK4 = (25/216*k1 + 1408/2565*k3 + 2197/4104*k4 - 1/5*k5)
            error = np.max(np.abs(RK4[:, 2:] - RK5[:, 2:]))
            h = h * (tol / (2*error))**(1/5)
            if error > tol or h == dt_min:
                continue
            elif verbose:
                print(f"t={t:.3g}, dt={h:.3g}, error={error:.3g}")
            h = max(dt_min, min(h, dt))
            t += h
            y = y + h*RK5
            parameters.append(y)
            times.append(t)
        times = np.array(times)
        interp_times = np.arange(0, T+dt, dt)
        interp = [np.stack([np.interp(interp_times, times, p) for p in obj.T], axis=1)
                  for obj in np.stack(parameters, axis=1)]
        objects = np.concatenate([np.full((len(_),), i) for i, _ in enumerate(interp)])
        interp_times = np.concatenate([interp_times]*len(interp))
        interp = np.concatenate(interp, axis=0)
        data = np.concatenate([interp, interp_times[:, None], objects[:, None]], axis=1)
        return pd.DataFrame(data=data, columns=["x", "y", "u", "v", "t", "obj"])
