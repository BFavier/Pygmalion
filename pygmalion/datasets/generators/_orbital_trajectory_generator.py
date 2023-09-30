import numpy as np
import pandas as pd


class OrbitalTrajectoryGenerator:

    def __init__(self, n_batches: int, batch_size: int, T: np.ndarray=np.linspace(0.0, 5.0, 501),
                 dt_min: float=0., tol: float=1.0E-6, verbose: bool=False):
        """
        Parameters
        ----------
        n_batches : int
            number of batches to yield in an interation
        batch_size : int
            number of objects orbital trajectory to generate each batch
        T : np.ndarray
            array of interpolation times of shape (n_times,)
        dt_min : float
            minimum time step during integration
        tol : float
            error tolerance per second of integration
        verbose : bool
            if True display integration progress
        """
        self.n_batches = n_batches
        self.batch_size = batch_size
        self.T = T
        self.dt_min = dt_min
        self.tol = tol
        self.verbose = verbose

    def __iter__(self):
        for _ in range(self.n_batches):
            yield self.generate_batch(self.batch_size, self.T, self.dt_min, self.tol, self.verbose)

    @staticmethod
    def generate_batch(batch_size: int, T: np.ndarray=np.linspace(0.0, 5.0, 501),
                       dt_min: float=0., tol=1.0E-6, verbose: bool=False):
        """
        Generates a single batch of trajectories
        """
        r = np.random.uniform(0.5, 1.5, size=(batch_size, 1))  # Do not generate points too close to the center
        phi = np.random.uniform(0, 2*np.pi, size=batch_size)
        X = r * np.stack([np.cos(phi), -np.sin(phi)], axis=-1)
        GM = 1.0
        escape_velocity = (2*GM/r)**0.5
        theta = np.random.uniform(0, 2*np.pi, size=batch_size)
        rot = np.stack([np.cos(theta), -np.sin(theta)], axis=1)
        V = np.maximum(1.0, np.random.normal(0.4, 1.2, size=(batch_size, 1)) * escape_velocity) * rot
        y0 = np.concatenate([X, V], axis=-1)
        df = OrbitalTrajectoryGenerator.runge_kutta_fehlberg(y0, T, dt_min, tol, verbose)
        df["obj"] = df["obj"].astype(int)
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
        ur = X/r
        return np.concatenate([V, GM/(r**2) * -ur], axis=-1)
    
    @staticmethod
    def runge_kutta_fehlberg(y0: np.ndarray, T: np.ndarray, dt_min: float, tol: float, verbose: bool) -> pd.DataFrame:
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
        T : np.ndarray
            array of interpolation times of shape (n_times,)
        dt_min : float
            minimum time step
        tol : float
            maximum tolerated order of magnitude of error per second of integration

        Returns
        -------
        pd.DataFrame :
            dataframe containing the (x, y, u, v) at each time 't', for each object 'obj'
        """
        n_obj, _ = y0.shape
        dydt = OrbitalTrajectoryGenerator.derivatives
        y, t, h = y0, T[0], (T[1] - T[0])
        obj = np.arange(n_obj)[:, None]
        data = [np.concatenate([obj, y0], axis=1)]
        for t_next in T[1:]:
            while t < t_next:
                k1 = dydt(y)
                k2 = dydt(y + h/4 * k1)
                k3 = dydt(y + h*3/32 * k1 + h*9/32 * k2)
                k4 = dydt(y + h*1932/2197 * k1 + h*-7200/2197 * k2 + h*7296/2197 * k3)
                k5 = dydt(y + h*439/216 * k1 + h*-8 * k2 + h*3680/513 * k3 + h*-845/4104 * k4)
                k6 = dydt(y + h*-8/27 * k1 + h*2 * k2 + h*-3544/2565 * k3 + h*1859/4104 * k4 + h*-11/40 * k5)
                RK5 = (16/135*k1 + 6656/12825*k3 + 28561/56430*k4 - 9/50*k5 + 2/55*k6)
                RK4 = (25/216*k1 + 1408/2565*k3 + 2197/4104*k4 - 1/5*k5)
                error = np.max(np.abs(RK4 - RK5))
                factor = (tol / (2*error))**(1/5)
                h = h * factor
                h = min(max(dt_min, h), t_next-t)
                if verbose:
                    print(f"t={t:.3g}, Δt={h:.3g}, error={error:.3g}, factor={factor:.3g}", flush=True)
                t += h
                y = y + h*RK5
            data.append(np.concatenate([obj, y], axis=1))
        data = np.concatenate([np.stack(data, axis=1), np.repeat(T[None, :, None], n_obj, axis=0)], axis=-1)
        df = pd.DataFrame(data=data.reshape(-1, 6), columns=["obj", "x", "y", "u", "v", "t"])
        return df
