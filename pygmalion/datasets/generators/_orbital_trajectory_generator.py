import numpy as np
import pandas as pd


class OrbitalTrajectoryGenerator:
    """
    Generator of eliptic orbital trajectories
    """

    def __init__(self, n_batches: int, batch_size: int, n_steps: int=501):
        self.n_batches = n_batches
        self.batch_size = batch_size
        self.n_steps = n_steps

    def __iter__(self):
        for _ in range(self.n_batches):
            yield self.generate_batch(self.batch_size, self.n_steps)

    @staticmethod
    def generate_batch(batch_size: int, n_steps: int=501):
        """
        Generates a single batch of trajectories
        """
        a = np.ones(batch_size)
        e = 1 - 2**(np.random.uniform(-4, 0, batch_size))
        # e = np.random.uniform(0.7, 1.0, batch_size)
        b = np.sqrt(a**2 * (1 - e**2))
        t0 = np.random.uniform(0., 1., batch_size)
        t = np.linspace(t0, t0+np.random.uniform(0.3, 2.0, batch_size), n_steps).transpose()
        p = OrbitalTrajectoryGenerator.xy(t, a, b)
        theta = np.random.uniform(-np.pi, np.pi, batch_size)
        rot = np.stack([np.stack([np.cos(theta), -np.sin(theta)], axis=-1),
                        np.stack([np.sin(theta), np.cos(theta)], axis=-1)], axis=-2)
        p = p @ rot
        df = pd.DataFrame(data=p.reshape(-1, 2), columns=["x", "y"])
        df["t"] = t.reshape(-1)
        df["ID"] = np.tile(np.arange(batch_size).reshape(batch_size, 1), (1, n_steps)).reshape(-1)
        return df

    @staticmethod
    def iterative_solver(E: np.ndarray, e: np.ndarray, M: np.ndarray, n_max_steps: int=100, tol: float=1.0E-4):
        """
        Solves iteratively 'E - e*sin(E) = M' using newton raphson
        """
        # E = M + e*np.sin(E)
        error = np.abs(E - e*np.sin(e) - M)
        if (np.max(error) < tol) or (n_max_steps < 1):
            return E
        return OrbitalTrajectoryGenerator.iterative_solver(E - (E - e*np.sin(E) - M)/(1 - e*np.cos(E)), e, M, n_max_steps-1, tol)

    @staticmethod
    def initialize(M: np.ndarray, e: np.ndarray):
        """
        Initialization point for the iterative solver
        """
        n = np.sqrt(5 + np.sqrt(16 + 9/e))
        a = n*(e*(n**2 - 1)+1)/6
        c = n*(1-e)
        d = -M
        assert np.all((a > 0) & (c > 0))
        p = c/a
        q = d/a
        k = np.sqrt(q**2/4 + p**3/27)
        s = np.cbrt(-q/2 - k) + np.cbrt(-q/2 + k)
        return n*np.arcsin(s)  

    @staticmethod
    def E(t: np.ndarray, e: np.ndarray) -> np.ndarray:
        """
        Returns the eccentricity anomaly E for given times using inverse Kepler's equation
        (assuming a period of T=1s, and t=0s corresponding to a time of periapsis)
        https://en.wikipedia.org/wiki/Kepler%27s_equation

        This returns the E such that E - e*sin(E) = M, with M = 2π*t/T

        Parameters
        ----------
        t : np.ndarray
            array of times of shape (n_objects, n_steps)
        e : np.ndarray
            array of trajectory eccentricities of shape (n_objects,)
            of values such that 0 < e < 1
        """
        t = (((t + 0.5) % 1.0) - 0.5)  # the trajectory is 1-periodic, and t must be in [-0.5, 0.5] for algorithm convergence
        M = 2*np.pi * t
        e = e.reshape(-1, 1)
        E = np.sign(M) * OrbitalTrajectoryGenerator.initialize(np.abs(M), e)
        return OrbitalTrajectoryGenerator.iterative_solver(E, e, M)

    @staticmethod
    def xy(t: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """

        Parameters
        ----------
        t : np.ndarray
            array of times of shape (n_objects, n_steps)
        a : np.ndarray
            array of major axis lengths, of shape (n_objects)
        b : np.ndarray
            array of minor axis lengths, of shape (n_objects)
        """
        a, b = np.maximum(a, b).reshape(-1, 1), np.minimum(a, b).reshape(-1, 1)
        e = np.sqrt(1 - b**2/a**2)
        E = OrbitalTrajectoryGenerator.E(t, e)
        x = a*(np.cos(E) - e)  # instead of a*(np.cos(E) - e) because in the referential of the foyer F and not of the origin F + a*e*ux
        y = b*np.sin(E)
        return np.stack([x, y], axis=-1)
