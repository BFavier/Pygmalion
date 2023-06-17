import numpy as np
import pandas as pd
from typing import Callable


class OrbitalTrajectoryGenerator:

    def __init__(self, n_batches: int, batch_size: int):
        self.n_bacthes = n_batches
        self.batch_size = batch_size

    def __iter__(self):
        pass

    @staticmethod
    def generate_batch(batch_size: int, dt: float, n_steps: int, runge_kutta_4: bool=True):
        """
        Generates a single batch of trajectories
        """
        X = np.random.uniform(-1, 1, size=(batch_size, 2))
        theta = np.random.uniform(0, 2*np.pi, size=batch_size)
        rot = np.stack([np.cos(theta), -np.sin(theta)], axis=1)
        V = rot # np.random.uniform(-2**0.5, 2**0.5, size=(batch_size, 1)) * rot
        data = np.concatenate([X, V], axis=-1)
        t = 0
        df = pd.DataFrame(data=data, columns=["x", "y", "u", "v"], dtype=np.float64)
        df["t"] = t
        # f = OrbitalTrajectoryGenerator.runge_kutta_4 if runge_kutta_4 else OrbitalTrajectoryGenerator.euler_method
        f = OrbitalTrajectoryGenerator.runge_kutta_5
        dydt = OrbitalTrajectoryGenerator.derivatives
        for _ in range(n_steps-1):
            t += dt
            data = f(data, dydt, dt)
            sub = pd.DataFrame(data, columns=["x", "y", "u", "v"])
            sub["t"] = t
            df = pd.concat([df, sub])
        df.index.rename("obj", inplace=True)
        df.reset_index(inplace=True)
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
        GMm = 1.0
        eps = 1.0E-20
        ur = X/r
        return np.concatenate([V, GMm/(r**2 + eps) * -ur], axis=-1)
    
    @staticmethod
    def runge_kutta_5(data: np.ndarray, dydt: Callable, dt: float) -> np.ndarray:
        """
        Dormand-Prince method
        https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods
        """
        k1 = dydt(data)
        k2 = dydt(data + dt/5 * k1)
        k3 = dydt(data + dt*3/40 * k1 + dt*9/40 * k2)
        k4 = dydt(data + dt*44/45 * k1 + dt*-56/15 * k2 + dt*32/9 * k3)
        k5 = dydt(data + dt*19372/6561 * k1 + dt*-25360/2187 * k2 + dt*64448/6561 * k3 + dt*-212/729 * k4)
        k6 = dydt(data + dt*9017/3168 * k1 + dt*-355/33 * k2 + dt*46732/5247 * k3 + dt*49/176 * k4 + dt*-5103/18656 * k5)
        k7 = dydt(data + dt*35/384 * k1 + dt*500/1113 * k3 + dt*125/192 * k4 + dt*-2187/6784 * k5 + dt*11/84 * k6)
        return data + dt*5179/57600*k1 + dt*7571/16695*k3 + dt*393/640*k4 + dt*-92097/339200*k5 + dt*187/2100*k6 + dt*1/40*k7

    @staticmethod
    def runge_kutta_4(data: np.ndarray, dydt: Callable, dt: float) -> np.ndarray:
        """
        """
        k1 = dydt(data)
        k2 = dydt(data + dt/2 * k1)
        k3 = dydt(data + dt/2 * k2)
        k4 = dydt(data + dt * k3)
        return data + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    
    @staticmethod
    def euler_method(data: np.ndarray, dydt: Callable, dt: float) -> np.ndarray:
        """
        """
        return data + dt * dydt(data)