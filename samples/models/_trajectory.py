from pygmalion.datasets.generators import OrbitalTrajectoryGenerator
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FuncAnimation
import time

N_OBJECTS = 100
t0 = time.perf_counter()
df = OrbitalTrajectoryGenerator.generate_batch(batch_size=N_OBJECTS, dt_min=0., tol=1.0E-6)
t1 = time.perf_counter()
print(f"Execution time: {t1-t0:.3g} seconds")

def plot_trajectory(df: pd.DataFrame):
    f, ax = plt.subplots()
    ax.set_xlim([-3, 3])
    ax.set_ylim([-3, 3])
    ax.set_aspect("equal")
    lines = []

    def update(df: pd.DataFrame, lines: list, frame: int):
        for line, (_, sub) in zip(lines, df.groupby("obj")):
            sub = sub.iloc[:frame]
            line.set_data(sub["x"].tolist()[-20:], sub["y"].tolist()[-20:])
        return lines

    def init(lines: list, ax: plt.Axes):
        lines.clear()
        for _ in range(N_OBJECTS):
            lines.append(ax.plot([], [])[0])

    anim = FuncAnimation(f, lambda x: update(df, lines, x), frames=range(1, len(df)+1), init_func=lambda : init(lines, ax), blit=False, interval=5/len(df)*1.0E3)
    plt.show()


plot_trajectory(df)
if __name__ == "__main__":
    import IPython
    IPython.embed()