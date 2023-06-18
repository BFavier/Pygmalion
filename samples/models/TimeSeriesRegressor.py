from pygmalion.datasets.generators import OrbitalTrajectoryGenerator
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

N_OBJECTS = 100
df = OrbitalTrajectoryGenerator.generate_batch(batch_size=N_OBJECTS, tol=1.0E-6)
f, ax = plt.subplots()
ax.set_xlim([-3, 3])
ax.set_ylim([-3, 3])
ax.set_aspect("equal")
lines = [ax.plot([], [])[0] for _ in range(N_OBJECTS)]

def update(frame: int):
    for line, (_, sub) in zip(lines, df.groupby("obj")):
        sub = sub.iloc[:frame]
        line.set_data(sub["x"].tolist()[-20:], sub["y"].tolist()[-20:])
    return lines

ani = FuncAnimation(f, update, frames=range(1, len(df)+1), blit=False, interval=1/len(df)*1.0E3)
plt.show()

if __name__ == "__main__":
    import IPython
    IPython.embed()