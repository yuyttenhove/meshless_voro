import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from pathlib import Path
import h5py


def read_faces(fname: Path) -> np.ndarray:
    data = h5py.File(fname, "r")
    start = data["Faces/Start"][:][:, :2]
    end = data["Faces/End"][:][:, :2]
    return np.stack([start, end], axis=1)


def plot(faces: np.ndarray):
    lines = LineCollection(faces, color="r")
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.add_collection(lines)
    ax.set_aspect("equal")
    ax.set_xlim(min([f[:, 0].min() for f in faces]), max([f[:, 0].max() for f in faces]))
    ax.set_ylim(min([f[:, 1].min() for f in faces]), max([f[:, 1].max() for f in faces]))
    ax.axis("off")
    fig.tight_layout()
    fig.savefig("test.png", dpi=300)


def main():
    faces = read_faces("test.hdf5")
    plot(faces)


if __name__ == "__main__":
    main()
