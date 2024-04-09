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


def read_generators(fname: Path) -> np.ndarray:
    data = h5py.File(fname, "r")
    return data["Cells/Generator"][:]


def read_centroids(fname: Path) -> np.ndarray:
    data = h5py.File(fname, "r")
    return data["Cells/Centroid"][:]


def plot(faces: np.ndarray, generators=None, centroids=None):
    lines = LineCollection(faces, color="r", lw=1)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.add_collection(lines)
    ax.set_aspect("equal")
    ax.set_xlim(
        min([f[:, 0].min() for f in faces]), max([f[:, 0].max() for f in faces])
    )
    ax.set_ylim(
        min([f[:, 1].min() for f in faces]), max([f[:, 1].max() for f in faces])
    )
    if generators is not None:
        ax.scatter(generators[:, 0], generators[:, 1], c="green", s=0.5)
    if centroids is not None:
        ax.scatter(centroids[:, 0], centroids[:, 1], c="blue", s=0.5)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig("test.png", dpi=300)


def main(fname):
    plot(
        faces=read_faces(fname),
        generators=read_generators(fname),
        centroids=read_centroids(fname),
    )


if __name__ == "__main__":
    main("test_2_d.hdf5")
