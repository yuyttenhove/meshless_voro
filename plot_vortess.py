import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from pathlib import Path


def read_faces(fname: Path) -> np.ndarray:
    data = np.fromregex(
        file = fname, 
        regexp = "(\S+)\s+\((\S+), (\S+), (\S+)\)\s+\((\S+), (\S+), (\S+)\)", 
        dtype = [
            ("area", np.float64), 
            ("mx", np.float64), 
            ("my", np.float64), 
            ("mz", np.float64), 
            ("nx", np.float64), 
            ("ny", np.float64), 
            ("nz", np.float64),
        ])
    faces = []
    for row in data:
        normal = np.array([row["nx"], row["ny"], row["nz"]])
        normal /= np.sqrt(normal.T.dot(normal))
        ort = np.array([normal[1], -normal[0], 0])
        midpoint = np.array([row["mx"], row["my"], row["mz"]])
        length = row["area"]
        a = midpoint + 0.5 * length * ort
        b = midpoint - 0.5 * length * ort
        faces.append(np.stack([a, b])[:, :2])
    return faces


def plot(faces: np.ndarray):
    lines = LineCollection(faces, color="r")
    fig, ax = plt.subplots()
    ax.add_collection(lines)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig("test.png")


def main():
    faces = read_faces("faces.txt")
    plot(faces)


if __name__ == "__main__":
    main()
