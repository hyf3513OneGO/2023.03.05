import os.path

import numpy as np


def loadDataset(datasetDir: str):
    car = np.load(os.path.join(datasetDir, "car.npy"))
    resource = np.load(os.path.join(datasetDir, "resource.npy"))
    data = np.load(os.path.join(datasetDir, "data.npy"))
    decision = np.load(os.path.join(datasetDir, "decision.npy"))
    time = np.load(os.path.join(datasetDir, "time.npy"))
    return time, data, decision, resource, car


if __name__ == "__main__":
    loadDataset("../datasets/2023.03.07")
