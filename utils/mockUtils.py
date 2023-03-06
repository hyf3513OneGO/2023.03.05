import math
import random
import time

import numpy as np
from random import shuffle as sl
from random import randint as rd

# node节点数量，edge边数量
from numpy import ndarray


def getRandomDAGMat(node: int, edge: int) -> np.array:
    n = node
    node = range(0, n)
    node = list(node)

    # sl(node)  # 生成拓扑排序
    m = edge
    result = []  # 存储生成的边，边用tuple的形式存储

    appeared_node = []
    not_appeared_node = node
    # 生成前n - 1条边
    while len(result) != n - 1:
        # 生成第一条边
        if len(result) == 0:
            p1 = rd(0, n - 2)
            p2 = rd(p1 + 1, n - 1)
            x = node[p1]
            y = node[p2]
            appeared_node.append(x)
            appeared_node.append(y)
            not_appeared_node = list(set(node).difference(set(appeared_node)))
            result.append((x, y))
        # 生成后面的边
        else:
            p1 = rd(0, len(appeared_node) - 1)
            x = appeared_node[p1]  # 第一个点从已经出现的点中选择
            p2 = rd(0, len(not_appeared_node) - 1)
            y = not_appeared_node[p2]
            appeared_node.append(y)  # 第二个点从没有出现的点中选择
            not_appeared_node = list(set(node).difference(set(appeared_node)))
            # 必须保证第一个点的排序在第二个点之前
            if node.index(y) < node.index(x):
                result.append((y, x))
            else:
                result.append((x, y))
    # 生成后m - n + 1条边
    while len(result) != m:
        p1 = rd(0, n - 2)
        p2 = rd(p1 + 1, n - 1)
        x = node[p1]
        y = node[p2]
        # 如果该条边已经生成过，则重新生成
        if (x, y) in result:
            continue
        else:
            result.append((x, y))

    matrix = np.zeros((n, n))
    for i in range(len(result)):
        random.seed(time.time() + random.randint(2, 100))
        matrix[result[i][0], result[i][1]] = random.randint(15, 40)

    return matrix


def getRandomDataMat(timeTasksMat: np.array, low: int, high: int) -> np.array:
    datasMat = timeTasksMat.copy()
    datasMat[datasMat != 0] = random.randint(low, high)
    return datasMat


def getRandomDAGMats(n: int, low: int, high: int) -> np.array:
    matList = []
    for i in range(n):
        matList.append(getRandomDAGMat(low, high))
    finalMat = np.stack(matList)
    return finalMat


def getRandomDecisionMats(inputMat: np.array, isContinuous: bool, itemRange: [0, 1]) -> np.array:
    inputMat[inputMat != 0] = 1
    if isContinuous:
        tempMat = np.round(np.random.uniform(itemRange[0], itemRange[1], (len(inputMat), 6, 6)), 2)
        return tempMat * inputMat
    return np.random.randint(itemRange[0], itemRange[1], (len(inputMat), 6, 6)) * inputMat


def getRandomCarInfos(n: int) -> np.array:
    matList = []
    for i in range(n):
        matList.append(np.array([random.randint(8, 16),
                                 round(random.uniform(0.01, 0.015), 2),
                                 round(random.randint(50, 200)),
                                 round(random.uniform(30, 90)),
                                 round(random.uniform(110, 160))
                                 ]))
    finalMat = np.stack(matList)
    return finalMat
