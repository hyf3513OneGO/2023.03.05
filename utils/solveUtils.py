import time

import geatpy as ea
import numpy as np

# decisionMat不是决策变量的矩阵而是子任务的矩阵
from utils.calcUtils import calcTasks
from utils.mockUtils import getRandomDAGMats, getRandomDataMat, getRandomDecisionMats, getRandomCarInfos


def mat2geatVar(decisionMat: np.array):
    geatVars = []
    noZeroPos = np.nonzero(decisionMat)
    lenNoneZero = len(noZeroPos[0])
    for item in range(lenNoneZero):
        var = decisionMat[noZeroPos[0][item]][noZeroPos[1][item]][noZeroPos[2][item]]
        geatVars.append(var)
    return geatVars, lenNoneZero


# decisionMat不是决策变量的矩阵而是子任务的矩阵
def geatVars2mat(geatVars: np.array, decisionMat: np.array):

    noZeroPos = np.nonzero(decisionMat)
    lenNoneZero = len(noZeroPos[0])
    finalMat = np.zeros_like(decisionMat)
    if len(geatVars) != lenNoneZero:
        print("err:len(geatVars)!=lenNoneZero")
        return finalMat
    for idx in range(lenNoneZero):
        finalMat[noZeroPos[0][idx]][noZeroPos[1][idx]][noZeroPos[2][idx]] = geatVars[idx]

    return finalMat, lenNoneZero


# 构建问题




def main():

    pass


if __name__ == "__main__":
    main()
