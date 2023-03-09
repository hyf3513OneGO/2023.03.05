import random

from utils.calcUtils import calcTask, calcTasks


def roulette(objList: []):
    sumTotal = sum(objList)
    probList = [i / sumTotal for i in objList]
    probAddList = []
    for idx in range(len(probList)):
        if idx != 0:
            probAddList.append(probList[idx] + probAddList[idx - 1])
        else:
            probAddList.append(probList[idx])
    randNum = random.random()

    for idx in range(len(probAddList) - 1):
        if randNum < probAddList[0]:
            return 0
        if probAddList[idx] <= randNum < probAddList[idx + 1]:
            return idx + 1


def evaluateDecisionPool(decisionPool: [], timeMat, dataMat, carInfo, MECInfo, B, isMax: True, rePlaceType: str):
    evaluateList = []
    idx = 0
    for i, val in enumerate(decisionPool):
        taskTimeTotal, taskEnergyTotal, transferTimeTotal, QOETotal, taskTimeOriginTotal, taskEnergyOriginTotal = calcTasks(
            timeMat, dataMat, val[0], val[1], carInfo, MECInfo, B)
        if rePlaceType == "time":
            evaluateList.append(taskTimeTotal)
        if rePlaceType == "energy":
            evaluateList.append(taskEnergyTotal)
        if rePlaceType == "transfer":
            evaluateList.append(transferTimeTotal)
        if rePlaceType == "QOE":
            evaluateList.append(QOETotal)
    if isMax:  # 寻找并返回最大值下标，从而进行替换优化
        idx = evaluateList.index(max(evaluateList))
    else:
        idx = evaluateList.index(min(evaluateList))
    return idx


def calcFitList(decisionPool, timeMat, dataMat, carInfo, MECInfo, B, fitType: str):
    objList = []
    for item in decisionPool:
        taskTimeTotal1, taskEnergyTotal1, transferTimeTotal1, QOETotal1, taskTimeOriginTotal1, taskEnergyOriginTotal1 = calcTasks(
            timeMat.copy(), dataMat.copy(), item[0].squeeze().squeeze(), item[1].squeeze().squeeze(), carInfo.copy(),
            MECInfo.copy(), B)
        if fitType == "time":
            objList.append(taskTimeTotal1)
        if fitType == "energy":
            objList.append(taskEnergyTotal1)
        if fitType == "transfer":
            objList.append(transferTimeTotal1)
        if fitType == "QOE":
            objList.append(QOETotal1)
    return objList


if __name__ == "__main__":
    selectList = [1, 4, 2, 3]
    replaceID = evaluateDecisionPool(decisionTimePool, timeMat, dataMat, carMat, MECInfo, B, False)
