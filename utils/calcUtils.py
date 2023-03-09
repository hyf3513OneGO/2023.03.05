import math

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from typing import List, Tuple, Any

from utils.graphUtils import getEST, showGraph, mat2Graph, getESTGraph


def getCPUResourceMat(decisionMat: np.array, resourceMat: np.array, taskMat: np.array,
                      localResource: float) -> np.array:
    finalMat = decisionMat * resourceMat
    finalMat[finalMat == 0] = localResource
    taskMat[taskMat != 0] = 1
    finalMat = finalMat * taskMat
    return finalMat


def getDoneTimeMat(decisionMat: np.array, taskMat: np.array, resourceMat: np.array, dataMat: np.array,
                   realBandwidthMat: np.array) -> np.array:
    resourceMat[resourceMat == 0] = 1
    realBandwidthMat[realBandwidthMat == 0] = 1
    transferTimeMat = dataMat / realBandwidthMat
    transferTimeMat = transferTimeMat * decisionMat
    doneTimeMat = np.round((taskMat / resourceMat) + transferTimeMat, 2)
    return np.squeeze(doneTimeMat)


def getTransferTimeMat(decisionMat: np.array, taskMat: np.array, resourceMat: np.array, dataMat: np.array,
                       realBandwidthMat: np.array) -> np.array:
    resourceMat[resourceMat == 0] = 1
    realBandwidthMat[realBandwidthMat == 0] = 1
    transferTimeMat = dataMat / realBandwidthMat
    transferTimeMat = transferTimeMat * decisionMat
    return transferTimeMat


def getTasksESTTime(tasksGraph: list) -> tuple[int, list[float]]:
    ESTS = 0
    ESTList = []
    for taskGraph in tasksGraph:
        EST = getEST(taskGraph)
        ESTS += EST
        ESTList.append(EST)
    return ESTS, ESTList


def getRealBandwidthMat(decisionMat: np.array, taskMat: np.array, carInfo: np.array, MecInfo: np.array):
    taskMat[decisionMat != 0] = MecInfo[2] * math.log10(1 + carInfo[4])
    return taskMat


def getTaskEnergy(decisionMat: np.array, taskMat: np.array, resourceMat: np.array, carInfo: np.array,
                  MECInfo: np.array) -> float:
    carInfo = np.squeeze(carInfo)
    MEC_K = np.ones_like(decisionMat)
    MEC_K[MEC_K == 1] = MECInfo[1]
    all_K = MEC_K * decisionMat
    all_K[all_K == 0] = carInfo[1]
    tasksMat_copy = taskMat
    tasksMat_copy[tasksMat_copy != 0] = 1
    all_K = all_K * tasksMat_copy
    all_energy = all_K * resourceMat ** 2 * taskMat
    # print("all_K")
    # print(all_K)
    # print("taskMat")
    # print(taskMat)
    # print("all_energy")
    # print(all_energy)
    # print("------------------")
    return all_energy.sum()


def calcTask(taskTimeMat: np.array, taskDataMat: np.array, decisionMat: np.array, resourceMat: np.array,
             carInfo: np.array, MECInfo: np.array, B: np.array):
    # 获取算力分配
    cpuResourceMat = getCPUResourceMat(decisionMat.copy(), resourceMat.copy(),
                                       taskTimeMat.copy(), carInfo[0].copy())
    # 获取真实带宽分配
    realBandwidthMat = getRealBandwidthMat(decisionMat.copy(), taskTimeMat.copy(),
                                           carInfo.copy(),
                                           MECInfo.copy())
    # 计算单个任务时间（生成EST图，计算EST）
    doneTimeMat = getDoneTimeMat(decisionMat.copy(), taskTimeMat.copy(), cpuResourceMat.copy(), taskDataMat,
                                 realBandwidthMat)
    graphDone = mat2Graph(doneTimeMat.copy())
    graphDoneEST = getESTGraph(graphDone)
    # 计算单个任务传输时间
    transferTimeMat = getTransferTimeMat(decisionMat.copy(), taskTimeMat.copy(), resourceMat.copy(), taskDataMat.copy(),
                                         realBandwidthMat)
    graphTransfer = mat2Graph(transferTimeMat.copy())
    graphTransferEST = getESTGraph(graphTransfer)
    # 计算单个任务能耗
    taskEnergy = getTaskEnergy(decisionMat.copy(), taskTimeMat.copy(), cpuResourceMat.copy(),
                               carInfo.copy(), MECInfo.copy())
    taskTime, taskTimeList = getTasksESTTime([graphDoneEST])
    transferTime, transferTimeList = getTasksESTTime([graphTransferEST])
    # 计算原始耗时与耗能
    taskOriginTime, taskOriginTimeList, taskOriginEnergy = calcTaskOrigin(taskTimeMat, carInfo)
    # 计算QOE
    # 计算体验指标
    QOE = calcQOE(taskTime, taskOriginTime, taskEnergy, taskOriginEnergy, B)
    # print(f"taskTime:{taskTime},taskEnergy:{taskEnergy}")
    # print(f"transferTime:{transferTime}")
    return taskTime, transferTime, taskEnergy, taskTimeList, transferTimeList, taskOriginTime, taskOriginTimeList, taskOriginEnergy, QOE


def calcTaskOrigin(taskTimeMat: np.array, carInfo: np.array):
    resourceMat = np.ones_like(taskTimeMat)
    resourceMat[resourceMat == 1] = carInfo[0] / np.count_nonzero(taskTimeMat)
    # 计算原有耗时
    timeMat = np.round(taskTimeMat / resourceMat, 2)
    timeGraph = mat2Graph(timeMat)
    timeESTGraph = getESTGraph(timeGraph)
    taskTime, taskTimeList = getTasksESTTime([timeESTGraph])
    # 计算原有耗能
    all_K = np.ones_like(taskTimeMat)
    all_K[all_K == 1] = carInfo[1]
    all_energy = all_K * resourceMat ** 2 * taskTimeMat
    all_energy = all_energy.sum()

    return taskTime, taskTimeList, all_energy
    # print(f"origin:")
    # print(f"taskTime：{taskTime},taskEnergy:{all_energy}")


def calcQOE(taskTime: float, taskOriginTime: float, taskEnergy: float, taskOriginEnergy: float, BMat: np.array):

    if taskOriginTime - taskTime>0:
        return BMat[0] * ((taskOriginTime - taskTime) / taskOriginTime) + BMat[1] * (
            (taskOriginEnergy - taskEnergy) / taskOriginEnergy)
    else:
        # print("in")
        return -10000000


def calcTasks(taskTimeMats: np.array, tasksDataMats: np.array, decisionMats: np.array, resourceMats: np.array,
              carInfos: np.array, MECInfo: np.array, B: np.array):
    taskTimeTotal = 0
    transferTimeTotal = 0
    taskEnergyTotal = 0
    taskTimeOriginTotal = 0
    taskEnergyOriginTotal = 0
    QOETotal = 0
    taskTimeList = []
    taskOriginTimeList = []
    taskEnergyList = []
    taskOriginEnergyList = []
    QOEList = []
    for idx, item in enumerate(taskTimeMats):
        taskTime, transferTime, taskEnergy, taskTimeList, _, taskOriginTime, _, taskOriginEnergy, QOE = calcTask(
            taskTimeMats[idx].copy(),
            tasksDataMats[idx],
            decisionMats[idx].copy(),
            resourceMats[idx].copy(),
            carInfos[idx].copy(),
            MECInfo.copy(), B[idx])
        taskTimeTotal += taskTime
        taskEnergyTotal += taskEnergy
        transferTimeTotal += transferTime
        taskTimeOriginTotal += taskOriginTime
        taskEnergyOriginTotal += taskOriginEnergy
        QOETotal += QOE
        taskTimeList.append(taskTime)
        taskEnergyList.append(taskEnergy)
        taskOriginTimeList.append(taskOriginTime)
        taskOriginEnergyList.append(taskOriginEnergy)
        QOEList.append(QOE)
    taskTimeTotal=round(taskTimeTotal,2)
    transferTimeTotal = round(transferTimeTotal, 2)
    taskEnergyTotal = round(taskEnergyTotal, 2)
    taskTimeOriginTotal = round(taskTimeOriginTotal, 2)
    taskEnergyOriginTotal = round(taskEnergyOriginTotal, 2)
    # print(taskTimeOriginTotal,taskEnergyOriginTotal)

    # print(
    #     f"taskTimeTotal:{taskTimeTotal},taskEnergyTotal:{taskEnergyTotal},transferTimeTotal:{transferTimeTotal},"
    #     f"taskOriginTimeTotal:{taskTimeOriginTotal},taskOriginEnergyTotal:{taskEnergyOriginTotal},QOETotal:{QOETotal}")
    return taskTimeTotal, taskEnergyTotal, transferTimeTotal, QOETotal,taskTimeOriginTotal,taskEnergyOriginTotal
    pass
