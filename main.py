import math

import matplotlib.pyplot as plt
import numpy as np

from utils.calcUtils import calcTasks
from utils.mockUtils import getRandomDAGMats, getRandomDecisionMats, getRandomCarInfos, \
    getRandomDataMat
from utils.resolvers.decisions.ga import gaResolve_class_decision
from utils.resolvers.resources.ga import gaResolve_class_resource
from utils.resolvers.resources.pso import psoResolve_class_resource

'''
DAG最大节点数：6
单任务矩阵6x6
最大车辆数10
'''
MECInfo = [196, 0.2, 10]  # f,k,B
MECInfo = np.array(MECInfo)

n = 10
B = [0.8, 0.2] * n
B = np.array(B)
B = B.reshape(n, 2)

randTasksTimeMats = getRandomDAGMats(n, 6, 10)
randTasksDataMats = getRandomDataMat(randTasksTimeMats, 5, 8)
randomDecision = getRandomDecisionMats(randTasksTimeMats.copy(), False, [0, 2])
randomResource = getRandomDecisionMats(randTasksTimeMats.copy(), True, [1, 10]) * randomDecision
randomCarInfos = getRandomCarInfos(n)

MECInfo = [196, 0.02, 10]  # f,k,B
MECInfo = np.array(MECInfo)

n = 10
B = [0.9, 0.1] * n
B = np.array(B)
B = B.reshape(n, 2)
ITER = 50


def main():
    ESTGraphs = []
    # taskTimeTotal, taskEnergyTotal, transferTimeTotal, QOETotal=calcTasks(randTasksTimeMats, randTasksDataMats, randomDecision, randomResource, randomCarInfos, MECInfo, B)
    # print(f"QOE-random:{QOETotal}")

    taskTimeTotal, taskEnergyTotal, transferTimeTotal, QOETotal, taskTimeOriginTotal, taskEnergyOriginTotal = calcTasks(
        randTasksTimeMats.copy(),
        randTasksDataMats.copy(),
        randomDecision.copy(),
        randomResource.copy(),
        randomCarInfos.copy(), MECInfo.copy(),
        B.copy())
    print("原始数据：")
    origin = [taskTimeTotal, taskEnergyTotal, transferTimeTotal, QOETotal]
    print(origin)
    # print(taskTimeTotal, taskEnergyTotal, transferTimeTotal, QOETotal)
    decisionMatResult = randomDecision.copy()
    resourceMatResult = randomResource.copy()
    x = [i for i in range(ITER)]
    timeList = []
    energyList = []
    transferList = []
    QOEList = []
    opts = []
    for i in range(ITER):
        print(f"正在迭代{i + 1}次")

        decisionMatResult = gaResolve_class_decision(randTasksTimeMats.copy(), randTasksDataMats.copy(),
                                                     decisionMatResult.copy(), resourceMatResult.copy(),
                                                     randomCarInfos.copy(), MECInfo.copy(), B.copy())
        # resourceMatResult = gaResolve_class_resource(randTasksTimeMats.copy(), randTasksDataMats.copy(),
        #                                              decisionMatResult.copy(), resourceMatResult.copy(),
        #                                              randomCarInfos.copy(), MECInfo.copy(), B.copy())
        resourceMatResult = psoResolve_class_resource(randTasksTimeMats.copy(), randTasksDataMats.copy(),
                                                      decisionMatResult.copy(), resourceMatResult.copy(),
                                                      randomCarInfos.copy(), MECInfo.copy(), B.copy())
        taskTimeTotal, taskEnergyTotal, transferTimeTotal, QOETotal,taskTimeOriginTotal,taskEnergyOriginTotal = calcTasks(randTasksTimeMats.copy(),
                                                                                randTasksDataMats.copy(),
                                                                                decisionMatResult.copy(),
                                                                                resourceMatResult.copy(),
                                                                                randomCarInfos.copy(), MECInfo.copy(),
                                                                                B.copy())
        print(f"{i + 1}次的结果")
        print(taskTimeTotal, taskEnergyTotal, transferTimeTotal, QOETotal)
        timeList.append(taskTimeTotal)
        energyList.append(taskEnergyTotal)
        transferList.append(transferTimeTotal)
        QOEList.append(QOETotal)
        opts = [taskTimeTotal, taskEnergyTotal, transferTimeTotal, QOETotal]
        print("----------------------------------------------------------------------------------")
    print(opts)
    optRatios = []
    for idx, _ in enumerate(opts):
        optRatios.append(str(round(math.fabs((origin[idx] - opts[idx]) / origin[idx]), 2) * 100) + '%')
    print("优化效果百分比:")
    print(optRatios)
    fig, axs = plt.subplots(2, 2, figsize=(5, 2.7), layout='constrained')
    l1 = axs[0, 0].plot(x, timeList, "r--", label="Time")
    l2 = axs[0, 1].plot(x, energyList, "g--", label="Energy")
    l3 = axs[1, 0].plot(x, transferList, "b--", label="Transfer")
    l4 = axs[1, 1].plot(x, QOEList, "c--", label="QOE")
    axs[0, 0].set_title('Time')
    axs[0, 1].set_title('Energy')
    axs[1, 0].set_title('Transfer')
    axs[1, 1].set_title('QOE')
    # plt.xlabel('Iters')
    # plt.ylabel('column')
    # plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
