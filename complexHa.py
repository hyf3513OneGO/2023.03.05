import datetime
import os.path
import threading

import numpy as np

from utils.datasetUtils import loadDataset
from utils.evaluateUtils import allEvaluate, plotGraph
from utils.calcUtils import calcTasks, calcTask
from utils.haUtils import evaluateDecisionPool, roulette, calcFitList
from utils.mockUtils import getRandomDAGMats, getRandomDecisionMats, getRandomCarInfos, \
    getRandomDataMat
from utils.resolvers.decisions.ga import gaResolve_class_decision
from utils.resolvers.resources.afsa import afsaResolve_class_resource
from utils.resolvers.resources.ga import gaResolve_class_resource
from utils.resolvers.resources.ia import iaResolve_class_resource
from utils.resolvers.resources.pso import psoResolve_class_resource

'''
DAG最大节点数：6
单任务矩阵6x6
最大车辆数10
'''
n = 10
B = [0.8, 0.2] * n
B = np.array(B)
B = B.reshape(n, 2)
ITER = 15
ITERin = 5
InitResolutions = 4
MECInfo = [196, 0.02, 10]  # f,k,B
MECInfo = np.array(MECInfo)

datasetDir = "./datasets/2023.03.07"


def main():
    # 载入数据集
    timeDataset, dataDataset, decisionDataset, resourceDataset, carDataset = loadDataset(os.path.join(datasetDir))
    decisionTimePool = [(decisionDataset[0], resourceDataset[0])]  # 构建原始解池-输入随机解
    timeMat = timeDataset[0]
    dataMat = dataDataset[0]
    decisionMat = decisionDataset[0]
    resourceMat = resourceDataset[0]
    carMat = carDataset[0]
    # 输出原始数据情况
    taskTimeTotal, taskEnergyTotal, transferTimeTotal, QOETotal, taskTimeOriginTotal, taskEnergyOriginTotal = calcTasks(
        timeDataset[0], dataDataset[0], decisionDataset[0], resourceDataset[0], carDataset[0], MECInfo, B)
    print("原始数据：")
    origin = [taskTimeTotal, taskEnergyTotal, transferTimeTotal, QOETotal]
    print(origin)
    # 构建原始解池-输入时间最低，能量最低，传输时间最短，QOE最高
    # 原始解池结果[[[timeMat],[resourceMat]],[[timeMat],[resourceMat]]]
    decisionBest, timeList, energyList, transferList, QOEList = decideOnce(timeMat.copy(), dataMat.copy(),
                                                                           decisionMat.copy(), resourceMat.copy(),
                                                                           carMat.copy(),
                                                                           decisionFunc=gaResolve_class_decision,
                                                                           resourceFunc=psoResolve_class_resource,
                                                                           iterIn=InitResolutions)
    # decisionBest结果[[[decision],[resource]],[[decision],[resource]],[[decision],[resource]],[[decision],[resource]]]
    # 分别是time,energy,transfer,QOE
    # resultList [[timeList],[energyList],[transferList],[QOEList]]
    decisionTimePool.extend(decisionBest)
    print("初始解构造完成")
    # 计算原始解池对应的适应度，用于进行轮盘赌抽取
    objList = calcFitList(decisionTimePool.copy(), timeMat.copy(), dataMat.copy(), carMat.copy(), MECInfo.copy(),
                          B.copy(), 'time')
    # 设置用于卸载决策与分配决策的函数
    # decisionFuncs = [gaResolve_class_decision, gaResolve_class_decision, gaResolve_class_decision]
    # resourceFuncs = [gaResolve_class_resource, psoResolve_class_resource, afsaResolve_class_resource]
    decisionFuncs = [gaResolve_class_decision, gaResolve_class_decision]
    resourceFuncs = [gaResolve_class_resource, psoResolve_class_resource]
    #
    timeTotalList = []
    energyTotalList = []
    transferTotalList = []
    QOETotalList = []
    fitBestTotalList = [[taskTimeTotal], [taskEnergyTotal], [transferTimeTotal], [QOETotal]]
    for idx in range(len(decisionFuncs)):
        timeTotalList.append([taskTimeTotal])
        energyTotalList.append([taskEnergyTotal])
        transferTotalList.append([transferTimeTotal])
        QOETotalList.append([QOETotal])
    for i in range(ITER):
        print(f"外部迭代次数{i}")
        for idx, _ in enumerate(decisionFuncs):
            timeNow = datetime.datetime.now()
            print(f"当前的优化函数:{decisionFuncs[idx].__name__},{resourceFuncs[idx].__name__}")
            # 轮盘赌获得一个初始解
            decisionIndex = roulette(objList)
            decisionMat = decisionTimePool[decisionIndex][0]
            resourceMat = decisionTimePool[decisionIndex][1]
            decisionBest, timeList, energyList, transferList, QOEList = decideOnce(timeMat.copy(), dataMat.copy(),
                                                                                   decisionMat.copy(),
                                                                                   resourceMat.copy(),
                                                                                   carMat.copy(),
                                                                                   decisionFunc=decisionFuncs[idx],
                                                                                   resourceFunc=resourceFuncs[idx],
                                                                                   iterIn=ITERin)
            # 输出本次迭代，本组优化器的效果
            taskTimeTotal, taskEnergyTotal, transferTimeTotal, QOETotal, taskTimeOriginTotal, taskEnergyOriginTotal = calcTasks(
                timeDataset[0], dataDataset[0], decisionBest[0][0], decisionBest[0][1], carDataset[0], MECInfo, B)
            timeTotalList[idx].append(taskTimeTotal)
            energyTotalList[idx].append(taskEnergyTotal)
            transferTotalList[idx].append(transferTimeTotal)
            QOETotalList[idx].append(QOETotal)
            timeDone = datetime.datetime.now()
            timeDelta = timeDone - timeNow
            print(f"{i}次的结果,{timeDone.strftime('%Y-%m-%d %H:%M:%S %f')},耗时:{timeDelta.seconds}秒")
            print(f"迭代{i}轮，求解器:{decisionFuncs[idx].__name__},{resourceFuncs[idx].__name__}效果")
            print([taskTimeTotal, taskEnergyTotal, transferTimeTotal, QOETotal])
            # 计算需要被替换的解，并替换
            replaceID = evaluateDecisionPool(decisionTimePool, timeMat, dataMat, carMat, MECInfo, B, False, 'QOE')
            decisionTimePool[replaceID] = decisionBest[0]
            print(f"解{replaceID}被替换")
            # 计算新的适应度列表
            objList = calcFitList(decisionTimePool.copy(), timeMat.copy(), dataMat.copy(), carMat.copy(),
                                  MECInfo.copy(),
                                  B.copy(), 'time')
            print(f"新的适应度列表")
            print(objList)
            bestID = evaluateDecisionPool(decisionTimePool, timeMat, dataMat, carMat, MECInfo, B, False, 'time')
            choiceBest = decisionTimePool[bestID]
            # print(choiceBest[0].shape)
            taskTimeBest, taskEnergyBest, transferTimeBest, QOEBest, taskTimeOriginTotal, taskEnergyOriginTotal = calcTasks(
                timeDataset[0], dataDataset[0], choiceBest[0], choiceBest[1], carDataset[0], MECInfo, B)
            fitBestTotalList[0].append(taskTimeBest)
            fitBestTotalList[1].append(taskEnergyBest)
            fitBestTotalList[2].append(transferTimeBest)
            fitBestTotalList[3].append(QOEBest)
    # 适应度中最好的结果
    bestID = evaluateDecisionPool(decisionTimePool, timeMat, dataMat, carMat, MECInfo, B, False, 'time')
    choiceBest = decisionTimePool[bestID]
    taskTimeBest, taskEnergyBest, transferTimeBest, QOEBest, taskTimeOriginTotal, taskEnergyOriginTotal = calcTasks(
        timeDataset[0], dataDataset[0], choiceBest[0], choiceBest[1], carDataset[0], MECInfo, B)

    for idx in range(len(decisionFuncs)):
        resultList = [timeTotalList[idx], energyTotalList[idx], transferTotalList[idx], QOETotalList[idx]]
        print(f"**func:{decisionFuncs[idx].__name__}**")
        allEvaluate(resultList)
        plotGraph(resultList)
    print("**fit-best**")
    # print(f"result:{[taskTimeBest, taskEnergyBest, transferTimeBest, QOEBest]}")
    # print(f"opt-ratio:")
    allEvaluate(fitBestTotalList)
    plotGraph(fitBestTotalList)


def decideOnce(timeMats, dataMats, decisionMat, resourceMat, carMat, decisionFunc, resourceFunc, iterIn):
    QOEBest = -1000000000
    timeBest = 100000000
    energyBest = 10000000000000
    transferBest = 1000000000000
    decisionBest = [(), (), (), ()]
    timeList = []
    energyList = []
    transferList = []
    QOEList = []
    for i in range(iterIn):
        print(f"内部迭代次数:{i}")
        decisionMat = decisionFunc(timeMats.copy(), dataMats.copy(),
                                   decisionMat.copy(),
                                   resourceMat.copy(),
                                   carMat.copy(), MECInfo.copy(),
                                   B.copy())
        resourceMat = resourceFunc(timeMats.copy(), dataMats.copy(),
                                   decisionMat.copy(),
                                   resourceMat.copy(),
                                   carMat.copy(), MECInfo.copy(),
                                   B.copy())
        taskTimeTotal, taskEnergyTotal, transferTimeTotal, QOETotal, taskTimeOriginTotal, taskEnergyOriginTotal = calcTasks(
            timeMats.copy(),
            dataMats.copy(),
            decisionMat.copy(),
            resourceMat.copy(),
            carMat.copy(), MECInfo.copy(),
            B.copy())
        if taskTimeOriginTotal < timeBest:
            timeBest = taskTimeTotal
            decisionBest[0] = (decisionMat, resourceMat)
        if taskEnergyTotal < energyBest:
            energyBest = taskEnergyTotal
            decisionBest[1] = (decisionMat, resourceMat)
        if transferTimeTotal < transferBest:
            transferBest = transferTimeTotal
            decisionBest[2] = (decisionMat, resourceMat)
        if QOETotal > QOEBest:
            QOEBest = QOETotal
            decisionBest[3] = (decisionMat, resourceMat)
        timeList.append(taskTimeTotal)
        energyList.append(taskEnergyTotal)
        transferList.append(transferTimeTotal)
        QOEList.append(QOETotal)
    return decisionBest, timeList, energyList, transferList, QOEList


if __name__ == "__main__":
    main()
