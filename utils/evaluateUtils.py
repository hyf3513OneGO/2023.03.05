import math

from matplotlib import pyplot as plt

strategyMap = {
    'time-first': 0,
    'energy-first': 1,
    'transfer-first': 2,
    'QOE-first': 3
}


def strategyEvaluate(strategy: str, resultList: list, sortType: str):
    if strategy not in strategyMap:
        print('err:strategy not in strategyMap')
        return 0
    strategyIndex = strategyMap[strategy]
    strategyResultList = resultList[strategyIndex]
    aimValue = 0
    if sortType == 'min':
        aimValue = min(strategyResultList)
    else:
        aimValue = max(strategyResultList)
    aimIndex = strategyResultList.index(aimValue)
    timeList = resultList[strategyMap['time-first']]
    energyList = resultList[strategyMap['energy-first']]
    transferList = resultList[strategyMap['transfer-first']]
    QOEList = resultList[strategyMap['QOE-first']]

    return [timeList[aimIndex], energyList[aimIndex], transferList[aimIndex], QOEList[aimIndex]]


def allEvaluate(resultList: list):
    originList = [i[0] for i in resultList]

    for i in strategyMap:
        if i in ['QOE-first']:
            resItem = strategyEvaluate(i, resultList, 'max')
        else:
            resItem = strategyEvaluate(i, resultList, 'min')

        print(i)
        print(f'result:{resItem}')
        print(f'opt-ratio:{ratioEvaluate(originList, resItem)}')
        print("----------------------------------------------------------------------------------")


def ratioEvaluate(originList: list, resultList: list):
    optRatios = []
    if len(originList) != len(resultList):
        print("err:len(originList)!=len(resultList)")
        return
    for idx, _ in enumerate(originList):

        optRatios.append(str(round((-(originList[idx] - resultList[idx]) / originList[idx]), 2) * 100) + '%')
    return optRatios


def plotGraph(resultList: list):
    timeList = resultList[strategyMap['time-first']]
    energyList = resultList[strategyMap['energy-first']]
    transferList = resultList[strategyMap['transfer-first']]
    QOEList = resultList[strategyMap['QOE-first']]
    x = [i for i in range(len(timeList))]
    fig, axs = plt.subplots(2, 2, figsize=(5, 2.7), layout='constrained')
    l1 = axs[0, 0].plot(x, timeList, "r--", label="Time")
    l2 = axs[0, 1].plot(x, energyList, "g--", label="Energy")
    l3 = axs[1, 0].plot(x, transferList, "b--", label="Transfer")
    l4 = axs[1, 1].plot(x, QOEList, "c--", label="QOE")
    axs[0, 0].set_title('Time')
    axs[0, 1].set_title('Energy')
    axs[1, 0].set_title('Transfer')
    axs[1, 1].set_title('QOE')
    plt.show()
