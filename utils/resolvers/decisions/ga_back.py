import numpy as np

from utils.calcUtils import calcTasks
from utils.solveUtils import mat2geatVar, geatVars2mat
import time

import geatpy as ea
import numpy as np

# decisionMat不是决策变量的矩阵而是子任务的矩阵
from utils.calcUtils import calcTasks
from utils.mockUtils import getRandomDAGMats, getRandomDataMat, getRandomDecisionMats, getRandomCarInfos

def gaResolve(randTasksTimeMats, randTasksDataMats, decisionMat, randomResource, randomCarInfos, MECInfo, B):
    print("--------------遗传算法-------------")
    geatVars, lenV = mat2geatVar(randTasksTimeMats)
    NIND = 150;  # 种群个体数目
    MAXGEN = 20;  # 最大遗传代数
    maxormins = [-1]  # 列表元素为1则表示对应的目标函数是最小化，元素为-1则表示对应的目标函数是最大化
    maxormins = np.array(maxormins)  # 转化为Numpy array行向量
    selectStyle = 'rws'  # 采用轮盘赌选择
    recStyle = 'xovdp'  # 采用两点交叉
    mutStyle = 'mutbin'  # 采用二进制染色体的变异算子
    Lind = lenV  # 计算染色体长度
    pc = 0.9  # 交叉概率
    pm = 1 / Lind  # 变异概率
    obj_trace = np.zeros((MAXGEN, 2))  # 定义目标函数值记录器
    var_trace = np.zeros((MAXGEN, Lind))  # 染色体记录器，记录历代最优个体的染色体
    Encoding = 'RI'  # 表示采用“实整数编码”，即变量可以是连续的也可以是离散的
    print(f"个体数量：{NIND}，最大遗传代数：{MAXGEN}，交叉概率:{pc},变异概率:{pm}")
    # 创建“区域描述器”，表明有4个决策变量，范围分别是[-3.1, 4.2], [-2, 2],[0, 1],[3, 3]，
    # FieldDR第三行[0,0,1,1]表示前两个决策变量是连续型的，后两个变量是离散型的
    FieldDR = np.array([[0 for i in range(lenV)],
                        [1 for i in range(lenV)],
                        [1 for i in range(lenV)]])
    Chrom = ea.crtpc(Encoding, NIND, FieldDR)

    """=========================开始遗传算法进化========================"""
    start_time = time.time()  # 开始计时

    for gen in range(MAXGEN):
        cvList = []
        QOElist = []
        # 计算目标函数值，每个个体都要计算一次
        print(f"遗传代数:{gen + 1}/{MAXGEN}")
        for idx in range(NIND):
            cvList.append([0])
            decisionMat, _ = geatVars2mat(Chrom[idx], randTasksTimeMats)
            taskTimeTotal, taskEnergyTotal, transferTimeTotal, QOETotal = calcTasks(randTasksTimeMats,
                                                                                    randTasksDataMats,
                                                                                    decisionMat, randomResource,
                                                                                    randomCarInfos, MECInfo, B)
            QOElist.append([QOETotal])
        CV = np.stack(cvList)
        ObjV = np.stack(QOElist)
        FitnV = ea.ranking(ObjV, CV, maxormins)
        SelCh = Chrom[ea.selecting(selectStyle, FitnV, NIND - 1), :]  # 选择
        SelCh = ea.recombin(recStyle, SelCh, pc)  # 重组
        # SelCh = ea.mutate(mutStyle, Encoding, SelCh, pm)  # 变异
        best_ind = np.argmax(FitnV)
        # 把父代精英个体与子代的染色体进行合并，得到新一代种群
        Chrom = np.vstack([Chrom[best_ind, :], SelCh])
        best_ind = np.argmax(FitnV)  # 计算当代最优个体的序号
        obj_trace[gen, 0] = np.sum(ObjV) / ObjV.shape[0]  # 记录当代种群的目标函数均值
        obj_trace[gen, 1] = ObjV[best_ind]  # 记录当代种群最优个体目标函数值
        var_trace[gen, :] = Chrom[best_ind, :]  # 记录当代种群最优个体的染色体
    end_time = time.time()  # 结束计时
    ea.trcplot(obj_trace, [['种群个体平均目标函数值', '种群最优个体目标函数值']])  # 绘制图像