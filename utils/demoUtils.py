import numpy as np

from utils.graphUtils import showGraph, mat2Graph, getESTGraph, getEST

task1_cpuCost = np.array([[0, 10, 30, 0, 0, 0],
                          [0, 0, 0, 20, 20, 0],
                          [0, 0, 0, 0, 0, 10],
                          [0, 0, 0, 0, 0, 10],
                          [0, 0, 0, 0, 0, 15],
                          [0, 0, 0, 0, 0, 0]
                          ])
task1_dataCost = np.array([[0, 20, 20, 0, 0, 0],
                           [0, 0, 0, 14, 26, 0],
                           [0, 0, 0, 0, 0, 16],
                           [0, 0, 0, 0, 0, 34],
                           [0, 0, 0, 0, 0, 25],
                           [0, 0, 0, 0, 0, 0]
                           ])
task2_cpuCost = np.array([[0, 12, 31, 0, 0, 0],
                          [0, 0, 0, 23, 0, 0],
                          [0, 0, 0, 24, 0, 0],
                          [0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0]
                          ])
task2_dataCost = np.array([[0, 11, 23, 0, 0, 0],
                           [0, 0, 0, 43, 0, 0],
                           [0, 0, 0, 13, 0, 0],
                           [0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0]
                           ])
def showGraphs():
    task1_cpuGraph = mat2Graph(task1_cpuCost)
    task2_cpuGraph = mat2Graph(task2_cpuCost)
    task1_dataGraph = mat2Graph(task1_dataCost)
    task2_dataGraph = mat2Graph(task2_dataCost)
    # showGraph(task1_cpuGraph, 'task1-' + 'Cpu Cost Graph')
    # showGraph(task2_cpuGraph, 'task2-' + 'Cpu Cost Graph')
    # showGraph(task1_dataGraph, 'task1-' + 'Data Cost Graph')
    # showGraph(task2_dataGraph, 'task2-' + 'Data Cost Graph')
    task1_ESTGraph=getESTGraph(task1_cpuGraph)
    est=getEST(task1_cpuGraph)
    print(est)
    showGraph(task1_ESTGraph,'task1-EST',["EST"])
def main():
    showGraphs()


if __name__=="__main__":
    main()