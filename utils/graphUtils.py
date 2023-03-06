import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout


def mat2Graph(taskMat: np.array) -> nx.Graph:
    edges = []
    for idx, arr_1d in enumerate(taskMat):
        for idy, item in enumerate(arr_1d):
            if item != 0:
                edges.append((idx, idy, {'weight': item, 'EST': 0}))
                # print(f"({idx},{idy}):{item}")
    G = nx.DiGraph(edges)
    if G.is_directed():

        return G
    else:
        print("err:不是DAG")
        return nx.Graph()


def graph2mat(taskGraph: nx.Graph, attrName: str) -> np.array:
    taskMat = np.zeros((6, 6))
    for edge in taskGraph.edges:
        taskMat[edge[0]][edge[1]] = taskGraph.get_edge_data(edge[0], edge[1])[attrName]
    return taskMat


def showGraph(G: nx.Graph, title: str, nodeAttrsName=[]):
    # print(f"{title}-edges:{G.edges}")
    pos = nx.spring_layout(G)
    weights = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx(G, pos, with_labels=True, arrows=True)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=weights)
    for idx, attr in enumerate(nodeAttrsName):
        attr_pos = [0] * len(G.nodes)
        for node, coords in pos.items():
            attr_pos[node] = (coords[0] + 0.1 * (idx + 1), coords[1] + 0.1 * (idx + 1))
        node_attrs = nx.get_node_attributes(G, attr)
        custom_node_attrs = {}
        for node, attrVal in node_attrs.items():
            attr=str(attr)
            custom_node_attrs[node] = f"{attr:.2}:{attrVal}"
        nx.draw_networkx_labels(G, attr_pos, labels=custom_node_attrs)
    ax = plt.gca()
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def getESTGraph(G_time: nx.Graph) -> nx.Graph:
    lenNodes = len(G_time.nodes)  # 顶点数量
    topoSeq = list(nx.topological_sort(G_time))  # 拓扑序列
    NodeESTList = [0 for i in range(lenNodes)]  # 初始化 事件最早开始时间
    G_EST = nx.DiGraph()
    if not G_time.is_directed():
        print("err:not a DAG")
        return nx.Graph()
    for i in range(lenNodes):
        nx.set_node_attributes(G_time, 0, "EST")
        for edge in G_time.in_edges(topoSeq[i]):
            edgeFull = G_time.get_edge_data(edge[0], edge[1])
            inNode = G_EST.nodes[edge[0]]
            NodeEST = inNode['EST'] + edgeFull['weight']
            # print(i,edge,inNode,NodeEST)
            if NodeEST > NodeESTList[i]:
                NodeESTList[i] = NodeEST
        G_EST.add_node(topoSeq[i], EST=NodeESTList[i])
    new_edges = []
    # for edge in G_time.edges:
    #     edgeFull = G_time.get_edge_data(edge[0], edge[1])
    #     print(edge[0],edge[1],edge,NodeESTList)
    #     G_EST.add_edge(edge[0], edge[1], weight=edgeFull['weight'], EST=NodeESTList[edge[0]],
    #                    ESD=NodeESTList[edge[0]] + edgeFull['weight'])
    return G_EST
    # showGraph(G_EST, 'G_EST', ['EST'])


def getEST(G_EST: nx.Graph) -> float:
    EST = 0
    if not G_EST.is_directed():
        print("err:not a DAG")
        return 0
    for i in nx.topological_sort(G_EST):
        if nx.get_node_attributes(G_EST,'EST')[i]>EST:EST=nx.get_node_attributes(G_EST,'EST')[i]
    return EST
