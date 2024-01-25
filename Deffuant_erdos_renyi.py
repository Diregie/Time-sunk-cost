import copy

from mpl_toolkits.mplot3d import Axes3D
import networkx as nx
import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('Agg')
# from matplotlib import pyplot as plt

import random as random
import math as math
# from colormap import Colormap
import numpy as np
from matplotlib import cm, patches
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from tempfile import TemporaryFile
import csv
import seaborn as sns


# from Hegselmann_Krause.HK import fig_scatter


def show_scatter(t):
    fig_scatter.hold(True)
    for i in range(1, N, 1):
        plt.scatter(time - t, G.node[i]['value'], marker='o', s=30, facecolors='none', edgecolors='r')
        plt.xlabel('time')
        plt.ylabel('opinion')
        plt.show


def show_graph(flag=0):
    plt.figure()
    pos = nx.random_layout(G)
    # nx.draw_networkx_nodes(G, pos, cmap=plt.get_cmap('coolwarm'), vmin=0, vmax=1, node_color=color_map, alpha=0.6)
    nx.draw_networkx_nodes(G, pos, alpha=0.5, node_size=20)
    nx.draw_networkx_edges(G, pos, width=0.01, alpha=0.2)
    # nx.draw_networkx_labels(G, pos)
    if flag == 0:
        plt.draw()
        plt.pause(0.05)
    else:
        plt.show()


def create_graph():
    # count = 0
    for i in range(0, N, 1):
        x = random.uniform(-1, 1)
        # if abs(x) > 0.9:
        #     count += 1
        color_map.append(x)
        G.node[i]['value'] = x
    # print(count)


def extreme_generated():
    extreme = 0
    for i in range(0, N, 1):
        if abs(G.node[i]['value']) > 0.9:
            extreme += 1
            if G.node[i]['value'] > 0:
                G.node[i]['value'] = 1
                break
            else:
                G.node[i]['value'] = -1
                break


def evolve_graph(time):
    # 置信度
    theta = 0.2
    mu = 0.4
    rho = 0.4
    extreme = 0
    batch = 700
    # roundData = [[0 for i in range(time)]]
    for t in range(0, time, 1):
        for b in range(batch):
            v = math.floor(random.uniform(1, N))
            if REA is True and abs(G.node[v]['value']) > 0.9 and t >= REA_time:
                x = random.uniform(-1, 1)           # 有一半概率拦截
                if x >= limit_value:
                    continue
            # degree = nx.degree(G, v)
            v_neighbors = list(nx.all_neighbors(G, v))
            if v_neighbors:
                w = v_neighbors[int(math.floor(random.uniform(0, nx.degree(G, v))))]
                if REA is True and abs(G.node[w]['value']) > 0.9 and t >= REA_time:
                    x = random.uniform(-1, 1)
                    if x >= limit_value:
                        continue
                w_neighbors = list(nx.all_neighbors(G, w))
                sunk_parameter_v = sunk_cost(G.node[w]['value'], G.node[v]['value'], v)  # 沉没系数
                sunk_parameter_w = sunk_cost(G.node[w]['value'], G.node[v]['value'], w)
                pre_opinion_v = G.node[v]['value']  # 融合前观点
                pre_opinion_w = G.node[w]['value']
                # 观望状态&观望状态
                if abs(G.node[v]['value']) <= 0.3 and abs(G.node[w]['value']) <= 0.3:
                    if abs(G.node[v]['value']) > abs(G.node[w]['value']):
                        G.node[w]['value'] = sunk_parameter_w * (G.node[w]['value'] + G.node[v]['value']) / 2
                        G.node[v]['value'] = sunk_parameter_v * (1 + rho) * (
                                G.node[w]['value'] + G.node[v]['value']) / 2
                        sunk_round(pre_opinion_v, G.node[v]['value'], v)
                        sunk_round(pre_opinion_w, G.node[w]['value'], w)
                        color_map[int(w - 1)] = G.node[w]['value']
                        color_map[int(v - 1)] = G.node[v]['value']
                    else:
                        G.node[v]['value'] = sunk_parameter_v * (G.node[w]['value'] + G.node[v]['value']) / 2
                        G.node[w]['value'] = sunk_parameter_w * (1 + rho) * (
                                G.node[w]['value'] + G.node[v]['value']) / 2
                        sunk_round(pre_opinion_v, G.node[v]['value'], v)
                        sunk_round(pre_opinion_w, G.node[w]['value'], w)
                        color_map[int(v - 1)] = G.node[v]['value']
                        color_map[int(w - 1)] = G.node[v]['value']
                # 观望状态&主观状态|坚定状态
                elif abs(G.node[v]['value']) <= 0.3 and 0.3 < abs(G.node[w]['value']) <= 0.9 or \
                        abs(G.node[w]['value']) <= 0.3 and 0.3 < abs(G.node[v]['value']) <= 0.9:
                    if abs(G.node[v]['value']) > 0.3:
                        G.node[w]['value'] = sunk_parameter_w * (G.node[w]['value'] + mu *
                                                                 (G.node[v]['value'] - G.node[w]['value']))
                        sunk_round(pre_opinion_v, G.node[v]['value'], v)
                        sunk_round(pre_opinion_w, G.node[w]['value'], w)
                        color_map[int(w - 1)] = G.node[w]['value']
                    else:
                        G.node[v]['value'] = sunk_parameter_v * (G.node[v]['value'] + mu *
                                                                 (G.node[w]['value'] - G.node[v]['value']))
                        sunk_round(pre_opinion_v, G.node[v]['value'], v)
                        sunk_round(pre_opinion_w, G.node[w]['value'], w)
                        color_map[int(v - 1)] = G.node[v]['value']
                # 观望状态&极端状态
                elif abs(G.node[v]['value']) <= 0.3 and abs(G.node[w]['value']) > 0.9 or \
                        abs(G.node[w]['value']) <= 0.3 and abs(G.node[v]['value']) > 0.9:
                    if abs(G.node[v]['value']) <= 0.3:
                        ex_neighbor = extreme_neighbor(v_neighbors)
                        G.node[v]['value'] = G.node[v]['value'] - mu * sunk_parameter_v * \
                                             (ex_neighbor + len(v_neighbors)) / \
                                             len(v_neighbors) * (G.node[w]['value'] - G.node[v]['value'])
                        sunk_round(pre_opinion_v, G.node[v]['value'], v)
                        sunk_round(pre_opinion_w, G.node[w]['value'], w)
                        # 修正
                        if G.node[v]['value'] < -1:
                            G.node[v]['value'] = -1
                        if G.node[v]['value'] > 1:
                            G.node[v]['value'] = 1
                        color_map[int(v - 1)] = G.node[v]['value']
                    else:
                        ex_neighbor = extreme_neighbor(w_neighbors)
                        G.node[w]['value'] = G.node[w]['value'] - mu * sunk_parameter_w * \
                                             (ex_neighbor + len(v_neighbors)) / \
                                             len(v_neighbors) * (G.node[v]['value'] - G.node[w]['value'])
                        sunk_round(pre_opinion_v, G.node[v]['value'], v)
                        sunk_round(pre_opinion_w, G.node[w]['value'], w)
                        # 修正
                        if G.node[w]['value'] < -1:
                            G.node[w]['value'] = -1
                        if G.node[w]['value'] > 1:
                            G.node[w]['value'] = 1
                        color_map[int(w - 1)] = G.node[w]['value']
                # 主观状态&主观状态 | 主观状态&坚定状态 | 坚定状态&主观状态
                elif 0.3 < G.node[v]['value'] <= 0.7 and 0.3 < G.node[w]['value'] <= 0.9 or \
                        0.3 <= G.node[v]['value'] < 0.7 and -0.7 <= G.node[w]['value'] < -0.3 or \
                        -0.7 <= G.node[v]['value'] < -0.3 and 0.3 < G.node[w]['value'] <= 0.7 or \
                        -0.7 <= G.node[v]['value'] < -0.3 and -0.9 <= G.node[w]['value'] < -0.3:
                    G.node[v]['value'] = G.node[v]['value'] + theta * sunk_parameter_v * \
                                         (G.node[w]['value'] - G.node[v]['value'])
                    G.node[w]['value'] = G.node[w]['value'] + theta * sunk_parameter_w * \
                                         (G.node[v]['value'] - G.node[w]['value'])
                    sunk_round(pre_opinion_v, G.node[v]['value'], v)
                    sunk_round(pre_opinion_w, G.node[w]['value'], w)
                    color_map[int(v - 1)] = G.node[v]['value']
                    color_map[int(w - 1)] = G.node[w]['value']
                # 主观状态&极端状态
                elif 0.3 < abs(G.node[v]['value']) <= 0.7 and abs(G.node[w]['value']) > 0.9 or \
                        0.3 < abs(G.node[w]['value']) <= 0.7 and abs(G.node[v]['value']) > 0.9:
                    if abs(G.node[v]['value']) > 0.9:
                        ex_neighbor = extreme_neighbor(w_neighbors)
                        G.node[w]['value'] = G.node[w]['value'] - theta * sunk_parameter_w * \
                                             (ex_neighbor + len(v_neighbors)) / \
                                             len(v_neighbors) * (G.node[v]['value'] - G.node[w]['value'])
                        sunk_round(pre_opinion_v, G.node[v]['value'], v)
                        sunk_round(pre_opinion_w, G.node[w]['value'], w)
                        # 修正
                        if G.node[w]['value'] < -1:
                            G.node[w]['value'] = -1
                        if G.node[w]['value'] > 1:
                            G.node[w]['value'] = 1
                        color_map[int(w - 1)] = G.node[w]['value']
                    else:
                        ex_neighbor = extreme_neighbor(v_neighbors)
                        G.node[v]['value'] = G.node[v]['value'] - theta * sunk_parameter_v * \
                                             (ex_neighbor + len(v_neighbors)) / \
                                             len(v_neighbors) * (G.node[w]['value'] - G.node[v]['value'])
                        sunk_round(pre_opinion_v, G.node[v]['value'], v)
                        sunk_round(pre_opinion_w, G.node[w]['value'], w)
                        if G.node[v]['value'] > 1:
                            G.node[v]['value'] = 1
                        if G.node[v]['value'] < -1:
                            G.node[v]['value'] = -1
                        color_map[int(v - 1)] = G.node[v]['value']
                # 坚定状态&坚定状态
                elif 0.7 < G.node[v]['value'] <= 0.9 and 0.7 < G.node[w]['value'] <= 0.9 or \
                        -0.9 <= G.node[v]['value'] < -0.7 and -0.9 <= G.node[w]['value'] < -0.7:
                    G.node[v]['value'] = G.node[v]['value'] + mu * sunk_parameter_v * \
                                         (G.node[w]['value'] - G.node[v]['value'])
                    G.node[w]['value'] = G.node[w]['value'] + mu * sunk_parameter_w * \
                                         (G.node[v]['value'] - G.node[w]['value'])
                    sunk_round(pre_opinion_v, G.node[v]['value'], v)
                    sunk_round(pre_opinion_w, G.node[w]['value'], w)
                    color_map[int(v - 1)] = G.node[v]['value']
                    color_map[int(w - 1)] = G.node[w]['value']
                # 坚定状态&极端状态
                elif 0.7 < G.node[v]['value'] <= 0.9 and G.node[w]['value'] < -0.9:
                    G.node[v]['value'] = 1
                    sunk_round(pre_opinion_v, G.node[v]['value'], v)
                    sunk_round(pre_opinion_w, G.node[w]['value'], w)
                    color_map[int(v - 1)] = G.node[v]['value']
                elif -0.9 <= G.node[v]['value'] < -0.7 and G.node[w]['value'] > 0.9:
                    G.node[v]['value'] = -1
                    sunk_round(pre_opinion_v, G.node[v]['value'], v)
                    sunk_round(pre_opinion_w, G.node[w]['value'], w)
                    color_map[int(v - 1)] = G.node[v]['value']
                elif 0.7 < G.node[w]['value'] <= 0.9 and G.node[v]['value'] < -0.9:
                    G.node[w]['value'] = 1
                    sunk_round(pre_opinion_v, G.node[v]['value'], v)
                    sunk_round(pre_opinion_w, G.node[w]['value'], w)
                    color_map[int(w - 1)] = G.node[w]['value']
                elif -0.9 <= G.node[w]['value'] < -0.7 and G.node[v]['value'] > 0.9:
                    G.node[w]['value'] = -1
                    sunk_round(pre_opinion_v, G.node[v]['value'], v)
                    sunk_round(pre_opinion_w, G.node[w]['value'], w)
                    color_map[int(w - 1)] = G.node[w]['value']


            else:
                print('selected vertex has no neighbors')

            # 每一轮的观点
            # roundData[t] = color_map
            # if (math.fmod(time,30) == 0):
            # show_graph()
        for i in range(0, N, 1):
            # print(repr(i) + ' vv' + repr(t))
            data[t][i] = G.node[i]['value']
        # if( time == 20000 or (Time - time) % 9800 == 0):
        # 	show_scatter(time)
        # time = time - 1


def sunk_cost(opinion_i, opinion_j, agent):
    if opinion_i * opinion_j > 0:  # 同号，相近观点
        sunk_parameter = 1
    else:
        sunk_parameter = 1 - math.log(sunk_time[agent - 1], 10) / math.log(time, 10) / 2
    return sunk_parameter


def sunk_round(pre_opinion, cur_opinion, agent):
    if pre_opinion * cur_opinion < 0:  # 观点转向
        sunk_time[int(agent - 1)] = 2
    else:
        sunk_time[int(agent - 1)] += 1


def multiple_experiments(rounds):
    for i in range(rounds):
        print("round: " + str(i + 1))
        execute()
        multi_sta_data[i] = copy.deepcopy(sta_data)
        multi_sta2_data[i] = copy.deepcopy(sta2_data)
        multi_data[i] = copy.deepcopy(data)
    multiple_heatmap()
    multiple_quantity_chart()
    print("over rounds")


# def extreme_limit(option):
#     if option == true:
#
#     return


def extreme_neighbor(neighbors):
    ex_neighbor = 0
    for i in range(len(neighbors)):
        if abs(G.node[i]["value"]) > 0.9:
            ex_neighbor += 1
    return ex_neighbor


def plot_comparison_test(op1, op_eps):
    c1 = sns.xkcd_rgb['red']
    c2 = sns.xkcd_rgb['green']
    # print('[+] HK Original Clusters = {}'.format(cluster_count(op1[-1], op_eps)))
    op1 = np.array(op1)
    # N = op1.shape[1]
    for i in range(N):
        plt.plot(op1[:, i], c1, alpha=0.3, linewidth=0.1)
    # plt.vlines(special_rounds, 0.0, 1.0, linewidth=5)
    p1 = patches.Patch(color=c1, label='HK model')
    legend = plt.legend(loc='upper right')
    legend.get_frame().set_facecolor('#FFFFFF')
    plt.xlabel('round')
    plt.ylabel('Group opinion')
    my_x_axi = []
    for i in range(1, 21, 1):
        my_x_axi.append(i)
    plt.xticks(np.arange(0, 20), my_x_axi)


def statistics():
    global sta_data
    for j in range(time):
        waitandsee_state = 0  # 第一种分类法，按状态分
        subjective_state = 0
        firm_state = 0
        extreme_state = 0
        waitandsee_state2 = 0
        subjective_state2 = 0
        firm_state2 = 0
        extreme_state2 = 0

        range1 = 0  # 第二种分类法，每0.2分
        range2 = 0
        range3 = 0
        range4 = 0
        range5 = 0
        range6 = 0
        range7 = 0
        range8 = 0
        range9 = 0
        range10 = 0
        for i in range(N):
            if data[j][i] < -0.9:
                extreme_state2 += 1
            elif data[j][i] < -0.7:
                firm_state2 += 1
            elif data[j][i] < -0.3:
                subjective_state2 += 1
            elif data[j][i] < 0:
                waitandsee_state2 += 1
            elif data[j][i] <= 0.3:
                waitandsee_state += 1
            elif data[j][i] <= 0.7:
                subjective_state += 1
            elif data[j][i] <= 0.9:
                firm_state += 1
            else:
                extreme_state += 1

            if data[j][i] < -0.8:
                range1 += 1
            elif data[j][i] < -0.6:
                range2 += 1
            elif data[j][i] < -0.4:
                range3 += 1
            elif data[j][i] < -0.2:
                range4 += 1
            elif data[j][i] < 0:
                range5 += 1
            elif data[j][i] < 0.2:
                range6 += 1
            elif data[j][i] < 0.4:
                range7 += 1
            elif data[j][i] < 0.6:
                range8 += 1
            elif data[j][i] < 0.8:
                range9 += 1
            else:
                range10 += 1
        sta_data[j][0] = extreme_state
        sta_data[j][1] = firm_state
        sta_data[j][2] = subjective_state
        sta_data[j][3] = waitandsee_state
        sta_data[j][4] = waitandsee_state2
        sta_data[j][5] = subjective_state2
        sta_data[j][6] = firm_state2
        sta_data[j][7] = extreme_state2
        sta2_data[j][0] = range1
        sta2_data[j][1] = range2
        sta2_data[j][2] = range3
        sta2_data[j][3] = range4
        sta2_data[j][4] = range5
        sta2_data[j][5] = range6
        sta2_data[j][6] = range7
        sta2_data[j][7] = range8
        sta2_data[j][8] = range9
        sta2_data[j][9] = range10
    print("was : " + str(waitandsee_state + waitandsee_state2))
    print("sub : " + str(subjective_state + subjective_state2))
    print("firm: " + str(firm_state + firm_state2))
    print("ex  : " + str(extreme_state + extreme_state2))


def statistics_graph():
    global sta_data
    staData = np.array(sta_data)
    for i in range(8):
        plt.plot(staData[:, 0], "blue", alpha=0.2, linewidth=0.5)
        plt.plot(staData[:, 1], "grey", alpha=0.3, linewidth=0.5)
        plt.plot(staData[:, 2], "red", alpha=0.2, linewidth=0.5)
        plt.plot(staData[:, 3], "green", alpha=0.2, linewidth=0.5)
        plt.plot(staData[:, 4], "yellow", alpha=0.6, linewidth=0.5)
        plt.plot(staData[:, 5], "orange", alpha=0.6, linewidth=0.5)
        plt.plot(staData[:, 6], "black", alpha=0.4, linewidth=0.5)
        plt.plot(staData[:, 7], "purple", alpha=0.4, linewidth=0.5)
    p1 = patches.Patch(color="yellow", label='Positive view of the watch state')
    p2 = patches.Patch(color="green", label='Negative view of the watch state')
    p3 = patches.Patch(color="orange", label='Positive view of the subjective state')
    p4 = patches.Patch(color="red", label='Negative view of the subjective state')
    p5 = patches.Patch(color="black", label='Positive view of the firm state')
    p6 = patches.Patch(color="grey", label='Negative view of the firm state')
    p7 = patches.Patch(color="purple", label='Positive view of the extreme state')
    p8 = patches.Patch(color="blue", label='Negative view of the extreme state')

    legend = plt.legend(handles=[p1, p2, p3, p4, p5, p6, p7, p8], shadow=True, frameon=True, loc=8,
                        bbox_to_anchor=(0.5, -0.2), ncol=10)  # 标注
    legend.get_frame().set_facecolor('#FFFFFF')
    my_x_axi = []
    for i in range(1, 21, 1):
        my_x_axi.append(i)
    plt.xticks(np.arange(0, 20), my_x_axi)


def multiple_heatmap():
    # sns.set_theme()

    # uniform_data = [[0 for i in range(8)] for j in range(time)]  # 以八种状态为纵坐标
    # for i in range(time):
    #     for j in range(8):
    #         single_sum = 0  # 单个总数
    #         for k in range(rounds):
    #             single_sum += multi_sta_data[k][i][j]
    #         uniform_data[i][j] = single_sum // rounds

    uniform_data2 = [[0 for i in range(10)] for j in range(time)]
    for i in range(time):
        for j in range(10):
            single_sum = 0  # 单个总数
            for k in range(rounds):
                single_sum += multi_sta2_data[k][i][j]
            uniform_data2[i][j] = single_sum // rounds
    x_axis = []
    y_axis = []
    for i in range(1, 21, 1):
        x_axis.append(i)
    # for i in range(-1, 1, 0.2):
    #     y_axis.append(i)
    ax = sns.heatmap(np.transpose(uniform_data2),  # 数据
                     annot=True, fmt="d",  # 是否标出对应数据， 格式
                     linewidths=0,  # 方格间距
                     xticklabels=x_axis,
                     yticklabels=("(0.8,1]", "(0.6,0.8]", "(0.4,0.6]", "(0.2,0.4]", "(0,0.2]",
                                  "(-0.2,0]", "(-0.4,-0.2]", "(-0.6,-0.4]", "(-0.8,-0.6]", "[-1,-0.8]"),
                     # yticklabels=("extreme+", "firm+", " subject+", "waitandsee+",
                     #              "waitandsee- ", " subject- ", "firm- ", "extreme- "),
                     # cbar_ax=(0, 1000),
                     center=300,
                     cmap="Greens"  # 背景配色
                     )
    ax.set_xlabel("Time")
    ax.set_ylabel("Opinion")
    # sns.set(font_scale=1.5)
    plt.show()


def multiple_quantity_chart():
    uniform_data3_final = [0 for i in range(8)]
    uniform_data3_5 = [0 for i in range(8)]
    uniform_data3_10 = [0 for i in range(8)]
    uniform_data3_15 = [0 for i in range(8)]
    for i in range(8):
        single_sum_final = 0  # 单个最终时间点总数
        single_sum_5 = 0  # 单个5时间点总数
        single_sum_15 = 0  # 单个5时间点总数
        single_sum_10 = 0  # 单个10时间点总数
        for j in range(rounds):
            single_sum_final += multi_sta_data[j][19][i]
            single_sum_5 += multi_sta_data[j][4][i]
            single_sum_15 += multi_sta_data[j][14][i]
            single_sum_10 += multi_sta_data[j][9][i]
        uniform_data3_final[i] = single_sum_final // rounds
        uniform_data3_5[i] = single_sum_5 // rounds
        uniform_data3_10[i] = single_sum_10 // rounds
        uniform_data3_15[i] = single_sum_15 // rounds
        # print(str(uniform_data3_5[i]) + "---->" + str(uniform_data3_10[i]))
        print(str(uniform_data3_15[i]) + "---->" + str(uniform_data3_final[i]))


def all_degrees():
    print("edges = ", end='')
    print(nx.number_of_edges(G))


def average_clustering_coefficient():
    print("clustering = ", end='')
    print(nx.average_clustering(G))


def output_to_csv(data1):
    with open('data3.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, dialect='excel')
        # for i in range(len(data1)):
        spamwriter.writerows(data1)
    print("output csv successfully")


def input_from_csv():
    i = 0
    with open('initial_data.csv', 'r') as csvfile:
        for row in csv.reader(csvfile, skipinitialspace=True):
            for item in row:
                # print(item)
                color_map.append(float(item))
                G.node[i]['value'] = float(item)
            i += 1
    csvfile.close()


def execute():
    input_from_csv()
    # create_graph()
    # extreme_generated()  # 生成极端
    # show_graph(1)
    # fig_scatter = plt.figure(2)     # 标准化列向量
    evolve_graph(time)
    # plt.show()
    # fig = plt.figure(3)
    # ax = fig.add_subplot(111)  # 在画布中增加轴域
    # x = np.arange(0, N, 1)
    # y = np.arange(0, time, 1)
    # X, Y = np.meshgrid(x, y)  # x中每一个数据和y中每一个数据组合生成很多点,然后将这些点的x坐标放入到X中,y坐标放入Y中,并且相应位置是对应的
    # print('X' + repr(np.shape(X)) + 'Y' + repr(np.shape(Y)) + 'Z' + repr(np.shape(data)))
    # # surf = ax.plot_surface(X, Y, data, cmap=cm.plasma, linewidth=0, antialiased=False)
    # contourqq = ax.contourf(Y, X, data, cmap='jet')
    # cbar = fig.colorbar(contourqq)
    # plt.show()
    # with open("output.csv", "w") as f:
    #     writer = csv.writer(f)
    #     writer.writerows(data)
    # data.dump("Deffuant_erdos_100_1000.dat")
    # show_graph(4)

    # plot_comparison_test(data, 0.25)    # 趋势图
    # plt.plot()
    # plt.show()
    #
    statistics()
    statistics_graph()      # 数量图
    # plt.plot()
    # plt.show()

    # output_to_csv(data)

    print("over")
    print("-------------------------------------------------------------")


N = 2000  # 点
# E = 10
# Time = 10
time = 20  # 进化次数
rounds = 50
REA = True
REA_time = 15
limit_value = -0.5
data = [[0 for i in range(N)] for j in range(time)]  # 点*进化次数的矩阵
sta_data = [[0 for i in range(8)] for j in range(time)]
sta2_data = [[0 for i in range(10)] for j in range(time)]
multi_sta_data = [[[0 for i in range(8)] for j in range(time)] for k in range(rounds)]
multi_sta2_data = [[[0 for i in range(10)] for j in range(time)] for k in range(rounds)]
multi_data = [[[0 for i in range(N)] for j in range(time)] for k in range(rounds)]
sunk_time = [2 for i in range(N)]

# G = nx.erdos_renyi_graph(N, 0.1)  # erdos-renyi随机图
G = nx.barabasi_albert_graph(N, 106)    # Barabási–Albert无标度图
G = nx.watts_strogatz_graph(N, 200, 0.3)     # Watts–Strogatz小世界网络

all_degrees()
# average_clustering_coefficient()
color_map = []  # 意见表
if __name__ == '__main__':
    multiple_experiments(rounds)
