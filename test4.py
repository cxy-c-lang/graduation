import re
import pandas as pd
from geopy.distance import geodesic
import networkx as nx
import matplotlib.pyplot as plt
from itertools import permutations
from PIL import Image

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


class CampusGraph:
    def __init__(self):
        self.att = {}
        self.latitude = {}
        self.longitude = {}
        self.edges = {}
        self.distance_matrix = {}
        self.complete_edges = {}


def getCampus(g, filename):
    data = pd.read_excel(filename)
    for index, row in data.iterrows():
        name = re.sub(r'[^\w\s]', '', row["配送点名称"])
        g.att[index + 1] = name
        g.latitude[index + 1] = row["纬度"]
        g.longitude[index + 1] = row["经度"]


def calcDistance(g):
    G = nx.Graph()
    for i in g.att:
        G.add_node(i)
    for i in range(1, len(g.att) + 1):
        for j in range(i + 1, len(g.att) + 1):
            loc1 = (g.latitude[i], g.longitude[i])
            loc2 = (g.latitude[j], g.longitude[j])
            distance = int(geodesic(loc1, loc2).meters)
            G.add_edge(i, j, weight=distance)
            g.edges[(i, j)] = distance
            g.edges[(j, i)] = distance
    g.distance_matrix = dict(nx.all_pairs_dijkstra_path_length(G, weight='weight'))
    for u in G.nodes():
        for v in G.nodes():
            g.complete_edges[(u, v)] = g.distance_matrix[u][v]


def get_distance(g, node1, node2):
    return g.distance_matrix[node1][node2]


def get_shortest_path(campus, start_id, end_id):
    G = nx.Graph()
    for u, v in campus.complete_edges:
        G.add_edge(u, v, weight=campus.complete_edges[(u, v)])
    return nx.shortest_path(G, start_id, end_id, weight='weight'), campus.complete_edges[(start_id, end_id)]


def plot_graph(campus, path_nodes, title):
    G = nx.Graph()
    pos = {node: (campus.longitude[node], campus.latitude[node]) for node in campus.att}

    plt.figure(figsize=(12, 8))
    nx.draw_networkx_nodes(G, pos, nodelist=pos.keys(), node_size=500, node_color='lightblue')
    nx.draw_networkx_labels(G, pos, labels=campus.att)

    path_edges = [(path_nodes[i], path_nodes[i + 1]) for i in range(len(path_nodes) - 1)]
    nx.draw_networkx_nodes(G, pos, nodelist=path_nodes, node_color='red', node_size=700)
    nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='blue', width=3)

    plt.title(title)
    plt.axis('off')


def get_best_delivery(campus, start, delivery_info):
    nodes = [n for n, _ in delivery_info]
    time_dict = {n: t for n, t in delivery_info}

    # 初始化各策略的最优值
    min_distance = float('inf')
    min_wait = float('inf')
    min_balance = float('inf')
    best_dist_path = []
    best_wait_path = []
    best_balance_path = []

    for perm in permutations(nodes):
        try:
            # 计算总距离
            total_dist = get_distance(campus, start, perm[0])
            for i in range(len(perm) - 1):
                total_dist += get_distance(campus, perm[i], perm[i + 1])

            # 计算时间指标
            current_time = 0
            total_wait = 0
            prev = start
            for node in perm:
                leg_time = get_distance(campus, prev, node) / 5  # 假设移动速度5m/s
                current_time += leg_time
                total_wait += current_time  # 累加到达时间
                current_time += time_dict[node]  # 增加处理时间
                prev = node

            # 计算平衡指标（总送餐时间 + 总等待时间）
            total_delivery_time = current_time  # 总送餐时间
            alpha, beta = 0.7, 0.3
            balance_cost = alpha*total_delivery_time + beta*total_wait

            # 更新最短路径策略
            if total_dist < min_distance:
                min_distance = total_dist
                best_dist_path = perm
                dist_wait = total_wait
                dist_delivery = total_delivery_time

            # 更新最优等待策略
            if total_wait < min_wait:
                min_wait = total_wait
                best_wait_path = perm
                wait_dist = total_dist
                wait_delivery = total_delivery_time

            # 更新平衡策略
            if balance_cost < min_balance:
                min_balance = balance_cost
                best_balance_path = perm
                balance_dist = total_dist
                balance_wait = total_wait
                balance_delivery = total_delivery_time

        except KeyError:
            continue

    return (
        ([start] + list(best_dist_path), min_distance, dist_wait, dist_delivery),
        ([start] + list(best_wait_path), wait_dist, min_wait, wait_delivery),
        ([start] + list(best_balance_path), balance_dist, balance_wait, balance_delivery)
    )


def main():
    campus = CampusGraph()
    getCampus(campus, "武汉纺织大学经纬度.xlsx")
    calcDistance(campus)

    try:
        image = Image.open('校园地图.jpg')
        image.show()
    except FileNotFoundError:
        print("未找到校园地图图片")

    print("============ 校园订单配送路径规划系统 ============")
    while True:
        print("\n1. 查看配送点列表\n2. 两点间最短路径\n3. 订单配送规划\n4. 退出")
        choice = input("请选择：")

        if choice == '1':
            for n, name in campus.att.items():
                print(f"{n}: {name}")

        elif choice == '2':
            start = int(input("起点编号："))
            end = int(input("终点编号："))
            path, dist = get_shortest_path(campus, start, end)
            print(f"最短路径：{'→'.join(map(str, path))} 距离：{dist}米")
            plot_graph(campus, path, "最短路径示意图")
            plt.show()

        elif choice == '3':
            start = int(input("起点编号："))
            n = int(input("订单数量："))
            orders = []
            for i in range(n):
                node = int(input(f"第{i + 1}单配送点："))
                time = int(input(f"第{i + 1}单配送时间（秒）："))
                orders.append((node, time))

            # 获取三种策略结果
            (d_path, d_dist, d_wait, d_delivery), \
                (w_path, w_dist, w_wait, w_delivery), \
                (b_path, b_dist, b_wait, b_delivery) = get_best_delivery(campus, start, orders)

            # 输出结果
            print("\n【最短路径策略】")
            print(f"路径：{'→'.join(map(str, d_path))}")
            print(f"总距离：{d_dist}米 | 总等待时间：{d_wait:.1f}秒 | 总送餐时间：{d_delivery:.1f}秒")

            print("\n【客户等待策略】")
            print(f"路径：{'→'.join(map(str, w_path))}")
            print(f"总距离：{w_dist}米 | 总等待时间：{w_wait:.1f}秒 | 总送餐时间：{w_delivery:.1f}秒")

            print("\n【平衡策略】")
            print(f"路径：{'→'.join(map(str, b_path))}")
            print(f"总距离：{b_dist}米 | 总等待时间：{b_wait:.1f}秒 | 总送餐时间：{b_delivery :.1f}秒")



            # 绘制三个图形
            plt.close('all')
            plot_graph(campus, d_path, "最短路径策略")
            plot_graph(campus, w_path, "最优等待策略")
            plot_graph(campus, b_path, "平衡策略")
            plt.show()

        elif choice == '4':
            print("系统已退出")
            break


if __name__ == "__main__":
    main()