import re
import pandas as pd     # 数据处理库，用于读取Excel文件
from geopy.distance import geodesic  # 地理距离计算库
import networkx as nx
import matplotlib.pyplot as plt     # 数据可视化库
from itertools import permutations      # 排列组合生成器
from PIL import Image

# 读取PNG图片文件并显示
image_path = '校园地图.jpg'
image = Image.open(image_path)
image.show()

class CampusGraph:
    def __init__(self):
        self.att = {}
        self.latitude = {}
        self.longitude = {}
        self.edges = {}

def getCampus(g, filename):
    data = pd.read_excel(filename)
    for index, row in data.iterrows():      # 遍历每一行数据
        name = re.sub(r'[^\w\s]', '', row["配送点名称"])
        g.att[index + 1] = name
        g.latitude[index + 1] = row["纬度"]
        g.longitude[index + 1] = row["经度"]

def calcDistance(g):
    for i in range(1, len(g.att) + 1):
        for j in range(i + 1, len(g.att) + 1):      # 避免重复计算
            loc1 = (g.latitude[i], g.longitude[i])
            loc2 = (g.latitude[j], g.longitude[j])
            distance = int(geodesic(loc1, loc2).meters)     # 计算真实地理距离（米）
            g.edges[(i, j)] = distance      # 存储双向距离
            g.edges[(j, i)] = distance  # 双向距离

def get_distance(g, node1, node2):
    return g.edges[(node1, node2)]

def get_shortest_path(campus, start_id, end_id):
    G = nx.Graph()
    for i in campus.att:
        G.add_node(i, pos=(campus.longitude[i], campus.latitude[i]))
    # 添加所有边（带权重）
    for i in campus.edges:
        G.add_edge(i[0], i[1], weight=campus.edges[i])
    # 计算最短路径
    shortest_path_nodes = nx.shortest_path(G, source=start_id, target=end_id, weight='weight')
    shortest_distance = nx.shortest_path_length(G, source=start_id, target=end_id, weight='weight')
    return shortest_path_nodes, shortest_distance

def plot_graph(campus, path_nodes):
    G = nx.Graph()
    pos = {node_id: (campus.longitude[node_id], campus.latitude[node_id]) for node_id in campus.att}
    for edge, weight in campus.edges.items():
        G.add_edge(edge[0], edge[1], weight=weight)
    # 绘制基础图
    nx.draw(G, pos, with_labels=True, node_size=700, font_size=8, node_color='skyblue', width=0.5)
    nx.draw_networkx_nodes(G, pos, nodelist=path_nodes, node_color='red', node_size=700)
    edges = [(path_nodes[i], path_nodes[i+1]) for i in range(len(path_nodes)-1)]
    nx.draw_networkx_edges(G, pos, edgelist=edges, width=2, edge_color='blue')
    plt.title("外卖配送最短路径")
    plt.axis('equal')
    plt.show()

def get_best_delivery_path(campus, start, delivery_nodes):
    best_distance = float('inf')
    best_permutation = []       # 最优路径排列
    # 遍历所有可能的配送顺序
    for permutation in permutations(delivery_nodes):
        current_distance = 0
        # 起点到第一个配送点
        current_distance += get_distance(campus, start, permutation[0])
        # 中间路径距离累加
        for i in range(len(permutation)-1):
            current_distance += get_distance(campus, permutation[i], permutation[i+1])
        # 更新最优解
        if current_distance < best_distance:
            best_distance = current_distance
            best_permutation = permutation
    # 返回完整路径（起点+配送顺序）
    full_path = [start] + list(best_permutation)
    return full_path, best_distance

def save_distance_matrix(campus, filename):
    """生成并保存距离矩阵到Excel文件"""
    node_ids = sorted(campus.att.keys())    # 排序的节点ID列表
    # 创建空的DataFrame
    distance_df = pd.DataFrame(index=node_ids, columns=node_ids)
    # 填充距离数据
    for i in node_ids:
        for j in node_ids:
            if i == j:
                distance_df.loc[i, j] = 0       # 对角线为0
            else:
                distance_df.loc[i, j] = campus.edges[(i, j)]    # 实际距离
    # 保存到Excel
    distance_df.to_excel(filename)
    print(f"距离矩阵已保存至 {filename}")

def main():
    campus = CampusGraph()
    filename = "武汉纺织大学经纬度.xlsx"
    getCampus(campus, filename)
    calcDistance(campus)

    print("============== 外卖配送最短路径规划系统 ==============")
    print("1. 查看所有配送点信息")
    print("2. 两点间最短路径查询")
    print("3. 外卖配送最短路径规划")
    print("4. 输出距离矩阵到Excel")
    print("5. 退出系统")

    while True:
        choice = input("请选择操作：")
        if choice == '1':
            print("配送点列表:")
            for node_id, name in campus.att.items():
                print(f"编号: {node_id}, 名称: {name}")
        elif choice == '2':
            start = int(input("请输入起点编号："))
            end = int(input("请输入终点编号："))
            path, distance = get_shortest_path(campus, start, end)
            print(f"最短路径：{' -> '.join(map(str, path))}, 距离：{distance}米")
            plot_graph(campus, path)
        elif choice == '3':
            start = int(input("请输入起点编号："))
            num_orders = int(input("请输入订单数量："))
            delivery_nodes = []
            for i in range(num_orders):
                node_id = int(input(f"第 {i+1} 个配送点编号："))
                delivery_nodes.append(node_id)
            path, distance = get_best_delivery_path(campus, start, delivery_nodes)
            print(f"最佳路径：{' -> '.join(map(str, path))}, 总距离：{distance}米")
            plot_graph(campus, path)
        elif choice == '4':
            output_filename = input("请输入要保存的Excel文件名（例如：距离矩阵.xlsx）：")
            save_distance_matrix(campus, output_filename)
        elif choice == '5':
            print("感谢使用！")
            break
        else:
            print("无效选项，请重新输入。")

if __name__ == "__main__":
    main()