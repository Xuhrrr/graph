import csv
from operator import truediv
from symbol import sync_comp_for
from collections import deque
from collections import defaultdict
import semver
import networkx as nx
from tqdm import tqdm
import os
import pickle
import numpy as np
from scipy.sparse import csr_matrix

import matplotlib.pyplot as plt
from packaging.version import Version
from datetime import datetime
csv.field_size_limit(10**7)

base_dir = os.path.dirname(os.path.abspath(__file__))
dump_dir = os.path.join(base_dir, "..", "data")

# Initialize directed graph
graph = nx.DiGraph()

# Create ID to name mapping for crates
crate_id_to_name = {}
# Create version_id to crate_id mapping
version_id_to_crate_id = {}
crate_latest_versions = {}  # crate_id -> (version_num, version_id)

# Load crate ID -> name mapping
crate_file = os.path.join(dump_dir, "crates.csv")
# Load version ID -> crate ID mapping
version_file = os.path.join(dump_dir, "versions.csv")
dep_file = os.path.join(dump_dir, "dependencies.csv")
#print(crate_file,version_file,dep_file)
def parse_rust_version(version_str):
    """解析 Rust 风格的版本号"""
    try:
        # 尝试标准 SemVer 解析
        return semver.VersionInfo.parse(version_str)
    except ValueError:
        # 处理特殊格式
        base_version = version_str.split('-')[0]  # 取主版本部分
        try:
            return semver.VersionInfo.parse(base_version)
        except ValueError:
            # 最终回退方案
            return semver.VersionInfo(major=0, minor=0, patch=0)


def compare_rust_versions(ver1, ver2):
    """
    比较两个 Rust 风格版本号的新旧
    返回:
     - 1 如果 ver1 > ver2
     - 0 如果 ver1 == ver2
     - -1 如果 ver1 < ver2
    """
    v1 = parse_rust_version(ver1)
    v2 = parse_rust_version(ver2)

    if v1 > v2:
        return 1
    else:
        return 0


def get_timestamp(time_str):
    """
    鲁棒的时间戳解析函数，支持以下格式：
    1. 带微秒和时区：'2023-02-24 21:31:37.123456+00'
    2. 带时区无微秒：'2023-02-24 21:31:37+00'
    3. 带微秒无时区：'2023-02-24 21:31:37.123456'
    4. 无微秒无时区：'2023-02-24 21:31:37'
    """
    formats = [
        "%Y-%m-%d %H:%M:%S.%f%z",  # 带微秒和时区
        "%Y-%m-%d %H:%M:%S%z",  # 带时区无微秒
        "%Y-%m-%d %H:%M:%S.%f",  # 带微秒无时区
        "%Y-%m-%d %H:%M:%S"  # 纯日期时间
    ]

    for fmt in formats:
        try:
            dt = datetime.strptime(time_str, fmt)
            return dt.timestamp()
        except ValueError:
            continue

    # 终极回退方案：尝试清理字符串
    try:
        cleaned = time_str.split('+')[0].split('.')[0]
        dt = datetime.strptime(cleaned, "%Y-%m-%d %H:%M:%S")
        return dt.timestamp()
    except Exception as e:
        print(f"Failed to parse timestamp '{time_str}': {e}")
        return 0  # 或 raise ValueError("Unparseable timestamp")
def compare_datetime(t1,t2) -> bool:
    if t1 > t2:
        return True
    else:
        return False
def load_info():
    for row in tqdm(read_csv_from_file(crate_file), desc="Loading crates.csv"):
        crate_id = int(row["id"])
        name = row["name"]
        crate_id_to_name[crate_id] = name

    for row in tqdm(read_csv_from_file(version_file), desc="Loading latest versions"):
        try:
            version_id = int(row["id"])
            crate_id = int(row["crate_id"])
            version_num = row["num"]
            crate_time = get_timestamp(row["created_at"])
            downloads = int(row["downloads"])
            if not version_num:
                #print("yes")
                continue
            #print(f"{crate_id}:{version_id}\n")
            current_version = parse_rust_version(version_num)
            prev_entry = crate_latest_versions.get(crate_id)
            version_id_to_crate_id[version_id] = crate_id
            if not prev_entry or compare_rust_versions(version_num,prev_entry[0]):
                #if prev_entry and compare_rust_versions(version_num,prev_entry[0]):
                #    print(f"{crate_id}:{version_id}\n")
                if prev_entry:
                    crate_latest_versions[crate_id] = (version_num, version_id,crate_time,downloads+prev_entry[3])
                else:
                    crate_latest_versions[crate_id] = (version_num, version_id, crate_time, downloads)
        except Exception as e:
            print(f"Error processing version_id={version_id}, crate_id={crate_id}, version='{version_num}': {str(e)}")
            continue
    # 构造 version_id -> crate_id 映射，只保留最新版本
    print(f"\nTotal latest versions loaded: {len(crate_latest_versions)}")


def load_graph_from_pkl(file_path):
    """从.pkl文件加载图对象"""
    with open(file_path, 'rb') as f:  # 注意必须是二进制模式 'rb'
        graph = pickle.load(f)

    # 验证是否为networkx图对象
    if not isinstance(graph, (nx.Graph, nx.DiGraph)):
        raise ValueError("Loaded object is not a networkx graph")

    return graph
def store_graph(graph):
    with open(os.path.join(base_dir, "..", "dag.pkl"), "wb") as f:
        pickle.dump(graph, f, protocol=pickle.HIGHEST_PROTOCOL)
# Helper to read CSV from extracted folder (as generator)
def read_csv_from_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            yield row

def create_graph_from_file():
    for row in tqdm(read_csv_from_file(crate_file), desc="Loading crates.csv"):
        crate_id = int(row["id"])
        name = row["name"]
        graph.add_node(crate_id, name=name)
    # Load dependencies and build edges
    for row in tqdm(read_csv_from_file(dep_file), desc="Loading dependencies.csv"):
        try:
            if row['kind'] != '0': continue
            from_version_id = int(row["version_id"])
            to_crate_id = int(row["crate_id"])
            #print(f"{from_version_id}:{to_crate_id}\n")
            if from_version_id in version_id_to_crate_id and to_crate_id in crate_id_to_name:
                #print(f"{from_version_id}:{to_crate_id}\n")
                from_crate_id = version_id_to_crate_id[from_version_id]
                if from_crate_id in crate_id_to_name:
                    graph.add_edge(from_crate_id, to_crate_id)
            # break
        except Exception:
            continue
    # Basic stats
    print(f"Graph has {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")


def break_circles(graph):
    print("breaking circles")
    """
    删除长度为1和2的循环依赖
    :param graph: 有向图（nx.DiGraph）
    :param crate_latest_versions: {crate_id: (version_num, version_id, crate_time)}
    :return: 修改后的图
    """
    modified = False

    # 1. 删除自循环（长度1）
    self_loops = list(nx.selfloop_edges(graph))
    if self_loops:
        print(f"删除 {len(self_loops)} 个自循环")
        graph.remove_edges_from(self_loops)
        modified = True

    # 2. 删除长度为2的环
    cycles_2 = list(nx.simple_cycles(graph,length_bound=2))
    print(f"删除 {len(cycles_2)} 个大小为2的环")
    for a, b in tqdm(cycles_2, desc="Processing cycles"):
        # 获取创建时间
        time_a = crate_latest_versions[a][2]  # crate_time
        time_b = crate_latest_versions[b][2]

        # 删除较旧节点指向较新节点的边
        if time_a < time_b:
            if graph.has_edge(a, b):
                print(f"删除边: {a} → {b} (依据创建时间 {time_a} < {time_b})")
                graph.remove_edge(a, b)
                modified = True
        else:
            if graph.has_edge(b, a):
                print(f"删除边: {b} → {a} (依据创建时间 {time_b} < {time_a})")
                graph.remove_edge(b, a)
                modified = True

    if not modified:
        print("未检测到需处理的循环依赖")
    return graph
def topo(graph):
    """
    带环检测和破环
    :param graph: 有向图
    :return:
        list：拓扑排序结果
        groph:DAG
    """
    working_graph = graph.copy()
    in_degree = {node:working_graph.in_degree(node) for node in working_graph.nodes()}
    queue = deque([node for node in working_graph.nodes() if in_degree[node] == 0])
    edges_to_remove = []
    topo_order = []
    nums = working_graph.number_of_nodes()
    print(nums)
    while nums > 0:
        if len(queue) == 0: #入度为0的点没有
            min_node = min([(node,deg) for node,deg in in_degree.items() if deg > 0],key=lambda x: x[1])[0]
            in_edges = list(working_graph.in_edges(min_node))#指向该点的边
            edges_to_remove.extend(in_edges) #准备移除的边
            for u,v in in_edges:
                working_graph.remove_edge(u,v)
                in_degree[v] -= 1
                if v != min_node:
                    print("yes")
                    break
            in_degree[min_node] = 0
            queue.append(min_node)
        else:
            node = queue.popleft()
            topo_order.append(node)
            successors = list(working_graph.successors(node))
            working_graph.remove_node(node)
            in_degree.pop(node,None)
            for nbr in successors:
                #working_graph.remove_edge(node,nbr)
                in_degree[nbr] -= 1
                if in_degree[nbr] == 0:
                    queue.append(nbr)
            nums -= 1
            #working_graph.remove_node(node)
            #del in_degree[node]
    return topo_order, edges_to_remove

def page_rank(graph,alpha=0.85,tol=1e-8): #tol为收敛域值
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("输入非DAG")
    longest_path_length = len(nx.dag_longest_path(graph)) - 1 #路径长度-1
    max_iter = longest_path_length + 5
    crate_ids = list(graph.nodes())
    crate_index = {crate_id:i for i,crate_id in enumerate(crate_ids)}
    n = graph.number_of_nodes()
    #print(n)
    #CSR格式稀疏矩阵
    row_ind,col_ind = [],[]
    out_degree = defaultdict(int)

    #出度，构建坐标列表
    for src,dst in graph.edges():
        row_ind.append(crate_index[dst])
        col_ind.append(crate_index[src])
        out_degree[src] += 1

    data = np.ones(len(graph.edges()))
    M = csr_matrix((data,(row_ind,col_ind)),shape=(n,n))
    out_degree_array = np.array([out_degree.get(crate_id,1) for crate_id in crate_ids])
    M = M.multiply(1 / out_degree_array[:,np.newaxis]).T
    downloads = np.array([crate_latest_versions.get(crate_id,(None,None,None,0))[3] for crate_id in crate_ids])
    #print(max_iter)
    if np.all(downloads == 0):
        R = np.ones(n) / n
    else:

        min_dl = downloads.min()
        max_dl = downloads.max()
        print(min_dl, max_dl)
        range_dl = max_dl - min_dl
        if range_dl > 0:
            #min-max归一化
            normalized_downloads = (downloads - min_dl) / range_dl  # 归一化到[0,1]
            R = normalized_downloads / normalized_downloads.sum()  # 保持概率和=1
        else:
            R = np.ones(n) / n
        '''
        mean_dl = downloads.mean()
        relative_dl = downloads / mean_dl
        R = relative_dl / relative_dl.sum()  # 标准化为概率分布
        '''
    teleport = (1 - alpha) / n
    for _ in range(max_iter):
        new_R = alpha * M.dot(R) + teleport
        if np.linalg.norm(new_R - R ,1) < tol:
            break
        R = new_R
    return {crate_id:R[crate_index[crate_id]] for crate_id in crate_ids}

def get_dependencies_with_names(crate_name):
    """
    Given a crate name, return a list of (dependency_id, dependency_name) tuples
    that represent its direct dependencies.
    """
    for node_id, name in crate_id_to_name.items():
        if name == crate_name and graph.has_node(node_id):
            dependencies = []
            for dep_id in graph.successors(node_id):
                dep_name = graph.nodes[dep_id].get("name", "(unknown)")
                dependencies.append((dep_id, dep_name))
            return dependencies
    return None  # crate not found
def remove_edges(graph,remove_list):
    copy_graph = graph.copy()
    copy_graph.remove_edges_from(remove_list)
    return copy_graph
if __name__ == "__main__":
    load_info()
    #graph = load_graph_from_pkl(os.path.join(base_dir, "..", "rust_dependency_graph_with_ids_and_names.pkl"))
    create_graph_from_file()
    #break_circles(graph)#这个地方用的是人家论文中提及的先删去自环以及环为2的情况
    order, remove_list = topo(graph)
    if(len(order) == graph.number_of_nodes()):
        #print("yes")
        new_graph = remove_edges(graph,remove_list)
        store_graph(new_graph)
    graph = load_graph_from_pkl(os.path.join(base_dir, "..", "dag.pkl"))
    pr_results = page_rank(graph)
    for crate_id, score in sorted(pr_results.items(), key=lambda x: -x[1])[:20]:
        name = graph.nodes[crate_id]['name']
        downloads = crate_latest_versions.get(crate_id, (None, None, None, 0))[3]
        print(f"{name:<30} (id={crate_id:<6}): PR={score:.10f}, Downloads={downloads}")
    '''
    is_dag = nx.is_directed_acyclic_graph(graph)
    try:
        topo_order = list(nx.topological_sort(graph))
        has_cycle = False
    except nx.NetworkXUnfeasible:
        has_cycle = True
    print(has_cycle,is_dag)
    '''
    #if(len(order) == graph.number_of_nodes()):
    #    print("yes")
    #    new_graph = remove_edges(graph,remove_list)
    #    store_graph(new_graph)
    #has_cycle = not nx.is_directed_acyclic_graph(graph)
    #store_graph(graph)
    '''
    crate = "base64"
    deps = get_dependencies_with_names(crate)
    if deps is None:
        print(f"Crate '{crate}' not found.")
    elif not deps:
        print(f"Crate '{crate}' has no dependencies.")
    else:
        print(f"{crate} depends on {len(deps)} crates:")
        for dep_id, dep_name in deps:
            print(f"  - {dep_name} (ID: {dep_id})")
    '''


