# import scipy.io
# import numpy as np
# import scipy.sparse
# import os
#
#
# def convert_bar31_to_tu(mat_graph_path='bar31.mat', mat_label_path='lbar31.mat', output_dir='BAR31'):
#     # 1. 创建输出目录
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#
#     print(f"正在加载数据...\n图数据: {mat_graph_path}\n标签数据: {mat_label_path}")
#
#     # 2. 加载 .mat 文件
#     try:
#         # 注意：这里假设变量名分别为 'bar31' 和 'lbar31'
#         # 如果实际加载失败，脚本会打印错误信息
#         graphs_data = scipy.io.loadmat(mat_graph_path)['bar31'][0]
#         graph_labels = scipy.io.loadmat(mat_label_path)['lbar31']
#     except Exception as e:
#         print(f"错误: 无法加载文件。请确保文件在当前目录下。详细信息: {e}")
#         return
#
#     # 初始化列表以存储 TU 格式数据
#     all_edges = []
#     graph_indicator = []
#     all_node_labels = []
#
#     # 全局节点索引偏移量 (TU格式要求索引在所有图中累加，且从1开始)
#     node_offset = 1
#
#     print(f"开始处理 {len(graphs_data)} 个图...")
#
#     # 3. 遍历每个图进行处理
#     for graph_id, g in enumerate(graphs_data):
#         # --- A. 提取邻接矩阵 (Adjacency Matrix) ---
#         am = g['am']
#         if not scipy.sparse.issparse(am):
#             am = scipy.sparse.csc_matrix(am)
#
#         # 获取边的坐标 (row, col)
#         rows, cols = am.nonzero()
#
#         # 将局部索引转换为全局索引，并转为 1-based (TU标准)
#         global_rows = rows + node_offset
#         global_cols = cols + node_offset
#
#         # 将边添加到总列表中
#         for r, c in zip(global_rows, global_cols):
#             all_edges.append((r, c))
#
#         num_nodes = am.shape[0]
#
#         # --- B. 生成 Graph Indicator ---
#         # 记录这 num_nodes 个节点都属于当前 graph_id (从1开始计数)
#         graph_indicator.extend([graph_id + 1] * num_nodes)
#
#         # --- C. 提取节点标签 (Node Labels) ---
#         nl_container = g['nl']
#         current_node_labels = np.array([])
#
#         # 处理嵌套结构: void -> array -> data
#         if nl_container.size > 0:
#             try:
#                 inner = nl_container[0, 0]
#                 if isinstance(inner, np.ndarray):
#                     current_node_labels = inner.flatten()
#                 elif isinstance(inner, np.void) and inner.size >= 1:
#                     # 针对某些可能的特殊封装
#                     current_node_labels = inner[0].flatten()
#             except:
#                 pass
#
#         # 异常处理：补零
#         if len(current_node_labels) != num_nodes:
#             # 这种情况如果不常见，打印个警告即可
#             # print(f"警告: 图 {graph_id} 节点数({num_nodes})与标签数({len(current_node_labels)})不匹配，已补零。")
#             current_node_labels = np.zeros(num_nodes, dtype=int)
#
#         all_node_labels.extend(current_node_labels)
#
#         # 更新全局偏移量
#         node_offset += num_nodes
#
#     # 4. 写入文件
#     dataset_name = "BAR31"
#
#     # 写入 A.txt (邻接矩阵)
#     path_A = os.path.join(output_dir, f"{dataset_name}_A.txt")
#     np.savetxt(path_A, all_edges, fmt='%d', delimiter=', ')
#     print(f"生成: {path_A}")
#
#     # 写入 graph_indicator.txt
#     path_gi = os.path.join(output_dir, f"{dataset_name}_graph_indicator.txt")
#     np.savetxt(path_gi, graph_indicator, fmt='%d')
#     print(f"生成: {path_gi}")
#
#     # 写入 graph_labels.txt
#     path_gl = os.path.join(output_dir, f"{dataset_name}_graph_labels.txt")
#     np.savetxt(path_gl, graph_labels, fmt='%d')
#     print(f"生成: {path_gl}")
#
#     # 写入 node_labels.txt
#     if len(all_node_labels) > 0:
#         path_nl = os.path.join(output_dir, f"{dataset_name}_node_labels.txt")
#         np.savetxt(path_nl, all_node_labels, fmt='%d')
#         print(f"生成: {path_nl}")
#
#     print(f"\n转换完成！所有文件已保存在 '{output_dir}' 文件夹中。")
#
#
# # --- 执行 ---
# if __name__ == "__main__":
#     convert_bar31_to_tu()
import scipy.io
import numpy as np
import scipy.sparse
import os


def convert_shock_to_tu(mat_graph_path='Shock.mat', mat_label_path='lShock.mat', output_dir='Shock'):
    # 1. 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"正在加载数据...\n图数据: {mat_graph_path}\n标签数据: {mat_label_path}")

    # 2. 加载 .mat 文件
    try:
        # 变量名通常与文件名一致，但在MATLAB中可能是大小写敏感的
        # 这里根据之前的检查，变量名为 'Shock' 和 'lShock'
        graphs_data = scipy.io.loadmat(mat_graph_path)['Shock'][0]
        graph_labels = scipy.io.loadmat(mat_label_path)['lShock']
    except Exception as e:
        print(f"错误: 无法加载文件。请确保文件在当前目录下。详细信息: {e}")
        return

    # 初始化列表以存储 TU 格式数据
    all_edges = []
    graph_indicator = []
    all_node_labels = []

    # 全局节点索引偏移量 (TU格式要求索引在所有图中累加，且从1开始)
    node_offset = 1

    print(f"开始处理 {len(graphs_data)} 个图...")

    # 3. 遍历每个图进行处理
    for graph_id, g in enumerate(graphs_data):
        # --- A. 提取邻接矩阵 (Adjacency Matrix) ---
        am = g['am']
        if not scipy.sparse.issparse(am):
            am = scipy.sparse.csc_matrix(am)

        # 获取边的坐标 (row, col)
        rows, cols = am.nonzero()

        # 将局部索引转换为全局索引，并转为 1-based (TU标准)
        global_rows = rows + node_offset
        global_cols = cols + node_offset

        # 将边添加到总列表中
        for r, c in zip(global_rows, global_cols):
            all_edges.append((r, c))

        num_nodes = am.shape[0]

        # --- B. 生成 Graph Indicator ---
        # 记录这 num_nodes 个节点都属于当前 graph_id (从1开始计数)
        graph_indicator.extend([graph_id + 1] * num_nodes)

        # --- C. 提取节点标签 (Node Labels) ---
        nl_container = g['nl']
        current_node_labels = np.array([])

        # 处理嵌套结构: void -> array -> data
        if nl_container.size > 0:
            try:
                inner = nl_container[0, 0]
                if isinstance(inner, np.ndarray):
                    current_node_labels = inner.flatten()
                elif isinstance(inner, np.void) and inner.size >= 1:
                    current_node_labels = inner[0].flatten()
            except:
                pass

        # 异常处理：如果有节点标签缺失，补零
        if len(current_node_labels) != num_nodes:
            # print(f"警告: 图 {graph_id} 节点数({num_nodes})与标签数({len(current_node_labels)})不匹配，已补零。")
            current_node_labels = np.zeros(num_nodes, dtype=int)

        all_node_labels.extend(current_node_labels)

        # 更新全局偏移量
        node_offset += num_nodes

    # 4. 写入文件
    dataset_name = "Shock"

    # 写入 A.txt (邻接矩阵)
    path_A = os.path.join(output_dir, f"{dataset_name}_A.txt")
    np.savetxt(path_A, all_edges, fmt='%d', delimiter=', ')
    print(f"生成: {path_A}")

    # 写入 graph_indicator.txt
    path_gi = os.path.join(output_dir, f"{dataset_name}_graph_indicator.txt")
    np.savetxt(path_gi, graph_indicator, fmt='%d')
    print(f"生成: {path_gi}")

    # 写入 graph_labels.txt
    path_gl = os.path.join(output_dir, f"{dataset_name}_graph_labels.txt")
    np.savetxt(path_gl, graph_labels, fmt='%d')
    print(f"生成: {path_gl}")

    # 写入 node_labels.txt
    if len(all_node_labels) > 0:
        path_nl = os.path.join(output_dir, f"{dataset_name}_node_labels.txt")
        np.savetxt(path_nl, all_node_labels, fmt='%d')
        print(f"生成: {path_nl}")

    print(f"\n转换完成！所有文件已保存在 '{output_dir}' 文件夹中。")


# --- 执行 ---
if __name__ == "__main__":
    convert_shock_to_tu()