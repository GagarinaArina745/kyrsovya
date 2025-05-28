import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.optimize import linear_sum_assignment

st.title("Задача о назначениях — ввод матрицы через текст")

default_text = "4 1 3\n2 0 5\n3 2 2"
text = st.text_area("Введите матрицу стоимости (через пробелы, строки через переносы):", default_text, height=150)

try:
    # Парсим введённый текст в numpy-массив
    rows = text.strip().split("\n")
    matrix = [list(map(float, row.strip().split())) for row in rows]
    cost_matrix = np.array(matrix)
except Exception as e:
    st.error(f"Ошибка при обработке матрицы: {e}")
    st.stop()

if st.button("Рассчитать оптимальное назначение"):
    max_value = cost_matrix.max()
    transformed_matrix = max_value - cost_matrix

    row_ind, col_ind = linear_sum_assignment(transformed_matrix)
    max_total_value = cost_matrix[row_ind, col_ind].sum()

    st.write("Оптимальное назначение:", list(zip(row_ind, col_ind)))
    st.write("Максимальная сумма стоимости:", max_total_value)

    G = nx.Graph()
    left_nodes = [f"L{i}" for i in range(cost_matrix.shape[0])]
    right_nodes = [f"R{j}" for j in range(cost_matrix.shape[1])]
    G.add_nodes_from(left_nodes, bipartite=0)
    G.add_nodes_from(right_nodes, bipartite=1)

    for i in range(cost_matrix.shape[0]):
        for j in range(cost_matrix.shape[1]):
            G.add_edge(left_nodes[i], right_nodes[j], weight=cost_matrix[i, j])

    pos = dict()
    pos.update((node, (0, index)) for index, node in enumerate(left_nodes))
    pos.update((node, (1, index)) for index, node in enumerate(right_nodes))

    plt.figure(figsize=(8, 5))
    nx.draw_networkx_edges(G, pos, alpha=0.3)
    nx.draw_networkx_nodes(G, pos, nodelist=left_nodes, node_color='skyblue', node_size=500)
    nx.draw_networkx_nodes(G, pos, nodelist=right_nodes, node_color='lightgreen', node_size=500)
    nx.draw_networkx_labels(G, pos)

    edge_labels = {(left_nodes[i], right_nodes[j]): cost_matrix[i, j]
                   for i in range(cost_matrix.shape[0]) for j in range(cost_matrix.shape[1])}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    opt_edges = [(left_nodes[i], right_nodes[j]) for i, j in zip(row_ind, col_ind)]
    nx.draw_networkx_edges(G, pos, edgelist=opt_edges, edge_color='red', width=2)

    plt.axis('off')
    plt.title('Оптимальное назначение')
    plt.tight_layout()

    st.pyplot(plt.gcf())
