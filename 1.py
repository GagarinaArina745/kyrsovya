import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.optimize import linear_sum_assignment
import time

st.title("Задача о назначениях")

# Ввод матрицы стоимости
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

# Выбор режима: минимизация или максимизация
mode = st.radio("Выберите режим расчёта:", ("Минимум стоимости", "Максимум стоимости"))

# Инициализация переменных для вывода
result_text = ""
total_value = None
calc_time = None 

if st.button("Рассчитать оптимальное назначение"):
    start_time = time.time() 
    if mode == "Минимум стоимости":
     # Для минимизации используем исходную матрицу
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        total_value = cost_matrix[row_ind, col_ind].sum()
        mode_text = "Минимальная сумма стоимости"
    else:
        # Для максимизации преобразуем матрицу: max - стоимость
        max_value = cost_matrix.max()
        transformed_matrix = max_value - cost_matrix
        row_ind, col_ind = linear_sum_assignment(transformed_matrix)
        total_value = cost_matrix[row_ind, col_ind].sum()
        mode_text = "Максимальная сумма стоимости"

    end_time = time.time()  # конец таймера
    elapsed_time = end_time - start_time

# Вывод результатов
    result_text += f"Результаты ({mode_text}):\n"
    for i, j in zip(row_ind, col_ind):
        result_text += f"Работник {i+1} → Профессия {j+1} (Стоимость: {cost_matrix[i,j]})\n"
    result_text += f"Общая сумма стоимости: {total_value}\n"
    result_text += f"Время расчёта: {calc_time:.4f} секунд\n"

start_time_graph = time.time()
    # Создаем граф для визуализации
G = nx.Graph()

    # Назначим работников и профессии как кружки
workers = [f"Работник {i+1}" for i in range(cost_matrix.shape[0])]
professions = [f"Профессия {j+1}" for j in range(cost_matrix.shape[1])]

G.add_nodes_from(workers, bipartite=0)
G.add_nodes_from(professions, bipartite=1)

    # Добавляем все возможные ребра с весами
for i in range(cost_matrix.shape[0]):
    for j in range(cost_matrix.shape[1]):
        G.add_edge(workers[i], professions[j], weight=cost_matrix[i,j])

    # Расположение узлов: слева работники, справа профессии
pos = dict()
pos.update((node, (0, i)) for i, node in enumerate(workers))
pos.update((node, (1, j)) for j, node in enumerate(professions))

# Расчет размеров фигуры в зависимости от размера матрицы
num_workers = cost_matrix.shape[0]
num_professions = cost_matrix.shape[1]
width = max(10, num_workers * 2)
height = max(8, num_professions * 2)

plt.figure(figsize=(width, height))

    # Рисуем все ребра полупрозрачными линиями
nx.draw_networkx_edges(G, pos, alpha=0.3)
    
    # Узлы работников и профессий
nx.draw_networkx_nodes(G, pos, nodelist=workers, node_color='skyblue', node_size=500)
nx.draw_networkx_nodes(G, pos, nodelist=professions, node_color='lightgreen', node_size=500)
    
    # Метки узлов
nx.draw_networkx_labels(G, pos)

    # Вписываем веса на ребра
edge_labels = {(workers[i], professions[j]): f"{cost_matrix[i,j]}" 
              for i in range(cost_matrix.shape[0]) 
              for j in range(cost_matrix.shape[1])}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    # Выделяем оптимальные ребра красным цветом
optimal_edges = [(workers[i], professions[j]) for i,j in zip(row_ind,col_ind)]
nx.draw_networkx_edges(G, pos, edgelist=optimal_edges, edge_color='red', width=2)

plt.axis('off')
    
st.pyplot(plt)
end_time_graph = time.time()
graph_time = end_time_graph - start_time_graph

   # Вывод времени построения графа и отображения
st.write(f"Время построения и отображения графа: {graph_time:.4f} секунд")
# Вывод результатов после вычислений (если есть)
if result_text:
   st.text(result_text)