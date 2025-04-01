#Файл содержит парсер для выходных данных из Optolithium. Писался для конкретных задач и специфичного оформления 
#вывода программы, требует доработок в зависимости от вашего формата данных.

import numpy as np

def parser_Optolithium(filename):
    """
    Парсит файл данных Optolithium и извлекает координаты по осям X, Z, а также данные интенсивности.

    Параметры:
    filename (str): Путь к файлу с данными Optolithium.

    Возвращает:
    tuple: Кортеж из трех элементов:
        - x_data (list): Список координат по оси X.
        - z_data (list): Список координат по оси Z.
        - data_parsed (list): Список словарей, содержащих дозу, фокус и данные интенсивности.
    """
    # Чтение файла
    with open(filename, 'r') as file:
        data = ''.join(file.readlines()).split('!!!')

    # Извлечение первого элемента данных
    first_elem = data[1]

    # Парсинг координат по оси X
    x_start = first_elem.find('Image in resist X-Axis data:') + len('Image in resist X-Axis data:')
    x_end = first_elem.find('Image in resist Y-Axis data:')
    x_data = first_elem[x_start:x_end].strip()[1:-2].replace('\n', '').split()
    x_data = [float(i) for i in x_data]

    # Парсинг координат по оси Z
    z_start = first_elem.find('Image in resist Z-Axis data:') + len('Image in resist Z-Axis data:')
    z_end = first_elem.find('Image in resist values:')
    z_data = first_elem[z_start:z_end].strip()[1:-2].replace('\n', '').split()
    z_data = [float(i) for i in z_data]

    # Парсинг данных интенсивности
    data_parsed = []
    for elem in data[1:]:
        # Извлечение данных интенсивности
        i_start = elem.find('[[[')
        i_end = elem.find(']]]') + len(']]]')
        i_data = elem[i_start:i_end].replace('\n', '').replace('[', '').replace(']', '').split()
        i_data = np.array([float(i) for i in i_data]).reshape((len(x_data), len(z_data)))

        # Добавление данных в список
        data_parsed.append(i_data)

    return x_data, z_data, data_parsed