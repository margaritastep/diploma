#Данный модуль содержит функции для построения красивых графиков, сопровождающие этапы моделирования модуля simulation.

import matplotlib.pyplot as plt
import numpy as np

def plot_exposure(x, z, m):
    """
    Визуализирует распределение концентрации светочувствительного компонента
    после этапа экспозиции.

    Параметры:
    x (np.array): 1D-массив координат по оси X (в нанометрах).
    z (np.array): 1D-массив координат по оси Z (в нанометрах).
    m (np.array): 2D-массив концентрации светочувствительного компонента.

    Возвращает:
    None
    """
    # Создание графика с уменьшенным размером
    fig, ax = plt.subplots(figsize=(8, 6))  # Уменьшенный размер для экрана ноутбука
    
    # Отображение данных
    p = ax.imshow(np.transpose(m), cmap='rainbow', aspect='auto', origin='lower',
                  extent=[x[0], x[-1], z[0], z[-1]])
    
    # Настройка осей и заголовка
    ax.set_xlabel('x, нм', fontsize=12)
    ax.set_ylabel('z, нм', fontsize=12)
    ax.set_title(
        'Этап экспозиции\nКонцентрация светочувствительного компонента\n',
        fontsize=14, pad=20
    )
    
    # Добавление цветовой шкалы
    cbar = fig.colorbar(p, ax=ax, location='bottom', pad=0.1)
    cbar.set_label('Концентрация', fontsize=12)
    
    # Отображение графика
    #plt.tight_layout()
    #plt.show()

def plot_PEB_noCAR(x, z, m):
    """
    Визуализирует распределение концентрации светочувствительного компонента (PAC)
    после этапа постэкспозиционной сушки (PEB).

    Параметры:
    x (np.array): 1D-массив координат по оси X (в нанометрах).
    z (np.array): 1D-массив координат по оси Z (в нанометрах).
    m (np.array): 2D-массив концентрации светочувствительного компонента (PAC).

    Возвращает:
    None
    """
    # Создание графика с уменьшенным размером
    fig, ax = plt.subplots(figsize=(8, 6))  # Уменьшенный размер для экрана ноутбука
    
    # Отображение данных
    p = ax.imshow(np.transpose(m), cmap='rainbow', aspect='auto', origin='lower',
                  extent=[x[0], x[-1], z[0], z[-1]])
    
    # Настройка осей и заголовка
    ax.set_xlabel('x, нм', fontsize=12)
    ax.set_ylabel('z, нм', fontsize=12)
    ax.set_title(
        'Этап постэкспозиционной сушки\nКонцентрация светочувствительного компонента (PAC)\n',
        fontsize=14, pad=20
    )
    
    # Добавление цветовой шкалы
    cbar = fig.colorbar(p, ax=ax, location='bottom', pad=0.1)
    cbar.set_label('Концентрация PAC', fontsize=12)
    
    # Отображение графика
    plt.tight_layout()
    plt.show()

def plot_PEB_CAR(x, z, m):
    """
    Визуализирует распределение концентрации блокированных центров полимера
    после этапа постэкспозиционной сушки (PEB) для резиста с химическим усилением.

    Параметры:
    x (np.array): 1D-массив координат по оси X (в нанометрах).
    z (np.array): 1D-массив координат по оси Z (в нанометрах).
    m (np.array): 2D-массив концентрации блокированных центров полимера.

    Возвращает:
    None
    """
    # Создание графика с уменьшенным размером
    fig, ax = plt.subplots(figsize=(8, 6))  # Уменьшенный размер для экрана ноутбука
    
    # Отображение данных
    p = ax.imshow(np.transpose(m), cmap='rainbow', aspect='auto', origin='lower',
                  extent=[x[0], x[-1], z[0], z[-1]])
    
    # Настройка осей и заголовка
    ax.set_xlabel('x, нм', fontsize=12)
    ax.set_ylabel('z, нм', fontsize=12)
    ax.set_title(
        'Этап постэкспозиционной сушки (резист с хим. усилением)\n'
        'Концентрация блокированных центров полимера\n',
        fontsize=14, pad=20
    )
    
    # Добавление цветовой шкалы
    cbar = fig.colorbar(p, ax=ax, location='bottom', pad=0.1)
    cbar.set_label('Концентрация блокированных центров', fontsize=12)
    
    # Отображение графика
    plt.tight_layout()
    plt.show()

def plot_development(x, z, r):
    """
    Визуализирует распределение скоростей проявления.

    Параметры:
    x (np.array): 1D-массив координат по оси X (в нанометрах).
    z (np.array): 1D-массив координат по оси Z (в нанометрах).
    r (np.array): 2D-массив скоростей проявления [нм/с].

    Возвращает:
    None
    """
    # Создание графика с уменьшенным размером
    fig, ax = plt.subplots(figsize=(8, 6))  # Уменьшенный размер для экрана ноутбука
    
    # Отображение данных
    p = ax.imshow(np.transpose(r), cmap='rainbow', aspect='auto', origin='lower',
                  extent=[x[0], x[-1], z[0], z[-1]], vmin=np.min(r), vmax=np.max(r))
    
    # Настройка осей и заголовка
    ax.set_xlabel('x, нм', fontsize=12)
    ax.set_ylabel('z, нм', fontsize=12)
    ax.set_title(
        'Этап проявления\nРаспределение скоростей проявления [нм/с]\n',
        fontsize=14, pad=20
    )
    
    # Добавление цветовой шкалы
    cbar = fig.colorbar(p, ax=ax, location='bottom', pad=0.1)
    cbar.set_label('Скорость проявления [нм/с]', fontsize=12)
    
    # Отображение графика
    plt.tight_layout()
    plt.show()

def discrete_cmap(N, base_cmap=None):
    """
    Создает дискретную цветовую карту с N интервалами на основе указанной цветовой карты.

    Параметры:
    N (int): Количество интервалов (бинов) в дискретной цветовой карте.
    base_cmap (str или matplotlib.colors.Colormap, опционально): Базовая цветовая карта.
        Если строка, то это имя цветовой карты (например, 'viridis', 'rainbow').
        Если None, используется цветовая карта по умолчанию.

    Возвращает:
    matplotlib.colors.Colormap: Дискретная цветовая карта с N интервалами.
    """
    # Получаем базовую цветовую карту
    base = plt.get_cmap(base_cmap)
    
    # Создаем список цветов, равномерно распределенных по базовой карте
    color_list = base(np.linspace(0, 1, N))
    
    # Генерируем имя для новой цветовой карты
    cmap_name = f"{base.name}_discrete_{N}"
    
    # Создаем и возвращаем дискретную цветовую карту
    return base.from_list(cmap_name, color_list, N)

def plot_contour(x, z, c, t, my_cmap):
    """
    Визуализирует контуры проявления и распределение времени проявления.

    Параметры:
    x (np.array): 1D-массив координат по оси X (в нанометрах).
    z (np.array): 1D-массив координат по оси Z (в нанометрах).
    c (np.array): 2D-массив времени проявления [с].
    t (float): Пороговое значение времени для построения контуров.
    my_cmap (matplotlib.colors.Colormap): Цветовая карта для визуализации.

    Возвращает:
    list: 2D-массив контуров проявления.
    """
    # Создание контуров проявления
    contour = []
    for i in range(len(c)):
        tmp = []
        for j in range(len(c[0])):
            if c[i][j] > t:  # Пороговое значение времени
                tmp.append(0)
            else:
                tmp.append(1)
        contour.append(tmp)

    # Создание графика
    fig, ax = plt.subplots(figsize=(8, 6))  # Уменьшенный размер для экрана ноутбука
    
    # Отображение данных времени проявления
    p = ax.imshow(np.transpose(c), cmap=my_cmap, aspect='auto', origin='lower',
                  extent=[x[0], x[-1], z[0], z[-1]])
    
    # Добавление контуров проявления
    cont = ax.contour(np.transpose(contour), origin='lower', colors=['black'],
                      extent=[x[0], x[-1], z[0], z[-1]])
    
    # Настройка осей и заголовка
    ax.set_xlabel('x, нм', fontsize=12)
    ax.set_ylabel('z, нм', fontsize=12)
    ax.set_title(
        'Этап проявления\nКонтуры проявления\n',
        fontsize=14, pad=20
    )
    
    # Добавление цветовой шкалы
    cb = fig.colorbar(p, ax=ax, location='bottom', pad=0.1)
    cb.set_label('Время проявления, с', fontsize=12)
    
    # Отображение графика
    plt.tight_layout()
    plt.show()
    
    return contour