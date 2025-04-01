#Данный модуль подготавливался для конкретной задачи, не тестировался на структурах отличного от неё вида.
#Скорее всего, модуль требует доработки.
#Именно поэтому он вынесен в отдельный модуль, а не стал частью модуля simulation.

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
from sklearn.metrics import mean_squared_error

def parabola(x, a, b, c):
    return (a * x + b) * x + c
    
def poly3(x, a, b, c, d):
    return a * x ** 3 + b * x**2 + c * x  + d

def hyperbola(x, a, b, c):
    return a / (x - b) + c

class Metrics:
    def __init__(self, up_level=0.9, down_level=0.1, mode='linear'):
        """
        Создаёт экземпляр класса Metrics.

        Параметры:
        up_level (float): Уровень от 0 до 1. Определяет уровень по оси z,
        выше которого точки не будут учитываться (шероховатость верхнего края).
        down_level (float): Уровень от 0 до 1. Определяет уровень по оси z,
        ниже которого точки не будут учитываться.
        mode (str): "linear"|"parabola"|"poly3"|"hyperbola" - функция для фита грани.

        Возвращает:
        Экземпляр класса
        """
        self.up_level = up_level
        self.down_level = down_level
        self.mode = mode

    def find_contour(self, conts, x_in, z_in, x_stop_l, x_stop_r, t):
        """
        Находит контуры левой и правой грани исследуемой структуры.

        Параметры:
        conts (2D array): 2D-массив контуров.
        x_in (1D array): 1D-массив координат по оси X (в нанометрах).
        z_in (1D array): 1D-массив координат по оси Z (в нанометрах).
        x_stop_l (int): Точка, ограничивающая левый край структуры (чтобы не захватить доп. структуры).
        x_stop_r (int): Точка, ограничивающая правый край структуры.
        t (int): Время проявления (в с).
        Возвращает:
        res_l (1D array): 1D-массив точек (x, y) левой грани.
        res_r (1D array): 1D-массив точек (x, y) правой грани.
        """
        step_x = int(np.abs(x_in[1] - x_in[0]))
        step_z = int(np.abs(z_in[1] - z_in[0]))
        conts_l = conts[x_in.index(x_stop_l):(len(x_in) // 2)]
        conts_r = conts[(len(x_in) // 2):x_in.index(x_stop_r)]
        x_center = x_in[len(x_in) // 2]
        res_l = []
        for i in range(len(conts_l[0])):
            for j in range(len(conts_l) - 1):
                if (conts_l[j][i] >= t and conts_l[j + 1][i] <= t) or (conts_l[j][i] <= t and conts_l[j + 1][i] >= t):
                    a = np.abs(conts_l[j][i] - t)
                    b = np.abs(conts_l[j + 1][i] - t)
                    x = x_stop_l + (j + a / (a + b)) * step_x
                    z = i * step_z
                    res_l.append((x, z))
        res_r = []
        for i in range(len(conts_r[0])):
            for j in range(len(conts_r) - 1):
                if (conts_r[j][i] >= t and conts_r[j + 1][i] <= t) or (conts_r[j][i] <= t and conts_r[j + 1][i] >= t):
                    a = np.abs(conts_r[j][i] - t)
                    b = np.abs(conts_r[j + 1][i] - t)
                    x = x_center + (j + a / (a + b)) * step_x
                    z = i * step_z
                    res_r.append((x, z))
        return(res_l, res_r)

    def find_edge(self, x, z, z_in):
        """
        Находит координаты грани, приближенные соответствующим методом интерполяции.

        Параметры:
        x (1D array): 1D-массив координат грани по оси X (в нанометрах).
        z (1D array): 1D-массив координатграни по оси Z (в нанометрах).
        z_in (1D array): Исходная сетка по оси Z (могут отличаться с z при неполной печати).

        Возвращает:
        my_model (1D array): массив "новых" координат грани по оси X, вычисленных по модели интерполяции.
        z (1D array): массив координат грани по оси Y.
        rms (float): Среднеквадратичное отклонение интерполяции грани.
        """
        if len(z) == 0:
            return None, None, None
        z_max = max(z)
        if z_max / max(z_in) < 0.70:
            #Считаем структуру меньше 70 процентов высоты непропечатанной
            return None, None, None
        elif min(z) > 10:
            #Такая структура не пропечаталась
            return None, None, None
        else:
            z_max_new = z_max  * self.up_level
            z_min_new = z_max * self.down_level
            x_new, z_new = [], []
            x_new_new, z_new_new = [], []
            for i in range(len(z)):
                if z[i] <= z_max_new and z[i] >= z_min_new:
                    x_new.append(x[i])
                    z_new.append(z[i])
            if self.mode == 'linear':
                slope, intercept, r, p, std_err = stats.linregress(z_new, x_new)
                mymodel = [slope * i + intercept for i in z]
            elif self.mode == 'parabola':
                params, _ = curve_fit(parabola, z_new, x_new)
                mymodel = [parabola(elem, *params) for elem in z]
            elif self.mode == 'poly3':
                params, _ = curve_fit(poly3, z_new, x_new)
                mymodel = [poly3(elem, *params) for elem in z]
            elif self.mode == 'hyperbola':
                params, _ = curve_fit(hyperbola, z_new, x_new)
                mymodel = [hyperbola(elem, *params) for elem in z]
            else:
                print('Такая интерполяция не предусмотрена')
                return None, None, None
            rms = np.sqrt(mean_squared_error(x, mymodel))
            return mymodel, z, rms

    def find_metrics(self, left_points, right_points, z_data, CD_x = (None, None), plot=True):
        """
        Находит величину CD структуры. Строит график структуры.

        Параметры:
        left_points (1D array): Координаты левой интерполированной грани.
        right_points (1D array): Координаты правой интерполированной грани.
        z_data (1D array): Координаты сетки по оси Z в нм.
        CD_x (float, float): Пара значений, отвечающих за линию маски слева и 
        справа от центра грани, нужна для графика.
        plot (bool): Если True, рисует график.

        Возвращает:
        CD (float): Напечатанный размер структуры по уровню z=0.
        rms_l (float): Среднеквадратичное отклонение интерполяции левой грани.
        rms_r (float): Среднеквадратичное отклонение интерполяции правой грани.
        """
        x_l_in = [elem[0] for elem in left_points]
        z_l_in = [elem[1] for elem in left_points]
        x_r_in = [elem[0] for elem in right_points]
        z_r_in = [elem[1] for elem in right_points]
        x_l, z_l, rms_l = self.find_edge(x_l_in, z_l_in, z_data)
        x_r, z_r, rms_r = self.find_edge(x_r_in, z_r_in, z_data)
        if x_l == None or x_r == None:
            return 0, None, None
        try:
            indl_05 = np.where(np.isclose(z_l, 0))[0][0]
            indr_05 = np.where(np.isclose(z_r, 0))[0][0]   
        except:
            return 0, None, None    

        CD = (x_r[indr_05] - x_l[indl_05])
        if plot:
            fig, axs = plt.subplots(1, 1, figsize=(12,12))
            axs.plot(x_l_in, z_l_in, color='cornflowerblue', label='моделирование')
            axs.plot(x_r_in, z_r_in, color='cornflowerblue')
            axs.fill_betweenx(z_l_in, x_l_in, color='cornflowerblue')
            axs.fill_betweenx(z_r_in, 0, x_r_in, color='cornflowerblue')
            axs.plot(x_l, z_l, color='r', linewidth=1.5, label='интерполяция')
            axs.plot(x_r, z_r, color='r', linewidth=1.5)
            axs.set_ylim(0)
            axs.set_xlabel('x, нм')
            axs.set_ylabel('z, нм')
            axs.set_title('Центральная структура')
            if CD_x[0] != None:
                axs.axvline(x=CD_x[0], color='k', linestyle='dashed', label='маска')
            if CD_x[1] != None:
                axs.axvline(x=CD_x[1], color='k', linestyle='dashed')
            axs.legend()
            axs.set_aspect(0.2) #позволяет хорошо увидеть неравномерность края
        return CD, rms_l, rms_r