
import numpy as np
import pykonal
from pde import FieldCollection, PDEBase, UnitGrid, ScalarField
import pde
from scipy.ndimage import gaussian_filter

class Exposure:
    def __init__(self, dill_C):
        """
        Инициализация класса Exposure.

        Параметры:
        dill_C (float): Константа Дилла C.
        """
        self.dill_C = dill_C

    def simulate(self, intensity_in, dose):
        """
        Симулирует этап экспозиции.

        Параметры:
        intensity_in (np.array): Распределение интенсивности (image in resist).
        dose (float): Доза в мДж/см^2.

        Возвращает:
        np.array: Распределение концентрации светочувствительного компонента.
        """
        return np.exp(-1 * self.dill_C * intensity_in * dose)


class PEB:
    def __init__(self):
        pass


class PEB_noCAR(PEB):
    def __init__(self, sigma):
        """
        Инициализация класса PEB_noCAR.

        Параметры:
        sigma (float): Параметр sigma диффузии.
        """
        super().__init__()
        self.sigma = sigma

    def simulate(self, step_x, step_z, m_concentration):
        """
        Вычисляет результат по формуле step_develop_2.

        Параметры:
        m_concentration (np.array): Распределение концентрации светочувств. компонента.
        step_x (float): Шаг по оси x (нм). Шаг, на котором измерялась m_concentration.
        step_z (float): Шаг по оси z (нм).

        Возвращает:
        np.array: Распределение концентрации светочувствительного компонента после PEB.
        """
        return gaussian_filter(m_concentration, (self.sigma/step_x, self.sigma/step_z))


class MyPDE(PDEBase):

    def __init__(self, D_h=12, D_q=12, K_a=0.01, K_q=0.000):
        super().__init__()
        
        self.bc = {"x": "periodic", "y" : "neumann"}
        self.dh = D_h
        self.dq = D_q
        self.ka = K_a
        self.kq = K_q

    def evolution_rate(self, state, t=0):
        h, q, w = state

        h_t = self.dh * h.laplace(self.bc) - self.kq * h * q
        q_t = self.dq * q.laplace(self.bc) - self.kq * h * q
        w_t = - self.ka * h * w

        return FieldCollection([h_t, q_t, w_t])


class PEB_CAR(PEB):
    def __init__(self, D_h, D_q, K_a, K_q, step_x, step_z):
        """
        Инициализация класса PEB_CAR.

        Параметры:
        D_h (float): Коэффициент диффузии кислоты [нм^2/с].
        D_q (float): Коэффициент диффузии основания [нм^2/с].
        K_a (float): Параметр диффузии [нм^2/с].
        K_q (float): Параметр диффузии [нм^2/с].
        step_x (int): Шаг сетки по x.
        step_z (int): Шаг сетки по z.
        """
        fix = step_x * step_z
        self.PDE = MyPDE(D_h/fix, D_q/fix, K_a/fix, K_q/fix)

    def simulate(self, q0, w0, t, x, z, m_concentration):
        """
        Вычисляет результат по формуле step_develop_2.

        Параметры:
        q0 (np.array): Начальная относительная концентрация основания.
        w0 (np.array): Начальная относительная концентрация деблок. центров полимера.
        t (int): Время сушки.
        x (np.array): Сетка по x.
        z (np.array): Сетка по z.
        m_concentration (np.array): Распределение концентрации светочувств. компонента.

        Возвращает:
        np.array: Распределение концентрации светочувствительного компонента после PEB.
        """
        grid = UnitGrid([len(x), len(z)], periodic=[True, False])
        
        h0 = ScalarField(grid, (np.full((len(x), len(z)), 1.0) - m_concentration).tolist())
        q0 = ScalarField(grid, q0)
        w0 = ScalarField(grid, w0)
        
        state = FieldCollection([h0, q0, w0])
        return self.PDE.solve(state, t_range=t, dt=1, tracker=None)[2].data


class Development:
    def __init__(self):
        pass


class DevelopmentMack(Development):
    def __init__(self, rmin, rmax, n, mth):
        """
        Инициализация класса DevelopmentMack.

        Параметры:
        rmin (float): Минимальное значение скорости.
        rmax (float): Максимальное значение скорости.
        n (float): Параметр n.
        mth (float): Пороговое значение концентрации mth.
        """
        self.rmin = rmin
        self.rmax = rmax
        self.n = n
        self.mth = mth

    def simulate(self, m_in):
        """
        Вычисляет распределение скоростей проявления по модели Мака.

        Параметры:
        m_in (np.array): Распределение концентрации светочувствительного компонента (не-CAR)
        или деблокированных центров полимера (CAR) после этапа PEB.

        Возвращает:
        np.array: Распределение скоростей проявления.
        """
        a = (self.n + 1) * (1 - self.mth)**self.n / (self.n - 1)
        return self.rmax * ((a + 1) * (1 - m_in)**self.n) / (a + (1 - m_in)**self.n) + self.rmin


class DevelopmentEnhanced(Development):
    def __init__(self, rmin, rmax, rresin, n, l):
        """
        Инициализация класса DevelopmentEnhanced.

        Параметры:
        rmin (float): Минимальное значение скорости.
        rmax (float): Максимальное значение скорости.
        rresin (float): Параметр R_resin.
        n (float): Параметр n.
        l (float): Параметр l.
        """
        self.rmin = rmin
        self.rmax = rmax
        self.rresin = rresin
        self.n = n
        self.l = l

    def simulate(self, m_in):
        """
        Вычисляет распределение скоростей проявления по улучшенной модели Мака (Enhanced Mack).

        Параметры:
        m_in (np.array): Распределение концентрации светочувствительного компонента (не-CAR)
        или деблокированных центров полимера (CAR) после этапа PEB.

        Возвращает:
        np.array: Распределение скоростей проявления.
        """
        kinh = self.rresin / self.rmin - 1
        kenh = self.rmax / self.rresin - 1
        return self.rresin * ((1 + kenh * (1 - m_in)**self.n) / (1 + kinh * m_in**self.l))


class DevelopmentNotch(Development):
    def __init__(self, rmin, rmax, n, mth, nnotch):
        """
        Инициализация класса DevelopmentNotch.

        Параметры:
        rmin (float): Минимальное значение скорости.
        rmax (float): Максимальное значение скорости.
        n (float): Параметр n.
        mth (float): Пороговое значение концентрации mth.
        nnotch (float): Параметр nnotch, определяющий форму выемки.
        """
        self.rmin = rmin
        self.rmax = rmax
        self.n = n
        self.mth = mth
        self.nnotch = nnotch

    def simulate(self, m_in):
        """
        Вычисляет распределение скоростей проявления по модели с выемкой(Notch).

        Параметры:
        m_in (np.array): Распределение концентрации светочувствительного компонента (не-CAR)
        или деблокированных центров полимера (CAR) после этапа PEB.

        Возвращает:
        np.array: Распределение скоростей проявления.
        """
        a = (self.nnotch + 1) * (1 - self.mth)**self.nnotch / (self.nnotch - 1)
        return self.rmax * (1 - m_in)**self.n * ((a + 1) * (1 - m_in)**self.nnotch) \
               / (a + (1 - m_in)**self.nnotch) + self.rmin


class DevelopmentNotchEnhanced(Development):
    def __init__(self, rmin, rmax, n, mth, nnotch, s):
        """
        Инициализация класса DevelopmentNotchEnhanced.

        Параметры:
        rmin (float): Минимальное значение скорости.
        rmax (float): Максимальное значение скорости.
        n (float): Параметр n.
        mth (float): Пороговое значение концентрации mth.
        nnotch (float): Параметр nnotch, определяющий форму выемки.
        s (float): Параметр s.
        """
        self.rmin = rmin
        self.rmax = rmax
        self.n = n
        self.mth = mth
        self.nnotch = nnotch
        self.s = s

    def simulate(self, m_in):
        """
        Вычисляет распределение скоростей проявления по улучшенной модели с выемкой (Enhanced Notch).

        Параметры:
        m_in (np.array): Распределение концентрации светочувствительного компонента (не-CAR)
        или деблокированных центров полимера (CAR) после этапа PEB.

        Возвращает:
        np.array: Распределение скоростей проявления.
        """
        a = (self.nnotch + 1) * (1 - self.mth)**self.nnotch / (self.nnotch - 1)
        term1 = self.rmax * (1 - m_in)**self.n * ((a + 1) * (1 - m_in)**self.nnotch) \
                / (a + (1 - m_in)**self.nnotch)
        term2 = self.rmin * (self.s / self.s**m_in) \
                * (1 - ((a + 1) * (1 - m_in)**self.nnotch) / (a + (1 - m_in)**self.nnotch))**2
        return term1 + term2


class DevelopmentNotchInhibition(Development):
    def __init__(self, rmin, rmax, n, mth, nnotch, delta):
        """
        Инициализация класса DevelopmentNotchInhibition.

        Параметры:
        rmin (float): Минимальное значение скорости.
        rmax (float): Максимальное значение скорости.
        n (float): Параметр n.
        mth (float): Пороговое значение концентрации mth.
        nnotch (float): Параметр nnotch, определяющий форму выемки.
        delta (float): Параметр delta, определяющий глубину ингибирования.
        """
        self.rmin = rmin
        self.rmax = rmax
        self.n = n
        self.mth = mth
        self.nnotch = nnotch
        self.delta = delta

    def simulate(self, m_in, z_in):
        """
        Вычисляет распределение скоростей проявления по модели с выемкой и учетом поверхностного ингибирования.

        Параметры:
        m_in (np.array): 2D-массив распределения концентрации светочувствительного компонента (не-CAR)
                         или деблокированных центров полимера (CAR) после этапа PEB.
        z_in (np.array): 1D-массив глубин (координаты по оси z).

        Возвращает:
        np.array: 2D-массив распределения скоростей проявления.
        """
        a = (self.nnotch + 1) * (1 - self.mth)**self.nnotch / (self.nnotch - 1)
        res = []
        for i in range(len(m_in)):
            res_tmp = []
            for j in range(len(m_in[0])):
                # Основная формула с учетом ингибирования
                tmp = (self.rmax * (1 - m_in[i][j])**self.n * (a + 1) * (1 - m_in[i][j])**self.nnotch \
                       / (a + (1 - m_in[i][j])**self.nnotch) + self.rmin) \
                       * np.exp(-z_in[len(m_in[0]) - j - 1] / self.delta)
                res_tmp.append(tmp)
            res.append(res_tmp)
        return np.array(res)


class TimeContours:
    def __init__(self):
        """
        Инициализация класса TimeContours.
        """
        pass

    def simulate(self, x_in, z_in, rates):
        """
        Вычисляет временные контуры на основе распределения скоростей проявления.

        Параметры:
        x_in (np.array): 1D-массив координат по оси x.
        z_in (np.array): 1D-массив координат по оси z.
        rates (np.array): 2D-массив распределения скоростей проявления.

        Возвращает:
        np.array: 2D-массив временных контуров.
        """
        # Вычисляем шаги по осям x и z
        step_x = int(np.abs(x_in[1] - x_in[0]))
        step_z = int(np.abs(z_in[1] - z_in[0]))
        
        # Длины массивов координат
        len_x = len(x_in)
        len_z = len(z_in)
        
        # Инициализация решателя Эйконала
        solver = pykonal.EikonalSolver(coord_sys="cartesian")
        
        # Установка минимальных координат и шагов
        solver.velocity.min_coords = int(x_in[0]), int(z_in[-1]), 0
        solver.velocity.node_intervals = step_x, step_z, 1
        solver.velocity.npts = len_x, len_z, 1
        
        # Задание скоростей (расширяем rates до 3D для совместимости с pykonal)
        solver.velocity.values = np.expand_dims(rates, axis=2)
        
        # Установка начальных условий (источники на поверхности z = z_in[-1])
        for i in range(0, len_x, step_x):
            src_idx = i, len_z - 1, 0
            solver.traveltime.values[src_idx] = 0  # Время в источнике равно 0
            solver.unknown[src_idx] = False  # Узел считается известным
            solver.trial.push(*src_idx)  # Добавляем узел в очередь для обработки
        
        # Решение уравнения Эйконала
        solver.solve()
        
        # Возвращаем 2D-массив временных контуров (убираем третье измерение)
        return solver.traveltime.values.squeeze()

