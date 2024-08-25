import math
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import tabulate as tl

#Mittelwert
def mean(l: list) -> int:
    return sum(l) / len(l)


def list_varianz(l: list) -> list:
    return [(i - mean(l)) ** 2 for i in l]

#Summe der einzelnen Varianzen (benötigt für die Standardabweichung)
def sum_varianz(x):
    return sum(list_varianz(x))

#Summe der x_ix_i
def get_xx(x) -> int:
    xx = 0
    for i in range(len(x)):
        xx += x[i] ** 2
    return xx

#Summe der x_iy_i
def get_xy(x: list, y: list) -> int:
    xy = 0
    for i in range(len(x)):
        # if (x[i] is not None) & (y[i] is not None): # Attempt at making it None-Proof
        xy += x[i] * y[i]
    return xy

#Summe der di2
def sum_d_i2(x, y):
    b = get_b(x, y)
    a = get_a(x, y)

    d = 0
    for i in range(len(x)):
        d += (y[i] - b * x[i] - a) ** 2
    return d

#Standardabweichung des Mittelwerts vom wahren Wert
def get_s_x(x: list) -> float:
    return math.sqrt(1 / (len(x) * (len(x) - 1)) * sum_varianz(x))

#Parameter b für die lineare Regression
def get_b(x: list, y: list) -> float:
    xy = get_xy(x, y)
    xx = get_xx(x)
    return (xy - len(x) * mean(x) * mean(y)) / (xx - len(x) * mean(x) ** 2)

#Parameter a für die lineare Regression
def get_a(x, y):
    return mean(y) - get_b(x, y) * mean(x)

#Standardabweichung von b
def get_s_b(x, y):
    if len(x) <= 2:
        return 9999999999999
    else:
        s = (1 / (len(x) - 2)) * ((sum_d_i2(x, y)) / (sum_varianz(x)))
        return math.sqrt(s)

#Standardabweichung von a
def get_s_a(x, y):
    if len(x) <= 2:
        return 9999999999999
    else:
        xx = get_xx(x)
        s = xx / len(x) * get_s_b(x, y) ** 2
        return math.sqrt(s)

#neu eingefügt: Standardabweichung des y-Wertes der linearen Regression
def get_s_y(x: list, y: list) -> float:
    s = (sum_d_i2(x,y))/(len(x) - 2)
    return math.sqrt(s)

#neu eingefügt: Zusammengesetzten Messfehler aus statistischem Fehler und Messfehler des Messgeräts berechnen
def Gesamtfehler(Gerätfehler, Stabw):
    Gesfehler= math.sqrt(Stabw**2+Gerätfehler**2)
    return Gesfehler

#gibt x-Wert mit Messfehler raus
def wert_x(x: list, name: str = None) -> tuple:
    x_mean = mean(x)
    s_x_mean = get_s_x(x)
    perc = s_x_mean / x_mean if x_mean != 0 else 9999999999
    if name is not None:
        print(f"{name: .3e}_mean = {x_mean: .3e} +- {s_x_mean: .3e}    (+- {perc: .3e})")
        print()
    return x_mean, s_x_mean

#lineare Regression
def wert_xy(x: list | np.ndarray, y: np.ndarray, name: str = None) -> tuple:
    if name is not None:
        print(f"{name}:")

    # Convert to np.ndrarray if not already so.
    if type(x) == list:
        x = np.array(x)
    if type(y) == list:
        y = np.array(y)

    # Remove all None entries from the given data
    pos1 = x == None  # '==' is correct. Is would check if array is None not whether elements of array are None.
    pos2 = y == None
    pos = np.logical_or(pos1, pos2)
    x = x[~pos]
    y = y[~pos]

    # do calcultions
    b = get_b(x, y)
    s_b = get_s_b(x, y)
    b_perc = s_b / b if b != 0 else 999999999999999

    a = get_a(x, y)
    s_a = get_s_a(x, y)
    a_perc = s_a / a if a != 0 else 999999999999999

    s_y = get_s_y(x, y)

    if name is not None:
        print(f" -  b = {b: .3e} +- {s_b: .3e}  (+- {b_perc: .3e})")
        print(f" -  a = {a: .3e} +- {s_a: .3e}  (+- {a_perc: .3e})")
        print()

    return b, a, s_b, s_a, s_y

#auch lineare Regression aber in fancy
def get_trendlinie(x: list, y: list):
    """

    :param x:
    :param y:
    :return:
    """
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    return p(x)


def graph(x: list | np.ndarray, y: list | tuple | np.ndarray, trendlinie: bool = False, title: str = None,
          xlabel: str = None, multiple=False,
          ylabel: str = None, xlog: bool = False, ylog: bool = False, graph="scatter", xlim=None, ylim=None) -> None:
    """

    :param multiple:
    :param x: data for x-axis
    :param y: data for y-axis
    :param trendlinie:
    :param title: graph title
    :param xlabel: Label for y-axis
    :param ylabel: Label for y-axis
    :param xlog: Determines wether the x-axis should be plotted logarithmic.
    :param ylog: Determines wether the y-axis should be plotted logarithmic.
    :param xlim: limits for x-axis
    :param ylim: limits for y-axis
    :return: None
    """
    fig, ax = plt.subplots(layout='constrained')
    plt.plot
    if (type(y) is tuple) or multiple:
        for y_i in y:
            if y_i[2] == "scatter":
                ax.scatter(x, y_i[0], linewidths=1, label=y_i[1])
            elif y_i[2] == "plot":
                ax.plot(x, y_i[0], linewidth=1, label=y_i[1])
            if trendlinie:
                b, a, s_b, s_a = wert_xy(x, y_i[0])
                ax.plot(x, [(b * i + a) for i in x], color="grey", linestyle="dashed",
                        label=rf"{y_i[1]}: Trendlinie: {b: .2e}$*x + {a: .2e}$")
        ax.legend()

    else:
        if graph == "scatter":
            ax.scatter(x, y, linewidths=2)
        elif graph == "plot":
            ax.plot(x, y, linewidth=1.5)
        if trendlinie:
            b, a, s_b, s_a = wert_xy(x, y)
            ax.plot(x, [(b * i + a) for i in x], color="grey", linestyle="dashed",
                    label=rf"Trendlinie: {b: .2e}$*x + {a: .2e}$")
            ax.legend()

    if xlog:
        ax.set_xscale("log")
    if ylog:
        ax.set_yscale("log")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid()
    #mache mal ein paar eigene dazu:
    if xlim is not None:
     ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    plt.show()


def table(data: list | tuple | np.ndarray, rowLabels: list = None, colLabels: list = None, transpose=True) -> None:
    """
    Generell function makin' some sleek lookin' tables.

    :param rowLabels: The text of the row header cells.
    :param colLabels: The text of the column header cells.
    :param data: The texts to place into the table cells.
    :param transpose: Should the data array be transposed
    :return: None
    """

    data = np.asarray(data)

    fig, ax = plt.subplots()
    if transpose:
        data = data.transpose()

    ax.table(cellText=data, rowLabels=rowLabels, colLabels=colLabels, loc='center', cellLoc='center', rowLoc='center')

    ax.axis('off')

    plt.show()


#Tabellen
def latex(data: list, headers: list):
    """Funktion für LaTeX Booktabs-Tabellen.

    :param data: table data as a list (of lists or arrays)(e.g. [x,y])
    :param headers: headers of the table as a list of strings (e.g. ["x","y"])"""
    tab = np.array(data).transpose()
    table = tl.tabulate(tab, headers=headers, tablefmt="latex_booktabs", numalign="center", stralign="center")
    return table

