import numpy as np
import yfinance as yf

from Python.plotting import plot_basis
from t_2_b_spline_iterative import t_2_b_spline_curve




import re
import subprocess
import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from plotting import plot_basis, line_plot
from preset_figure import spiral, duck

WSL_DIR = r"/home/vitalii/cuda-mch/"
INPUT_FILE_CONTROL_POINTS = r"\\wsl.localhost\Ubuntu\home\vitalii\cuda-mch\control_points.txt"
INPUT_FILE_CONTROL_POINTS_WS = r"\home\vitalii\cuda-mch\control_points.txt"
INPUT_FILE_CONTROL_POINTS_WS = r"~/cuda-mch/control_points.txt"
INPUT_FILE_KNOTS = r"\\wsl.localhost\Ubuntu\home\vitalii\cuda-mch\knots.txt"
INPUT_FILE_KNOTS_WS = r"\home\vitalii\cuda-mch\knots.txt"
INPUT_FILE_KNOTS_WS = r"~/cuda-mch/knots.txt"
OUTPUT_FILE = r"\\wsl.localhost\Ubuntu\home\vitalii\cuda-mch\result.txt"
OUTPUT_FILE_WS = r"~/cuda-mch/result.txt"


def write_points(points, filename):
    with open(filename, 'w') as file:
        file.write(str(len(points)) + "\n")

        for point in points:
            if len(point) == 3:
                file.write(f"{point[0]} {point[1]} {point[2]}\n")
            else:
                file.write(f"{point[0]} {point[1]}\n")


def write_knots(knots, filename):
    with open(filename, 'w') as file:
        file.write(str(len(knots)) + "\n")
        for point in knots:
            file.write(f"{point}\n")


def read_points(filename):
    points = []
    with open(filename, 'r') as file:
        for line in file:
            values = line.split()
            points.append((float(values[0]), float(values[1])))

    return points


def main():
    data = yf.download("AAPL", period="1y", interval="1d")
    x = range(len(data))
    # x = data.index
    # y = data.iloc[:, -1].astype('float64')
    y = data.iloc[:, 2].astype('float64')
    control_points = np.array([[x_i, y_i] for x_i, y_i in zip(x, y)])
    degree = 2
    p = len(control_points) - 1
    # alpha = np.pi / 4
    alpha = np.pi / 2
    # alpha = np.pi-1e-10
    # alpha = 1e-10

    def u():

        knots = [0, 0, 0]
        for i in range(3, p + 1):
            knots += [(i - 2) * alpha]
        knots += [(p - 1) * alpha] * 3
        # knots+=knots[0:2]

        return knots

    def pi():

        knots = []
        for i in range(p + 3):
            knots += [i * alpha]
        return knots

    knots = u()



    write_points(control_points, INPUT_FILE_CONTROL_POINTS)
    write_knots(knots, INPUT_FILE_KNOTS)

    space = 1000

    # command = ["mpiexec", "-np", str(num_procs), "./cpp/hello-world", str(N)]
    # result = subprocess.run(command, capture_output=True, text=True)
    # command = ["wsl", "./cuda-mch/test", str(degree), str(num_points), INPUT_FILE_CONTROL_POINTS, INPUT_FILE_KNOTS,
    #            OUTPUT_FILE]
    # result=subprocess.run(command, capture_output=True, text=True)
    # print(result)
    # command = ["wsl","./cuda-mch/cuda_t_2_b",str(degree),str(num_points),str(alpha),INPUT_FILE_CONTROL_POINTS,INPUT_FILE_KNOTS,OUTPUT_FILE]

    command = ["wsl", "nvcc", f"{WSL_DIR}/cuda-t-2-b.cu", "-o", f"{WSL_DIR}/a.out"]
    result = subprocess.run(command, capture_output=True, text=True)
    print(result)
    # space_array = [1e2, 1e4, 1e6,1e8,1e10]
    space_array = [1e2, 1e4, 1e6]
    # space_array = [1e2]
    result_arr = pd.DataFrame(np.zeros((len(space_array), 2)))
    result_arr.columns=["number", "cuda"]
    result_arr.iloc[:, 0] = pd.Series(space_array)
    for index, i in enumerate(space_array):


        command2 = ["wsl", fr"{WSL_DIR}/a.out", str(degree), str(i), str(alpha), str(space),
                    INPUT_FILE_CONTROL_POINTS_WS, INPUT_FILE_KNOTS_WS, OUTPUT_FILE_WS]
        result = subprocess.run(command2, capture_output=True, text=True)
        result_arr.iloc[index, 1] = float(result.stdout)
        print(result)



        print(result)


    print(result_arr.to_string())

    curve_points = np.array(read_points(OUTPUT_FILE))
    curve_points = curve_points[~np.all(curve_points == 0, axis=1)]

    # Plotting
    fig, ax_main = plt.subplots(figsize=(25, 6))

    # ax_main.plot(curve_points[:,0], curve_points[:,1], 'ro', label='B-spline Approximation')

    # ax_main.plot(x, data.iloc[:, 3], 'bo-', label='Closing Prices')
    ax_main.plot(curve_points[:,0], curve_points[:,1], 'r-', label='B-spline Approximation')
    ax_main.plot(control_points[:,0], control_points[:,1], 'bo', label='Closing price')
    # Adding volume traded on the right side
    ax_volume = ax_main.twinx()
    lims = ax_main.get_xlim()
    i = np.where((x > lims[0]) & (x < lims[1]))[0]
    ax_main.set_ylim(y[i].min()-5, y[i].max()+5)
    # ax_main.set_ylim(ymin=150)
    # ax_volume.bar(np.arange(len(data)), data.iloc[:,3], color='orange', alpha=0.5, label='Closing Prices')
    ax_main.bar(control_points[:,0], data.iloc[:,3], color='orange', alpha=0.5, label='Closing Prices')
    ax_main.set_ylabel('Closing Prices', color='orange')

    # Setting labels and legend
    ax_main.set_xticks(x[::10])
    ax_main.set_xticklabels(data.index[::10])
    ax_main.set_xlabel('Date')
    ax_main.set_ylabel('Closing Prices')
    ax_main.legend(loc='upper left')
    plt.title('Stock Prices with B-spline Approximation and Volume Traded')
    plt.gcf().autofmt_xdate()
    plt.show()
    # # Plotting
    # fig, ax_main = plt.subplots(figsize=(25, 6))
    # ax_main.plot(curve_points[:,0], curve_points[:,1], 'r-', label='B-spline Approximation')
    # ax_main.plot(control_points[:,0], control_points[:,1], 'bo', label='B-spline Approximation')
    # # ax_main.plot(curve_points[:,0], curve_points[:,1], 'ro', label='B-spline Approximation')
    #
    # # ax_main.plot(x, data.iloc[:, 3], 'bo-', label='Closing Prices')
    #
    # # Adding volume traded on the right side
    # ax_volume = ax_main.twinx()
    # ax_volume.bar(np.arange(len(data)), y, color='orange', alpha=0.5, label='Volume Traded')
    # ax_volume.set_ylabel('Volume Traded', color='orange')
    #
    # # Setting labels and legend
    # ax_main.set_xticks(x[::10])
    # ax_main.set_xticklabels(data.index[::10])
    # ax_main.set_xlabel('Date')
    # ax_main.set_ylabel('Closing Prices')
    # ax_main.legend(loc='upper left')
    # plt.title('Stock Prices with B-spline Approximation and Volume Traded')
    # plt.gcf().autofmt_xdate()
    # plt.show()
    # plot_basis(control_points,curve_points)
    # plt.show()


if __name__ == '__main__':
    main()
    #'(wsl nvcc /home/vitalii/cuda-mch//cuda-t-2-b.cu -o /home/vitalii/cuda-mch//a.out ) -and (wsl /home/vitalii/cuda-mch//a.out 2 1000 0.7853981633974483 ~/cuda-mch/control_points.txt ~/cuda-mch/knots.txt ~/cuda-mch/result.txt )'
    # (wsl nvcc /home/vitalii/cuda-mch//cuda-t-2-b.cu -o /home/vitalii/cuda-mch//a.out ) && (wsl /home/vitalii/cuda-mch//a.out 2 1000 0.7853981633974483 \\wsl.localhost\Ubuntu\home\vitalii\cuda-mch\control_points.txt \\wsl.localhost\Ubuntu\home\vitalii\cuda-mch\knots.txt \\wsl.localhost\Ubuntu\home\vitalii\cuda-mch\result.txt )
#(wsl nvcc /home/vitalii/cuda-mch//cuda-t-2-b.cu ) && (wsl /home/vitalii/cuda-mch//a.out 2 1000 0.7853981633974483 \\wsl.localhost\Ubuntu\home\vitalii\cuda-mch\control_points.txt \\wsl.localhost\Ubuntu\home\vitalii\cuda-mch\knots.txt \\wsl.localhost\Ubuntu\home\vitalii\cuda-mch\result.txt )
# wsl /home/vitalii/cuda-mch//a.out 2 1000 0.7853981633974483 \home\vitalii\cuda-mch\control_points.txt \home\vitalii\cuda-mch\knots.txt \home\vitalii\cuda-mch\result.txt
# (wsl nvcc /home/vitalii/cuda-mch//cuda-t-2-b.cu ) -and (wsl /home/vitalii/cuda-mch//a.out 2 1000 0.7853981633974483 \\wsl.localhost\Ubuntu\home\vitalii\cuda-mch\control_points.txt \\wsl.localhost\Ubuntu\home\vitalii\cuda-mch\knots.txt \\wsl.localhost\Ubuntu\home\vitalii\cuda-mch\result.txt )
