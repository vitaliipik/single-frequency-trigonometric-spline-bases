import numpy as np
import yfinance as yf

from Python.b_spline import b_spline_curve
from Python.plotting import plot_basis, line_plot_upd, bar_plot, bar_plot_upt
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
    # control_points=duck()
    p = len(control_points) - 1
    # alpha = np.pi / 4
    alpha = np.pi / 2
    alpha = 1
    # alpha = np.pi-1e-10
    # alpha = 1e-10

    def u(p,alpha):

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

    knots = u(p,alpha)

    def stock_graph(control_points,curve_points,x,y,text='Stock Prices with T-2-B-spline Approximation and Volume Traded'):
        # Plotting
        fig, ax_main = plt.subplots(figsize=(20, 15))

        # ax_main.plot(curve_points[:,0], curve_points[:,1], 'ro', label='B-spline Approximation')

        # ax_main.plot(x, data.iloc[:, 3], 'bo-', label='Closing Prices')
        ax_main.plot(curve_points[:, 0], curve_points[:, 1], 'r-', label='T-2-B-spline Approximation')
        # ax_main.plot(control_points[:,0], control_points[:,1], 'bo', label='Closing price')
        # Adding volume traded on the right side
        ax_volume = ax_main.twinx()
        lims = ax_main.get_xlim()
        i = np.where((x > lims[0]) & (x < lims[1]))[0]
        ax_main.set_ylim(y[i].min() - 5, y[i].max() + 5)
        # ax_main.set_ylim(ymin=150)
        # ax_volume.bar(np.arange(len(data)), data.iloc[:,3], color='orange', alpha=0.5, label='Closing Prices')
        ax_main.bar(control_points[:, 0], control_points[:, 1], color='orange', alpha=0.5, label='Closing Prices')
        ax_main.set_ylabel('Closing Prices', color='orange')

        # Setting labels and legend
        ax_main.set_xticks(x[::10])
        ax_main.set_xticklabels(data.index[::10])
        ax_main.set_xlabel('Date')
        ax_main.set_ylabel('Closing Prices')
        ax_main.legend(loc='upper left')
        plt.title(text)
        plt.gcf().autofmt_xdate()
        plt.show()

    write_points(control_points, INPUT_FILE_CONTROL_POINTS)
    write_knots(knots, INPUT_FILE_KNOTS)

    space = 1000
    # space = 100

    # command = ["mpiexec", "-np", str(num_procs), "./cpp/hello-world", str(N)]
    # result = subprocess.run(command, capture_output=True, text=True)
    # command = ["wsl", "./cuda-mch/test", str(degree), str(num_points), INPUT_FILE_CONTROL_POINTS, INPUT_FILE_KNOTS,
    #            OUTPUT_FILE]
    # result=subprocess.run(command, capture_output=True, text=True)
    # print(result)
    # command = ["wsl","./cuda-mch/cuda_t_2_b",str(degree),str(num_points),str(alpha),INPUT_FILE_CONTROL_POINTS,INPUT_FILE_KNOTS,OUTPUT_FILE]

    command = ["wsl", "g++", "-Wall", "-g", "-o", f"{WSL_DIR}/openmp", f"{WSL_DIR}/openmp.cpp",f"-fopenmp"]
    result = subprocess.run(command, capture_output=True, text=True)
    print(result)

    command = ["wsl", "g++", "-Wall", "-g", "-o", f"{WSL_DIR}/openmp-b-spline", f"{WSL_DIR}/b-spline-openmp.cpp",f"-fopenmp"]
    result = subprocess.run(command, capture_output=True, text=True)
    print(result)

    command = ["wsl", "nvcc", f"{WSL_DIR}/cuda-t-2-b.cu", "-o", f"{WSL_DIR}/a.out"]
    result = subprocess.run(command, capture_output=True, text=True)
    print(result)
    # space_array = [1e2, 1e4, 1e6,1e8,1e10]
    space_array = [1e2, 1e4, 1e6]
    # space_array = [1e2, 1e4]
    array_linear=[0.011516,1.68596,156.77,1381.1]
    space_array = [1e2,1e3,1e4]
    space_array = [1e2,1e4,1e6,1e7]
    space_array = [1e2,1e4,1e6]
    space_array = [4e2,4e4,4e6,4e7]
    space_array = [4e2,4e4]
    space_array = [4e6,4e7]
    space_array = [4e6]
    # space_array = [4e3]
    # space_array = [4e6]
    # space_array = [16e2,16e4,16e6,16e7]

    # space_array = [1e2, 1e4, 1e6, 1e7]
    # space_array = [1e2]
    # space_array = [1e2,1e3]
    thread=[2,4,8,16]
    thread=[4]
    result_arr = pd.DataFrame(np.zeros((len(space_array),7)))
    result_arr.columns=["number", "cuda","openmp 2","openmp 4","openmp 8","openmp 16","linear"]
    result_arr.iloc[:, 0] = pd.Series(space_array)
    for index, i in enumerate(space_array):
        alpha = np.pi / 2

        knots = u(p,alpha)
        write_knots(knots, INPUT_FILE_KNOTS)

        command2 = ["wsl", fr"{WSL_DIR}/a.out", str(degree), str(i), str(alpha), str(space),
                    INPUT_FILE_CONTROL_POINTS_WS, INPUT_FILE_KNOTS_WS, OUTPUT_FILE_WS]
        result = subprocess.run(command2, capture_output=True, text=True)
        print(result)
        result_arr.iloc[index, 1] = float(result.stdout)

        curve_points_t_b=np.array(read_points(OUTPUT_FILE))
        curve_points_t_b = curve_points_t_b[~np.all(curve_points_t_b == 0, axis=1)]
        stock_graph(control_points,curve_points_t_b,x,y)

        alpha = 1
        alpha = np.pi / 2

        knots = u(p,alpha)
        write_knots(knots, INPUT_FILE_KNOTS)
        command2 = ["wsl", fr"{WSL_DIR}/openmp-b-spline", str(degree), str(i), str(alpha), str(space),
                                    INPUT_FILE_CONTROL_POINTS_WS, INPUT_FILE_KNOTS_WS, OUTPUT_FILE_WS, str(16)]
        result = subprocess.run(command2, capture_output=True, text=True)
        print(result)
        curve_points_b = np.array(read_points(OUTPUT_FILE))
        curve_points_b= curve_points_b[~np.all(curve_points_b == 0, axis=1)]
        # plot_basis(control_points,curve_points_b,text="B-spline-curve")
        # curve_points =np.array([b_spline_curve(control_points, degree, knots, t) for t in np.linspace(0, 1000,int(i))])
        # curve_points_b = curve_points[~np.all(curve_points == 0, axis=1)]
        # stock_graph(control_points,curve_points_b,x,y,text="b-spline")
        # def add_noise(X_train, noise_level=0.01):
        #     noise = np.random.uniform(low=0, high=noise_level, size=X_train.shape)
        #     return X_train + noise
        # curve_points_b=add_noise(curve_points_t_b)
        # plot_basis(control_points,curve_points_b,text="B-spline-curve")
        error2 = np.linalg.norm(curve_points_t_b[:int(i/4)] - curve_points_b[:int(i/4)], axis=1)  # Error between result1 and result3
        error2 = error2[5:]
        # error3 = np.linalg.norm(curve_points_t_2_b - curve_points_t_b, axis=1)  # Error between result2 and result3
        print(np.sum(error2) / len(error2))
        # Plotting
        plt.figure(figsize=(8, 6))
        fig, ax = plt.subplots()
        labels = [item.get_text() for item in ax.get_yticklabels()]
        labels=[""]*11
        labels[0]="0"
        labels[2]="0.2"
        labels[4]="0.4"
        labels[6]="0.6"
        labels[8]="0.8"
        labels[10]="1"
        labels = [""] * 11
        labels[0] = "0"
        labels[3] = "0.1"

        labels[6] = "0.2"

        labels[10] = "0.3"

        ax.set_yticklabels(labels)
        # plt.plot(error1, label='Error between Algorithm 1 and 2')
        plt.plot(error2, label='Error between scipy and t-2-b-spline')
        # plt.plot(error3, label='Error between Algorithm 2 and 3')
        plt.xlabel('Point Index')
        plt.ylabel('Error')
        plt.title('Error Comparison')
        plt.legend()
        plt.grid(True)
        plt.show()

        # curve_points_b = curve_points_b[~np.all(curve_points_b == 0, axis=1)]
        # for j_i,j in enumerate(thread):
        #
        #
        #     command2 = ["wsl", fr"{WSL_DIR}/openmp", str(degree), str(i), str(alpha), str(space),
        #                 INPUT_FILE_CONTROL_POINTS_WS, INPUT_FILE_KNOTS_WS, OUTPUT_FILE_WS, str(j)]
        #     result = subprocess.run(command2, capture_output=True, text=True)
        #     print(result)
        #     result_arr.iloc[index, j_i+2] = float(result.stdout)



        # command2 = ["wsl", fr"{WSL_DIR}/cuda_t_2_b_l", str(degree), str(i), str(alpha), str(space),
        #             INPUT_FILE_CONTROL_POINTS_WS, INPUT_FILE_KNOTS_WS, OUTPUT_FILE_WS]
        # result = subprocess.run(command2, capture_output=True, text=True)
        # result_arr.iloc[index, 6] = float(result.stdout)
        result_arr.iloc[index, 6] = array_linear[index]
        alpha = np.pi / 2

        knots = u(p,alpha)
        write_knots(knots, INPUT_FILE_KNOTS)
        command2 = ["wsl", fr"{WSL_DIR}/a.out", str(degree), str(i), str(alpha), str(space),
                    INPUT_FILE_CONTROL_POINTS_WS, INPUT_FILE_KNOTS_WS, OUTPUT_FILE_WS]
        result = subprocess.run(command2, capture_output=True, text=True)
        curve_points = np.array(read_points(OUTPUT_FILE))
        curve_points = curve_points[~np.all(curve_points == 0, axis=1)]

        # line_plot_upd(result_arr)
        # bar_plot(speedups,space_array,y_tick=True)
        # bar_plot(effect,space_array,text="effectivity")
        # # bar_plot_upt(speedups,space_array)
        # line_plot(result_arr)
        stock_graph(control_points,curve_points,x,y)
        print(result)





    print(result_arr[list(result_arr.columns[1:])])
    # speedups = result_arr['linear']/(result_arr[list(result_arr.columns[1:])])
    speedups =(1/result_arr[list(result_arr.columns[1:])]).div(1/result_arr['linear'],axis=0)
    speedups=speedups.iloc[:,:-1]
    # efficiencies = speedups / best_implementation

    effect=speedups.iloc[:,1:].div(thread,axis=1)
    print(result_arr.to_string())
    print(speedups.to_string())
    print(effect.to_string())

    curve_points = np.array(read_points(OUTPUT_FILE))
    curve_points = curve_points[~np.all(curve_points == 0, axis=1)]

    # line_plot_upd(result_arr)
    # bar_plot(speedups,space_array,y_tick=True)
    # bar_plot(effect,space_array,text="effectivity")
    # # bar_plot_upt(speedups,space_array)
    # line_plot(result_arr)
    stock_graph(control_points,curve_points,x,y)


if __name__ == '__main__':
    main()
    #'(wsl nvcc /home/vitalii/cuda-mch//cuda-t-2-b.cu -o /home/vitalii/cuda-mch//a.out ) -and (wsl /home/vitalii/cuda-mch//a.out 2 1000 0.7853981633974483 ~/cuda-mch/control_points.txt ~/cuda-mch/knots.txt ~/cuda-mch/result.txt )'
    # (wsl nvcc /home/vitalii/cuda-mch//cuda-t-2-b.cu -o /home/vitalii/cuda-mch//a.out ) && (wsl /home/vitalii/cuda-mch//a.out 2 1000 0.7853981633974483 \\wsl.localhost\Ubuntu\home\vitalii\cuda-mch\control_points.txt \\wsl.localhost\Ubuntu\home\vitalii\cuda-mch\knots.txt \\wsl.localhost\Ubuntu\home\vitalii\cuda-mch\result.txt )
#(wsl nvcc /home/vitalii/cuda-mch//cuda-t-2-b.cu ) && (wsl /home/vitalii/cuda-mch//a.out 2 1000 0.7853981633974483 \\wsl.localhost\Ubuntu\home\vitalii\cuda-mch\control_points.txt \\wsl.localhost\Ubuntu\home\vitalii\cuda-mch\knots.txt \\wsl.localhost\Ubuntu\home\vitalii\cuda-mch\result.txt )
# wsl /home/vitalii/cuda-mch//a.out 2 1000 0.7853981633974483 \home\vitalii\cuda-mch\control_points.txt \home\vitalii\cuda-mch\knots.txt \home\vitalii\cuda-mch\result.txt
# (wsl nvcc /home/vitalii/cuda-mch//cuda-t-2-b.cu ) -and (wsl /home/vitalii/cuda-mch//a.out 2 1000 0.7853981633974483 \\wsl.localhost\Ubuntu\home\vitalii\cuda-mch\control_points.txt \\wsl.localhost\Ubuntu\home\vitalii\cuda-mch\knots.txt \\wsl.localhost\Ubuntu\home\vitalii\cuda-mch\result.txt )
