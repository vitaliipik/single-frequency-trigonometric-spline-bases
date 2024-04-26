import re
import subprocess

import numpy as np
from plotting import plot_basis
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
        file.write(str(len(points))+"\n")

        for point in points:
            if len(point) == 3:
                file.write(f"{point[0]} {point[1]} {point[2]}\n")
            else:
                file.write(f"{point[0]} {point[1]}\n")

def write_knots(knots, filename):
    with open(filename, 'w') as file:
        file.write(str(len(knots))+"\n")
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


    control_points=duck()


    degree =2
    alpha=np.pi/4
    # alpha=1.33

    # Generate points along the curve
    num_points = 1000

    p = len(control_points)
    knots = [0, 0, 0]
    for i in range(3, p):
        knots += [(i - 2) * alpha]
    knots += [(p - 1) * alpha] * 3
    write_points(control_points, INPUT_FILE_CONTROL_POINTS)
    write_knots(knots, INPUT_FILE_KNOTS)


    space=80


    # command = ["mpiexec", "-np", str(num_procs), "./cpp/hello-world", str(N)]
    # result = subprocess.run(command, capture_output=True, text=True)
    # command = ["wsl", "./cuda-mch/test", str(degree), str(num_points), INPUT_FILE_CONTROL_POINTS, INPUT_FILE_KNOTS,
    #            OUTPUT_FILE]
    # result=subprocess.run(command, capture_output=True, text=True)
    # print(result)
    # command = ["wsl","./cuda-mch/cuda_t_2_b",str(degree),str(num_points),str(alpha),INPUT_FILE_CONTROL_POINTS,INPUT_FILE_KNOTS,OUTPUT_FILE]



    command = ["wsl","nvcc",f"{WSL_DIR}/cuda-t-2-b.cu","-o",f"{WSL_DIR}/a.out"]
    result=subprocess.run(command, capture_output=True, text=True)
    print(result)
    command2 = ["wsl",fr"{WSL_DIR}/a.out",str(degree),str(num_points),str(alpha),str(space),INPUT_FILE_CONTROL_POINTS_WS,INPUT_FILE_KNOTS_WS,OUTPUT_FILE_WS]
    result=subprocess.run(command2, capture_output=True, text=True)
    print(result)



    # curve_points = np.array([b_spline_curve(control_points, degree, knots, t) for t in np.linspace(0, 3, num_points)])

    curve_points=np.array(read_points(OUTPUT_FILE))
    curve_points = curve_points[~np.all(curve_points == 0, axis=1)]

    # curve_points=curve_points[:333]

    plot_basis(control_points,curve_points)

if __name__ == '__main__':
    main()
    #'(wsl nvcc /home/vitalii/cuda-mch//cuda-t-2-b.cu -o /home/vitalii/cuda-mch//a.out ) -and (wsl /home/vitalii/cuda-mch//a.out 2 1000 0.7853981633974483 ~/cuda-mch/control_points.txt ~/cuda-mch/knots.txt ~/cuda-mch/result.txt )'
    # (wsl nvcc /home/vitalii/cuda-mch//cuda-t-2-b.cu -o /home/vitalii/cuda-mch//a.out ) && (wsl /home/vitalii/cuda-mch//a.out 2 1000 0.7853981633974483 \\wsl.localhost\Ubuntu\home\vitalii\cuda-mch\control_points.txt \\wsl.localhost\Ubuntu\home\vitalii\cuda-mch\knots.txt \\wsl.localhost\Ubuntu\home\vitalii\cuda-mch\result.txt )
#(wsl nvcc /home/vitalii/cuda-mch//cuda-t-2-b.cu ) && (wsl /home/vitalii/cuda-mch//a.out 2 1000 0.7853981633974483 \\wsl.localhost\Ubuntu\home\vitalii\cuda-mch\control_points.txt \\wsl.localhost\Ubuntu\home\vitalii\cuda-mch\knots.txt \\wsl.localhost\Ubuntu\home\vitalii\cuda-mch\result.txt )
# wsl /home/vitalii/cuda-mch//a.out 2 1000 0.7853981633974483 \home\vitalii\cuda-mch\control_points.txt \home\vitalii\cuda-mch\knots.txt \home\vitalii\cuda-mch\result.txt
# (wsl nvcc /home/vitalii/cuda-mch//cuda-t-2-b.cu ) -and (wsl /home/vitalii/cuda-mch//a.out 2 1000 0.7853981633974483 \\wsl.localhost\Ubuntu\home\vitalii\cuda-mch\control_points.txt \\wsl.localhost\Ubuntu\home\vitalii\cuda-mch\knots.txt \\wsl.localhost\Ubuntu\home\vitalii\cuda-mch\result.txt )