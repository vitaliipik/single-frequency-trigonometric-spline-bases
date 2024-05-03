#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <chrono>

double basisFunction(int i, int k, double t, double* knots, double alpha) {
    if (k == 0) {
        return (knots[i] <= t && t < knots[i + 1]) ? 1.0 : 0.0;
    }

    auto denom2 = [=] (int i) {
        return 2 * sin(t - knots[i + 1]) * (cos(alpha) - 1);
    };

    auto coeff1 = [=] (int i) {
        return (alpha == 0.0) ? 0.0 : (sin((t - knots[i]) / 2) * cos((alpha - t + knots[i]) / 2)) / sin(alpha / 2);
    };

    auto top = [=] (int i) {
        return sin(t - knots[i + 1] + alpha) - sin(t - knots[i + 1]) - sin(alpha);
    };

    auto coeff2 = [=] (int i) {
        return (denom2(i) == 0.0) ? 0.0 : top(i) / denom2(i);
    };

    if (k == 2) {
        double p = 1 - coeff2(i + 1);
        return coeff2(i) * basisFunction(i, k - 1, t, knots, alpha) + p * basisFunction(i + 1, k - 1, t, knots, alpha);
    }
    double p = 1 - coeff1(i + 1);
    return coeff1(i) * basisFunction(i, k - 1, t, knots, alpha) + p * basisFunction(i + 1, k - 1, t, knots, alpha);
}

void bSplineCurve(double* controlPoints, int numControlPoints, int degree, double* knots, int numPoints, double* curvePoints, double alpha, int numAxes, double space) {
    for (int idx = 0; idx < numPoints; ++idx) {
        double t = idx * space / numPoints;
        double result[2] = {0, 0};
        for (int i = 0; i < numControlPoints + 2 - degree; ++i) {
            double basis = basisFunction(i, degree, t, knots, alpha);
            for (int j = 0; j < numAxes; ++j) {
                result[j] += controlPoints[i * numAxes + j] * basis;
            }
        }
        for (int j = 0; j < numAxes; ++j) {
            curvePoints[idx * numAxes + j] = result[j];
        }
    }
}

void readControlPoints(const std::string& fileName, std::vector<double>& controlPoints, int& numPoints) {
    std::ifstream file(fileName);
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file " << fileName << std::endl;
        return;
    }

    file >> numPoints;
    double val;
    while (file >> val) {
        controlPoints.push_back(val);
    }

    file.close();
}

int main(int argc, char* argv[]) {
    if (argc < 8) {
        std::cerr << "Usage: " << argv[0] << "degree numPoints alpha space inputFilePoints inputFileKnots outputFile" << std::endl;
        return 1;
    }

    int degree = std::stoi(argv[1]);
    int numPoints = std::stoi(argv[2]);
    double alpha = std::atof(argv[3]);
    double space = std::atof(argv[4]);
    std::string inputFilePoints = argv[5];
    std::string inputFileKnots = argv[6];
    std::string outputFile = argv[7];

    int numControlPoints;
    std::vector<double> controlPoints;
    int numAxes;
    readControlPoints(inputFilePoints, controlPoints, numControlPoints);
    numAxes = controlPoints.size() / numControlPoints;

    std::ifstream knotsFile(inputFileKnots);
    int numKnots;
    knotsFile >> numKnots;
    double* knots = new double[numKnots];
    for (int i = 0; i < numKnots; ++i) {
        knotsFile >> knots[i];
    }
    knotsFile.close();

    auto start = std::chrono::high_resolution_clock::now();

    double* curvePoints = new double[numPoints * numAxes];
    bSplineCurve(controlPoints.data(), numControlPoints, degree, knots, numPoints, curvePoints, alpha, numAxes, space);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::ofstream outFile(outputFile);
    for (int i = 0; i < numPoints; ++i) {
        if (numAxes == 3) {
            outFile << curvePoints[i * 3] << " " << curvePoints[i * 3 + 1] << " " << curvePoints[i * 3 + 2] << std::endl;
        } else {
            outFile << curvePoints[i * 2] << " " << curvePoints[i * 2 + 1] << std::endl;
        }
    }
    outFile.close();

    delete[] curvePoints;
    delete[] knots;

    std::cout <<  elapsed.count() << std::endl;


    return 0;
}
