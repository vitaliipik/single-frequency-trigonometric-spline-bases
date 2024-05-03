#include <iostream>
#include <vector>
#include <cmath>
#include <stdlib.h>
#include <fstream>

__device__ __host__ double basisFunction(int i, int k, double t, double* knots, double alpha) {
    if (k == 0) {
        return (knots[i] <= t && t < knots[i + 1]) ? 1.0 : 0.0;
    }

     auto denom2 = [=]  (int i) {
        return 2 * sin(t - knots[i + 1]) * (cos(alpha) - 1);
    };

    auto coeff1 = [=]  (int i) {
        return (alpha == 0.0) ? 0.0 : (sin((t - knots[i]) / 2) * cos((alpha - t + knots[i]) / 2)) / sin(alpha / 2);
    };

    auto top = [=]  (int i) {
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

__global__ void bSplineCurveKernel(double* controlPoints, int numControlPoints, int degree, double* knots, int numPoints, double* curvePoints, double alpha, int numAxes, double space) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numPoints) {
        double t = idx * space / (numPoints);
        double result[2] = {0,0};
        for (int i = 0; i < numControlPoints+2-degree; ++i) {
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
        // std::cout<<(val)<<std::endl;
        controlPoints.push_back(val);
    }

    file.close();
}

int main(int argc, char* argv[]) {

if (argc < 7) {
        std::cerr << "Usage: " << argv[0] << " numControlPoints degree numKnots numPoints apha inputFile outputFile" << std::endl;
        return 1;
    }
    
    int degree = std::stoi(argv[1]);
    int numPoints = std::stoi(argv[2]);
    double alpha = std::atof(argv[3]);
    double space= std::atof(argv[4]);
    
    std::string inputFilePoints = argv[5];
      std::string inputFileKnots = argv[6];
    std::string outputFile = argv[7];
    // double alpha= (double)alpha_i*(180.0/M_PI);
    // std::cout<<alpha;
    int numControlPoints;
    std::vector<double> controlPoints;
    int numAxes;
    readControlPoints(inputFilePoints, controlPoints, numControlPoints);

    numAxes=controlPoints.size()/numControlPoints;
    // for (int i = 0; i < numControlPoints; ++i) {
    //     controlPointsFile >> numAxes;
    //     std::cout<<numAxes;
    //     controlPointsFile >> controlPoints[i * 2] >> controlPoints[i * 2 + 1] >> controlPoints[i * 2 + 2];
    // }
    // controlPointsFile.close();

    // Read knots from file
    std::ifstream knotsFile(inputFileKnots);
    int numKnots;
    knotsFile >> numKnots;
    double* knots = new double[numKnots];
    for (int i = 0; i < numKnots; ++i) {
        knotsFile >> knots[i];
    }
    knotsFile.close();

       // Output curve points
    

// for (int i = 0; i < numControlPoints; ++i) {
//         std::cout << controlPoints[i * 2] <<controlPoints[i * 2 + 1]<<"\n";
//     }
// for (int i = 0; i < numKnots; ++i) {
//         std::cout << knots[i]<<"\n";
//     }
 cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start); // Start the timer

    int numBlocks = (numPoints + 255) / 256;
    double* devControlPoints, * devKnots, * devCurvePoints;
    cudaMalloc((void**)&devControlPoints, sizeof(double) * numControlPoints * numAxes);
    cudaMalloc((void**)&devKnots, sizeof(double) * numKnots);
    cudaMalloc((void**)&devCurvePoints, sizeof(double) * numPoints * numAxes);
    cudaMemcpy(devControlPoints, controlPoints.data(), sizeof(double) * numControlPoints * numAxes, cudaMemcpyHostToDevice);
    cudaMemcpy(devKnots, knots, sizeof(double) * numKnots, cudaMemcpyHostToDevice);
    bSplineCurveKernel<<<numBlocks, 256>>>(devControlPoints, numControlPoints, degree, devKnots, numPoints, devCurvePoints,alpha, numAxes,space);
    double* curvePoints = new double[numPoints * numAxes];
    cudaMemcpy(curvePoints, devCurvePoints, sizeof(double) * numPoints * numAxes, cudaMemcpyDeviceToHost);
    cudaFree(devControlPoints);
    cudaFree(devKnots);
    cudaFree(devCurvePoints);

    // // Output curve points
    // for (int i = 0; i < numPoints; ++i) {
    //     std::cout << curvePoints[i * 2] << ", " << curvePoints[i * 2 + 1] << std::endl;
    // }
   // Write result to file
   cudaEventRecord(stop); // Stop the timer
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop); // Get the execution time in milliseconds

    std::ofstream outFile(outputFile);
    for (int i = 0; i < numPoints; ++i) {
        if(numAxes==3){
        outFile << curvePoints[i * 2] << " " << curvePoints[i * 2 + 1] <<  " " << curvePoints[i * 2 + 2] << std::endl;
        }else{
             outFile << curvePoints[i * 2] << " " << curvePoints[i * 2 + 1] << std::endl;
        }
    }
    
    outFile.close();
    std::cout << milliseconds/1000 << std::endl;

    
    // std::cout << "Time elapsed: " << milliseconds/1000 << std::endl;
    // std::cout<<"success";
    delete[] curvePoints;
    return 1;
}
