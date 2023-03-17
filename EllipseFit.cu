/*
    * EllipseFit.cu
    *
    *  Created on: 2018-11-27
    *  Author:     arturo.vicentej@um.es
*/
#include <opencv2/highgui.hpp>
#include <opencv2/core/types_c.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include "Timer.cpp"
#include <cuda.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include "cublas_v2.h"
#include <cusolverDn.h>

using namespace std;
using namespace cv;

//--------------------------------------------------------------------------------------------
__global__
void sumColumns(float *matrix, float *result, int width, int height) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int i = index / width;
    int j = index % width;
    if (i < height && j < width) {
        atomicAdd(&result[j], matrix[i * width + j]);
    }
}

static cv::RotatedRect GPUfitEllipse(std::vector <Point2f> points)
{
    cv::RotatedRect box;
    box.center.x = -1;
    box.center.y = -1;
    box.size.width = -1;
    box.size.height = -1;
    box.angle = -1;
    if (points.size() >= 5)
    {
        float* A, *h_x;
        cudaMallocHost((void**)&A, 5*points.size()*sizeof(float));
        cudaMallocHost((void**)&h_x, 5*sizeof(float));
        float mean_x = 0;
        float mean_y = 0;
        //Calculate mean of points
        for (int i = 0; i < points.size(); i++)
        {
            mean_x += points[i].x;
            mean_y += points[i].y;
        }
        mean_x /= points.size();
        mean_y /= points.size();
        //Substract mean to points and create points matrix
        for (int i = 0; i < points.size(); i++)
        {
            points[i].x -= mean_x;
            points[i].y -= mean_y;
            A[5*i] = (float)points[i].x*(float)points[i].x;
            A[5*i+1] = (float)points[i].x*(float)points[i].y;
            A[5*i+2] = (float)points[i].y*(float)points[i].y;
            A[5*i+3] = (float)points[i].x;
            A[5*i+4] = (float)points[i].y;
        }
        //Initialize variables
        double threshold = 1e-3;
        float *d_A, *d_b, *d_x, alpha = 1.0f, beta = 0.0f, *gdwork;
        float AA, B, C, D, E, phi, cos_phi, sin_phi, cos_phi2, sin_phi2, sin_phi_cos_phi, A1, C1, D1, E1, F2, mean_xx;
        cublasHandle_t cublas_handle;
        cublasCreate(&cublas_handle);
        cusolverDnHandle_t solver_handle;
        cusolverDnCreate(&solver_handle);
        size_t work_bytes;
        int *d_info, hinfo, nil = 0;
        cudaMalloc((void**)&d_info, sizeof(int));
        cudaMalloc((void**)&d_A, 5*points.size()*sizeof(float));
        cudaMalloc((void**)&d_b, 5*sizeof(float));
        cudaMalloc((void**)&d_x, 5*sizeof(float));
        //Copy points matrix to device
        cudaMemcpy(d_A, A, 5*points.size()*sizeof(float), cudaMemcpyHostToDevice);
        //Not count in time as it could be the same for all iterations
        cusolverDnSSgels_bufferSize(solver_handle, 5, 5, 1, d_A, 5, d_b, 5, d_x, 5, NULL, &work_bytes);
        cudaMalloc(&gdwork, work_bytes * sizeof(float));
        //Start timer
        Timer t;
        t.start();
        //Calculate sum of columns
        sumColumns<<<points.size()*5/64+1, 64>>>(d_A, d_b, 5, points.size());
        //Calculate covariance matrix (A^t*A)
        cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, 5, 5, points.size(), &alpha, d_A, 5, d_A, 5, &beta, d_A, 5);
        //Solve system Cov * x = Sum(X)
        cusolverStatus_t stat = cusolverDnSSgels(solver_handle, 5, 5, 1, d_A, 5, d_b, 5, d_x, 5, gdwork, work_bytes, &nil, d_info);
        //Pass result to host
        cudaMemcpy(h_x, d_x, 5*sizeof(float), cudaMemcpyDeviceToHost);
        AA = h_x[0];
        B = h_x[1];
        C = h_x[2];
        D = h_x[3];
        E = h_x[4];
        phi = 0.5*atan2(B, C-AA);
        cos_phi = cos(phi);
        sin_phi = sin(phi);
        cos_phi2 = cos_phi*cos_phi;
        sin_phi2 = sin_phi*sin_phi;
        sin_phi_cos_phi = sin_phi*cos_phi;
        A1 = AA*cos_phi2 - B*sin_phi_cos_phi + C*sin_phi2;
        C1 = AA*sin_phi2 + B*sin_phi_cos_phi + C*cos_phi2;
        D1 = D*cos_phi - E*sin_phi;
        E1 = D*sin_phi + E*cos_phi;
        if (A1 < 0) {
            A1 = -A1;
            C1 = -C1;
            D1 = -D1;
            E1 = -E1;
        }
        F2 = 1 + (D1 * D1)/(4*A1) + (E1 * E1)/(4*C1);
        mean_xx = cos_phi*mean_x - sin_phi*mean_y;
        mean_y = sin_phi*mean_x + cos_phi*mean_y;
        mean_x = mean_xx;

        if (A1*C1 <= 0)
        {
            cout << "No ellipse found" << endl;
            return box;
        }

        auto time = t.elapsed();
        cout << "GPU Ellipse: " << time/10e6 << endl;

        box.center.x = (mean_x - D1/2/A1)*cos_phi + (mean_y - E1/2/C1)*sin_phi;
        box.center.y = (mean_y - E1/2/C1)*cos_phi - (mean_x - D1/2/A1)*sin_phi;
        box.size.width = 2*sqrt(abs(F2/A1));
        box.size.height = 2*sqrt(abs(F2/C1));
        box.angle = phi*180/3.14159265359;

        cudaFree(d_A);
        cudaFree(d_b);
        cudaFree(d_x);
        cudaFree(d_info);
        cudaFree(gdwork);
        cusolverDnDestroy(solver_handle);
        cublasDestroy(cublas_handle);
        cudaFreeHost(h_x);
        cudaFreeHost(A);
    }
    return box;
}

int main(int argc, char *argv[]) {
    //create vector with lots of points
    vector <Point2f> points;
    points.push_back(Point2f(15, 101));
    points.push_back(Point2f(40, 157));
    points.push_back(Point2f(115, 140));
    points.push_back(Point2f(37, 55));
    points.push_back(Point2f(80, 35));
    for (int i = 0; i < 500; i++) {
        points.push_back(Point2f(15, 101));
        points.push_back(Point2f(40, 157));
        points.push_back(Point2f(115, 140));
        points.push_back(Point2f(37, 55));
        points.push_back(Point2f(80, 35));
    }
    // loop to test speed
    for (int i = 0; i < 30; i++) {
        //GPU fit ellipse
        auto b = GPUfitEllipse(points);
        cout << b.center.y << endl;
        cout << b.center.x << endl;
        cout << b.size.height << endl;
        cout << b.size.width << endl;
        cout << b.angle << endl;
        Mat img = Mat::zeros(200, 200, CV_8UC1);
        cvtColor(img, img, COLOR_GRAY2BGR);
        for (auto p: points) {
            circle(img, p, 1, Scalar(0, 0, 255), -1);
        }
        ellipse(img, b, Scalar(0, 255, 0), 2);
        imshow("img", img);
        cout << "end" << endl;
        //CV fit ellipse
        Timer t;
        t.start();
        auto bb = cv::fitEllipse(points);
        auto time = t.elapsed();
        cout << "CV Ellipse: " << time / 10e6 << endl;
        cout << bb.center.y << endl;
        cout << bb.center.x << endl;
        cout << bb.size.height << endl;
        cout << bb.size.width << endl;
        cout << bb.angle << endl;
        Mat img2 = Mat::zeros(200, 200, CV_8UC1);
        cvtColor(img2, img2, COLOR_GRAY2BGR);
        for (auto p: points) {
            circle(img2, p, 1, Scalar(0, 0, 255), -1);
        }
        ellipse(img2, b, Scalar(0, 255, 0), 2);
        imshow("img2", img2);
        cout << "end" << endl;
    }
    waitKey(0);
    exit(0);
}
