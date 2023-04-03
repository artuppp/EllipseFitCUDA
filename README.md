# EllipseFitCUDA
Ellipse Fit Implementation in CUDA

This repository contains an CUDA GPU-accelerated implementation of a fitting ellipse based on least-squares approach. This is completely based on 
Ohad Gal (2023). fit_ellipse (https://www.mathworks.com/matlabcentral/fileexchange/3215-fit_ellipse), MATLAB Central File Exchange.

It uses CUBLAS and CUSOLVE libraries which are included into the NVIDIA CUDA TOOLKIT

**Note**: for small 5x5 system solve I recomend using this code block instead of cuSolve call. This is because for last matrix-vector multiplication (which is also performed on the cuSolve call) there is some auto-tunning overhead that is slowing our execution (avoided down using simple 5x5 matrix multiplcation and picking just first column of the result). Since it also depends on your hardware capabilities, you should try both versions and use the faster one for your device.

```
        // Solving through PSEUDOINVERSE (A^T*A)-1 * A^T * b (with A = A^T*A)
        cudaDeviceSynchronize();
        t.start();
        // Calculate A^T*A
        cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, 5, 5, size, &alpha, d_a, 5, d_a, 5, &beta, d_a2inv,
                    5);
        // Inverse (A^T*A)-1
        cublasSmatinvBatched(cublas_handle, 5, d_array, 5, d_array, 5, d_info, 1);

        // Multiply (A^T*A)-1 * A^T
        cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, 5, 5, 5, &alpha, d_a2inv, 5, d_a, 5, &beta, d_a,
                    5);
        // Multiply (A^T*A)-1 * A^T
        // I dont know why it is quite fast to do the 5x5 * 5x5 matrix multiplication instead of 5x5 * 5x1 matrix multiplication, so I do it this way and take the first column of the result
        cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, 5, 5, 5, &alpha, d_a, 5, d_b, 5, &beta, d_x,
                    5);

        cudaDeviceSynchronize();
        time = t.elapsed();
        cout << "Time solve system: " << time / 10e6 << " ms" << endl;
        //        printBMatrix<<<1, 1>>>(d_x);

```
