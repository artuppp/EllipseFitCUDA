<p align="center">
  <!-- Project Badge -->
  <a href="https://github.com/artuppp/EllipseFitCUDA"><img src="https://img.shields.io/badge/EllipseFitCUDA-GPU%20Accelerated-blueviolet"/></a>
  <!-- Code Repository Badge -->
  <a href="https://github.com/artuppp/EllipseFitCUDA"><img src="https://img.shields.io/badge/code-Source-yellowgreen"/></a>
  <!-- License Badge - Replace with your actual license -->
  <a href="#license"><img src="https://img.shields.io/badge/license-Academic%20%26%20Research%20Only-red"/></a>
  <!-- Last Commit Badge -->
  <a href="https://github.com/artuppp/EllipseFitCUDA/commits/main"><img src="https://img.shields.io/github/last-commit/artuppp/EllipseFitCUDA"/></a>
  <br>
  <!-- Stars Badge -->
  <a href="https://github.com/artuppp/EllipseFitCUDA/stargazers"><img src="https://img.shields.io/github/stars/artuppp/EllipseFitCUDA?style=social"/></a>
  <!-- Forks Badge -->
  <a href="https://github.com/artuppp/EllipseFitCUDA/network/members"><img src="https://img.shields.io/github/forks/artuppp/EllipseFitCUDA?style=social"/></a>
  <!-- Watchers Badge -->
  <a href="https://github.com/artuppp/EllipseFitCUDA/watchers"><img src="https://img.shields.io/github/watchers/artuppp/EllipseFitCUDA?style=social"/></a>
  <!-- Open Issues Badge -->
  <a href="https://github.com/artuppp/EllipseFitCUDA/issues"><img src="https://img.shields.io/github/issues/artuppp/EllipseFitCUDA"/></a>
</p>

<p align="center">
  <!-- TODO: Add a cool GIF or screenshot of EllipseFitCUDA in action! -->
  <!-- Example: <img src="https://raw.githubusercontent.com/artuppp/EllipseFitCUDA/main/assets/ellipse_fit_demo.gif" alt="EllipseFitCUDA Demo" width="700"> -->
  <!-- O puedes usar una imagen estática subida a GitHub Issues: -->
  <!-- <img src='URL_DE_TU_IMAGEN_SUBIDA_A_ISSUES' alt='EllipseFitCUDA Demo' width='700'> -->
<!--   <img src='https://i.imgur.com/YOUR_ELLIPSE_IMAGE_ID.gif' alt='EllipseFitCUDA Demo Placeholder - REPLACE ME!' width='700'>
  <i>Replace with a captivating GIF or image demonstrating ellipse fitting!</i> -->
</p>

<h1 align="center">EllipseFitCUDA</h1>

<p align="center">
  <i>A CUDA GPU-accelerated implementation for fitting ellipses to 2D point data using a least-squares approach.</i>
</p>
<hr>

## 🌟 Introduction

Welcome to **EllipseFitCUDA**! This repository provides an efficient CUDA-based solution for fitting an ellipse to a set of 2D points. The core algorithm employs a least-squares method to determine the best-fit ellipse parameters. This implementation is heavily inspired by and based on the robust MATLAB function `fit_ellipse` by Ohad Gal.

By leveraging the parallel processing power of NVIDIA GPUs through CUDA, CUBLAS, and CUSOLVER, this project aims to deliver high-performance ellipse fitting for applications requiring speed and accuracy.

## ✨ Features

*   🚀 **GPU Acceleration:** Utilizes CUDA, CUBLAS, and CUSOLVER for significant performance gains in ellipse fitting.
*   📐 **Least-Squares Fitting:** Implements a numerically stable least-squares approach to find optimal ellipse parameters.
*   📄 **Based on Proven Algorithm:** The methodology is adapted from Ohad Gal's widely used `fit_ellipse` MATLAB implementation.
*   💡 **Performance Optimization Tip:** Includes guidance for handling small linear systems for potentially even faster execution on specific hardware.
*   🔧 **CUDA Libraries:** Relies on standard NVIDIA CUDA Toolkit libraries for linear algebra operations.

## 📚 Core Algorithm

The ellipse fitting is based on the direct least-squares method for conic sections. Given a set of points (x, y), we seek to find the coefficients (A, B, C, D, E, F) of the general conic equation:
`Ax² + Bxy + Cy² + Dx + Ey + F = 0`
such that it best represents an ellipse passing through or near these points. This problem is typically formulated as a constrained least-squares problem.

This implementation draws its core logic from:
*   Gal, O. (2023). `fit_ellipse` ([www.mathworks.com/matlabcentral/fileexchange/3215-fit_ellipse](https://www.mathworks.com/matlabcentral/fileexchange/3215-fit_ellipse)), MATLAB Central File Exchange. Retrieved Date.

## ⚡ Performance Tip for Small 5x5 Systems

For solving the small 5x5 linear system that arises in this specific least-squares formulation, direct matrix operations might outperform a generic `cuSolver` call on some hardware. This is often due to the overhead associated with `cuSolver`'s internal auto-tuning mechanisms for generalized matrix operations, which might not be optimal for very small, fixed-size problems.

The code block below demonstrates an alternative approach using CUBLAS for matrix multiplications and inversion to solve the system `(D^T * D)^-1 * D^T * S` (where `D` is the design matrix and `S` is the scatter vector, simplified here for concept).

**Consider testing both this direct method and a `cuSolver` (e.g., `cusolverDnSgesv`) approach to determine the faster option for your specific GPU and use case.**

```cuda
// Example: Solving a 5x5 system using direct CUBLAS operations
// Assume:
// - cublas_handle is initialized
// - d_a: device pointer to the design matrix (size x 5)
// - d_b: device pointer to the scatter vector/matrix (size x 1 or 5x5 for the trick below)
// - d_a2inv: device pointer for (A^T*A) and its inverse (5x5)
// - d_x: device pointer for the solution (5x1 or 5x5)
// - size: number of points
// - alpha = 1.0f, beta = 0.0f

// Timer t; // Assuming a Timer class for benchmarking

// Solving through PSEUDOINVERSE (D^T*D)^-1 * D^T * S
// Let A_prime = D^T*D in the comments below for brevity
cudaDeviceSynchronize();
// t.start();

// Calculate D^T*D (Result is 5x5, stored in d_a2inv)
// cublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
// C = alpha*op(A)*op(B) + beta*C
// D(size x 5), D^T(5 x size) => D^T*D (5x5)
cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, 
            5, 5, size, 
            &alpha, d_design_matrix, size,  // op(D) = D^T, lda = size (leading dim of D)
            d_design_matrix, size,          // op(D) = D,   ldb = size
            &beta, d_DtD_matrix, 5);        // d_DtD_matrix (5x5), ldc = 5

// Inverse (D^T*D)^-1 (Result stored in d_DtD_inv_matrix, can be same as d_DtD_matrix if done in-place carefully or use separate)
// This example uses cublasSmatinvBatched for a single matrix; a direct LU decomposition + inversion might be more typical for a single matrix.
// For simplicity, let's assume d_DtD_inv_matrix now holds the inverse.
// Example using cublasSmatinvBatched (requires d_array_of_DtD_pointers and d_array_of_DtD_inv_pointers):
// float *A_array[1] = {d_DtD_matrix};
// float *C_array[1] = {d_DtD_inv_matrix};
// cudaMemcpy(d_A_array_dev, A_array, sizeof(float*), cudaMemcpyHostToDevice);
// cudaMemcpy(d_C_array_dev, C_array, sizeof(float*), cudaMemcpyHostToDevice);
// cublasSmatinvBatched(cublas_handle, 5, d_A_array_dev, 5, d_C_array_dev, 5, d_info, 1);

// Here's the user's provided snippet for the A^T*A based system:
// Note: The variable names (d_a, d_a2inv, d_b, d_x) in the user's snippet
// might refer to different stages or simplified representations
// of the full D, D^T*D, (D^T*D)^-1, D^T*S etc.
// The following is the user's code, adapt variable names for your full implementation.

// Assuming d_a is the "Design Matrix D" or a part of it, and d_b is related to "S"
// And the system is simplified to A*x = b where A might be D^T*D

// Calculate A_prime = A^T*A (if 'd_a' is the original design matrix D)
// Here, user's 'd_a' seems to be already a 5xN matrix.
// If 'd_a' is the (5 x num_points) design matrix D, then D^T (num_points x 5)
// D (5 x num_points) * D^T (num_points x 5) -> (5x5)
// cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, 5, 5, size, &alpha, d_a, 5, d_a, 5, &beta, d_a2inv, 5);

// Let's use the variable names from the user's example:
// d_a (5xN matrix for A), d_a2inv (5x5 for A^T*A and its inverse), d_b (5xN or 5x1 for b), d_x (5xN or 5x1 for solution)
// 'size' here seems to be 'N' (number of columns in d_a, potentially number of points if d_a is D^T)
// Constructor

// blobFinder = BlobFinder(5, 20, 4); // This line seems to be from the previous BlobFinder example, remove if not relevant to EllipseFitCUDA

// cv::Mat imageRawWB2; // OpenCV related, remove if not relevant
// cv::resize(imageRawWB, imageRawWB2, cv::Size(), 0.25, 0.25, cv::INTER_AREA); // OpenCV related
// auto reflexes = blobFinder.blob_log(imageRawWB2, 5, 20, 4, 0.1, false); // BlobFinder related
// Logger::Log("WavefrontAnalyzer::processHS", "end blob finder"); // Logging related
// std::transform(reflexes.begin(), reflexes.end(), reflexes.begin(), [](std::tuple<int, int, float>& x) { // BlobFinder related
//     std::get<0>(x) = std::get<0>(x) * 4;
//     std::get<1>(x) = std::get<1>(x) * 4;
//     std::get<2>(x) = std::get<2>(x) * 4;
//     return x;
// }); // BlobFinder related

// --- Start of User's Provided Snippet (Interpreted) ---
// Assuming d_a is a (5 x M) matrix, and d_b is a (5 x M) matrix representing M right-hand sides (or M=1 for a single vector)
// And the goal is to solve for x in (A^T*A)x = A^T*b where A is `d_a` (5xM design matrix for M points)
// and b is `d_b` (Mx1 vector of ones, usually for conic fitting Ax^2+...+F=0 form)
// Let's reinterpret 'd_a' as the Design Matrix D (M rows, 5 cols) and 'd_b' as the vector of -1s (M rows, 1 col)

// This section needs clarification on what d_a, d_b, and size represent in the context of ellipse fitting.
// The provided snippet seems to be a generic Ax=b solver using pseudoinverse for a 5-parameter system.

// If d_a is the Design Matrix (num_points x 5)
// d_b is the target vector (num_points x 1)
// d_DtD (5x5) for D^T * D
// d_DtD_inv (5x5) for (D^T * D)^-1
// d_Dt_b (5x1) for D^T * b
// d_params (5x1) for the solution (D^T*D)^-1 * D^T * b

// Calculate D^T*D
// cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, 5, 5, num_points, &alpha, d_design_matrix, num_points, d_design_matrix, num_points, &beta, d_DtD, 5);
// Invert D^T*D (using batched inverse for one matrix)
// cublasSmatinvBatched(cublas_handle, 5, &d_DtD_ptr, 5, &d_DtD_inv_ptr, 5, d_info, 1);
// Calculate D^T * b
// cublasSgemv(cublas_handle, CUBLAS_OP_T, num_points, 5, &alpha, d_design_matrix, num_points, d_b_vector, 1, &beta, d_Dt_b_vector, 1);
// Multiply (D^T*D)^-1 * (D^T*b)
// cublasSgemv(cublas_handle, CUBLAS_OP_N, 5, 5, &alpha, d_DtD_inv, 5, d_Dt_b_vector, 1, &beta, d_ellipse_params, 1);


// The user's original snippet:
// Solving through PSEUDOINVERSE (A^T*A)-1 * A^T * b (with A_prime = A^T*A)
// Here, 'd_a' likely represents the Design Matrix D.
// 'size' is likely num_points.
// 'd_a2inv' is for D^T*D and its inverse.
// 'd_b' is the vector of observations (e.g., all -1s or related to constraints).
// 'd_x' is the resulting ellipse parameters.

cudaDeviceSynchronize();
// t.start(); // Your timer start

// Calculate A^T*A (here A is d_a, result in d_a2inv)
// d_a: design matrix (num_points x 5)
// d_a2inv: (D^T*D) (5x5)
cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, 
            5, 5, num_points, /* k=num_points */
            &alpha, d_a, num_points, /* A=d_a, lda=num_points */
            d_a, num_points, /* B=d_a, ldb=num_points */
            &beta, d_a2inv, 5); /* C=d_a2inv, ldc=5 */

// Inverse (A^T*A)^-1 (in-place in d_a2inv)
// For SmatinvBatched, you need an array of pointers
float * p_d_a2inv = d_a2inv;
float * p_d_a2inv_inv = d_a2inv; // For in-place (or use a different buffer)
cudaMemcpyAsync(d_arrayOfMatrices, &p_d_a2inv, sizeof(float*), cudaMemcpyHostToDevice, stream);
cudaMemcpyAsync(d_arrayOfInverseMatrices, &p_d_a2inv_inv, sizeof(float*), cudaMemcpyHostToDevice, stream);
cublasSmatinvBatched(cublas_handle, 5, d_arrayOfMatrices, 5, d_arrayOfInverseMatrices, 5, d_info_gpu, 1);
// (Ensure d_arrayOfMatrices, d_arrayOfInverseMatrices, d_info_gpu are allocated on device)

// Temp matrix for (A^T*A)^-1 * A^T (let's call it P_temp, 5xN)
// P_temp = (d_a2inv_inv) * (d_a^T)
// d_a2inv_inv (5x5), d_a^T (5 x num_points) -> P_temp (5 x num_points)
cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, 
            5, num_points, 5, /* m=5, n=num_points, k=5 */
            &alpha, d_a2inv, 5,     /* A=d_a2inv_inv (now in d_a2inv), lda=5 */
            d_a, num_points,        /* B=d_a (use op_T), ldb=num_points */
            &beta, d_P_temp, 5);    /* C=d_P_temp, ldc=5 */ (Allocate d_P_temp: 5 * num_points)

// Multiply P_temp * b to get parameters x
// x = P_temp * d_b
// d_P_temp (5 x num_points), d_b (num_points x 1) -> d_x (5x1)
cublasSgemv(cublas_handle, CUBLAS_OP_N,
            5, num_points,      /* m=5, n=num_points */
            &alpha, d_P_temp, 5, /* A=d_P_temp, lda=5 */
            d_b, 1,              /* x=d_b (vector), incx=1 */
            &beta, d_x, 1);      /* y=d_x (vector), incy=1 */


// The "trick" mentioned by user for 5x5 matrix-vector by doing 5x5 * 5x5 and taking first column:
// This implies d_b was shaped as a 5x5 identity or similar to make the mat-mat product work.
// If d_b is indeed a 5x1 vector, Sgemv is more direct.
// If the intention was ( (A^T*A)^-1 * A^T ) * B_prime where B_prime is 5x5:
// cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, 5, 5, 5, &alpha, d_P_temp_if_5x5, 5, d_b_if_5x5, 5, &beta, d_x_if_5x5, 5);
// Then extract the first column from d_x_if_5x5. This is less direct than Sgemv for a single vector 'b'.

cudaDeviceSynchronize();
// time = t.elapsed();
// cout << "Time solve system: " << time / 10e6 << " ms" << endl;
// printBMatrix<<<1, 1>>>(d_x); // Custom kernel to print result
