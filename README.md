# Parallel Programming with CPU and GPU

This repository contains information and examples on parallel programming with CPU and GPU, including OpenMP, pragma, MPI, CUDA, and HIP.

## System Architecture

### CPU

A Central Processing Unit (CPU) is the primary component of a computer that performs most of the processing inside a computer. It is designed to handle a wide variety of tasks quickly and efficiently. CPUs are optimized for single-threaded performance and are capable of executing complex instructions.

### GPU

A Graphics Processing Unit (GPU) is a specialized processor designed to accelerate graphics rendering. GPUs are highly parallel in nature and are capable of handling thousands of threads simultaneously. This makes them well-suited for parallel processing tasks, such as scientific simulations, machine learning, and image processing.

### QPU

A Quantum Processing Unit (QPU) is a type of processor that uses the principles of quantum mechanics to perform computations. QPUs are designed to solve complex problems that are intractable for classical computers, such as factoring large numbers, simulating quantum systems, and optimizing large-scale problems. Quantum computing is still in its early stages, but it has the potential to revolutionize fields such as cryptography, materials science, and artificial intelligence.

> Detailed videos of System Design concepts for Parallel Programming available at: [System Design for Parallel Programming](https://youtube.com/playlist?list=PLWyBQeJgIuzD2o9ZVw5oI-P2fX4uyNKZA&si=947INnGI2lcnDbJE)

## Parallel Programming Models

### OpenMP

OpenMP is an API that supports multi-platform shared memory multiprocessing programming in C, C++, and Fortran. It consists of a set of compiler directives, library routines, and environment variables that influence run-time behavior.

For more information, visit the [OpenMP official website](https://www.openmp.org/).

### Pragma

Pragma is a directive in C and C++ that provides additional information to the compiler. It is often used to specify parallelism and optimization hints.

For more information, visit the [Pragma documentation](https://en.cppreference.com/w/cpp/preprocessor/impl).

### MPI

Message Passing Interface (MPI) is a standardized and portable message-passing system designed to function on parallel computing architectures. MPI is widely used for parallel programming in distributed memory systems.

For more information, visit the [MPI official website](https://www.mpi-forum.org/).

### CUDA

Compute Unified Device Architecture (CUDA) is a parallel computing platform and application programming interface (API) model created by NVIDIA. It allows developers to use NVIDIA GPUs for general-purpose processing (GPGPU).

For more information, visit the [CUDA official website](https://developer.nvidia.com/cuda-zone).

### HIP

Heterogeneous-computing Interface for Portability (HIP) is a C++ runtime API that allows developers to create portable applications that can run on AMD and NVIDIA GPUs. HIP provides a common interface for GPU programming, making it easier to write code that can run on different hardware platforms.

For more information, visit the [HIP official website](https://github.com/ROCm-Developer-Tools/HIP).
