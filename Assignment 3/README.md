# Assignment 3: Implement an evolution of the 2D Ising model using CUDA

## Description

This assignment targets the implementation of an evolution of the 2D Ising model. The focus of this assignment is centered on finding the states of the atomic "spins" on an Ising square lattice, on a given time "step". The implementations done for this assignment are the following:
- Impl. #1 is sequential algorithm.
- Impl. #2 is parallel algorithm using CUDA with as many blocks as lattice points and with each block containing a single thread.
- Impl. #3 is parallel algorithm using CUDA with blocks less than the lattice points and with each block containing a single thread that takes up the calculation of the states of a block of points.
- Impl. #4 is parallel algorithm using CUDA with blocks less than the lattice points and with each block containing multiple threads that read from a shared memory.

You can try running the files to check the difference in execution time between these implementations. 
You can read more about the Ising model and its usages here: 
https://en.wikipedia.org/wiki/Ising_model

## Usage

This project makes use of CUDA programming. Make sure you have a compatible NVIDIA GPU in your computer and the NVIDIA CUDA toolkit installed.

You can then compile the CUDA files with `nvcc` and run them as usual at a Linux terminal.

