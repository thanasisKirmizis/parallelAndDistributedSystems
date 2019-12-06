# Assignment 2: Find the kNN of Distributed Data Points

## Description

This assignment targets the implementation of the kNN algorithm for data that are distributed in disjoint
blocks. The implementations done for this assignment are the following:
- Impl. #1 is sequential algorithm.
- Impl. #2 is distributed algorithm using MPI with synchronous communication.
- Impl. #3 is distributed algorithm using MPI with asynchronous communication.

You can try adjusting the size of input to check the difference in execution time between these implementations. 
You can read more about kNN and its usages here: 
https://www.analyticsvidhya.com/blog/2018/03/introduction-k-neighbours-algorithm-clustering/

## Program analysis from execution on the HPC cluster 

Even though running the MPI implementation of a code on a single node (such as your local machine) might not provide any 
substantial improvements, running it on a truly distributed system shows the true potential of the MPI "parallelization".
In this case, the tests were done on this HPC cluster (https://it.auth.gr/en/hpc), in order to demonstrare the difference in execution time between the asynchronous versus the synchronous implementation. Below are the figures of the results:



## Usage

This project makes use of the OpenBLAS library. Make sure you have it installed and included the path
of its installation files in your Makefile. To do that, edit the variable INCLUDES insided the provided Makefile to
correspond to the path of your OpenBLAS installation.

By then running `make lib` at a Linux terminal, the makefile creates two static libraries in the `/lib` folder, one for
the sequential and one for the MPI implementation. 
You can then use those libraries to run them through a program of yours.
