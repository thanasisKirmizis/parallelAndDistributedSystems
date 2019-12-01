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

## Usage

This project makes use of the OpenBLAS library. Make sure you have it installed and included the path
of its installation files in your Makefile. To do that, edit the variable INCLUDES insided the provided Makefile to
correspond to the path of your OpenBLAS installation.
By then running `make lib` at a Linux terminal, the makefile creates the four static libraries in the `/lib` folder, one for
each implementation. 
You can then use those libraries to run them through a program of yours.