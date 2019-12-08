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
In this case, the tests were done on this HPC cluster (https://it.auth.gr/en/hpc), in order to demonstrare the difference in execution time between the asynchronous versus the synchronous implementation. The goal here is to "hide" the communications time under calculations time, so running the program even on many different nodes may create small difference in terms of time. 
Below are the results of the tests run to find the k=20 nearest neighbors on various input data sizes:

|Graph 1      |
| :---------: |
| ![graph 1](https://github.com/thanasisKirmizis/parallelAndDistributedSystems/blob/master/Assignment%202/graphs/b18.png)| 

|Graph 2   |
| :---------: |
|![graph 2](https://github.com/thanasisKirmizis/parallelAndDistributedSystems/blob/master/Assignment%202/graphs/b24.png)| 

| Graph 3 |
|:---------: |
|![graph 3](https://github.com/thanasisKirmizis/parallelAndDistributedSystems/blob/master/Assignment%202/graphs/b42.png)|

Despite the fact that a greater improvement in time was probably expected from the asynchronous implementation, the difference is existing and obvious nonetheless. It's possible that this experiment was not really ideal to demonstrate the true capabilities of the asynchronous communications, likely due to the bottleneck here being the calculations, and not so much the communications.

However, as we can see, the **more different nodes** of the distributed system are used to run the code, the more does the **communication time** between the various processes **increases**. Note that for very small input sizes, the calculations time becomes so small that the communications time cannot be hidden beneath it!

Also note that by changing the k to k=200, there is no difference in communications time, so no improvement can be seen when compared to the corresponding graph for k=20:

|Graph 4   |
| :---------: |
|![graph 4](https://github.com/thanasisKirmizis/parallelAndDistributedSystems/blob/master/Assignment%202/graphs/b42_k200.png)|


## Usage

This project makes use of the OpenBLAS library. Make sure you have it installed and included the path
of its installation files in your Makefile. To do that, edit the variable INCLUDES insided the provided Makefile to
correspond to the path of your OpenBLAS installation.

By then running `make lib` at a Linux terminal, the makefile creates two static libraries in the `/lib` folder, one for
the sequential and one for the MPI implementation. 
You can then use those libraries to run them through a program of yours.
