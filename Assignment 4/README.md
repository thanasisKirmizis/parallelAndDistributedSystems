# Assignment 4: Implement the construction of a VP Tree and perform All-kNN search on the tree using CUDA

## Description

This assignment targets the optimization of the construction of a VP Tree data structure with the use of CUDA, as well as the performance of an All-kNN search on the points of the tree.

You can try adjusting the size of input to check the difference in execution time between the implementations. You can read more about Vantage Point Trees and its usages here: https://fribbels.github.io/vptree/writeup

## Usage

This project makes use of CUDA programming. Make sure you have a compatible NVIDIA GPU in your computer and the NVIDIA CUDA toolkit installed.

You can then compile the CUDA files with `nvcc main.cu` and run them as usual at a Linux terminal.

By running the file `main.cu`, the program is going to ask for user input. Press 1 to display simple results or 2 to display detailed results.
At the end of each part of the program, a mini-tester has been implemented to check whether the program finished correctly or any error (usually memory leaks) has been occured.

