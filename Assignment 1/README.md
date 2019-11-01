# Assignment 1: Build a Vantage Point Tree

## Description

This assignment targets the creation of a vantage point tree through a variety of implementations. 
- Impl. #1 is sequential algorithm.
- Impl. #2 is parallel algorithm using C's pthreads.
- Impl. #3 is parallel algorithm using the Cilk API.
- Impl. #4 is parallel algorithm using the OpenMP API.

You can try adjusting the size of input to check the difference in execution time between these implementations. 
You can read more about Vantage Point Trees and its usages here: https://fribbels.github.io/vptree/writeup

## Usage

By running `make lib` at a Linux terminal, the makefile creates the four static libraries in the `/lib` folder, one for
each implementation. 
You can then use those libraries to run them through a program of yours.
