Instructions on how to run 

1. Compilation
First, compile the project by running the following command in the terminal:
nvcc src/assignment2_subtask1.cu -o subtask1
nvcc src/assignment2_subtask2.cu -o subtask2
nvcc src/assignment2_subtask3.cu -o subtask3
nvcc src/assignment2_subtask4.cu -o subtask4

This will compile all subtasks and generate the executables.

2. Running Subtasks

Subtask 1
To run Subtask 1, execute the following command:
./subtask1 [task of choice 1=convolution, 2=non-linear-activations, 3=subsampling, 4=converting a vector]

Subtask 2
To run Subtask 2, execute the following command:
./subtask2 [task of choice 1=convolution, 2=non-linear-activations, 3=subsampling, 4=converting a vector]

Subtask 3
To run Subtask 3, execute the following command:
./subtask3

Subtask 4
To run Subtask 4, execute the following command:
./subtask4 [1 - with streams, 0 - without streams] 

3. Cleaning Up
To clean up the generated files and executables, run:
make clean
This will remove all object files, executables, and temporary files.
