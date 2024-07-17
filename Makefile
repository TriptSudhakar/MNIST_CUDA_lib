# Define the C++ compiler
CXX = g++

# Define the CUDA compiler
NVCC = nvcc

# Define any compile-time flags for C++
CXXFLAGS = -Wall -Wextra -O2

# Define any compile-time flags for CUDA
NVCCFLAGS = 

# Define any directories containing header files
INCLUDES = 

# Output directory
OUTDIR = output

# Subtask 1 files and executable
CPP_SRCS1 = src/assignment2_subtask1.cpp
MAIN1 = subtask1

# Subtask 2 files and executable
CUDA_SRCS2 = src/assignment2_subtask2.cu 
MAIN2 = subtask2

# Subtask 3 files and executable
CUDA_SRCS3 = src/assignment2_subtask3.cu 
MAIN3 = subtask3

# Subtask 4 files and executable
CUDA_SRCS4 = src/assignment2_subtask4.cu 
MAIN4 = subtask4

FILES = $(python3 preprocessing.py)

.PHONY: clean all subtask1 subtask2 subtask3 subtask4

all: subtask1 subtask2 subtask3 subtask4


$(MAIN1): $(CPP_SRCS1)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $(MAIN1) $(CPP_SRCS1)

$(MAIN2): $(CUDA_SRCS2)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $(MAIN2) $(CUDA_SRCS2)

$(MAIN3): $(CUDA_SRCS3) 
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $(MAIN3) $(CUDA_SRCS3)

$(MAIN4): $(CUDA_SRCS4)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $(MAIN4) $(CUDA_SRCS4)

run_python_script:
	python3 preprocessing.py

clean:
	$(RM) *.o *~ $(MAIN1) $(MAIN2) $(MAIN3) $(MAIN4)