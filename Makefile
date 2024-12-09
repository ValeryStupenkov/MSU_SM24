CXX = g++
MPICXX = mpicxx
CXXFLAGS = -O3 -w -Wall -g -fopenmp

all: clean prog prog_mpi

prog:
	$(CXX) $(CXXFLAGS) main.cpp -o prog
	
prog_mpi:
	$(MPICXX) $(CXXFLAGS) -std=gnu++11 main_mpi.cpp -o prog_mpi

clean::
	rm -f prog*
