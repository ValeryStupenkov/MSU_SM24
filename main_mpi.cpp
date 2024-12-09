#include <iostream>
#include "omp.h"
#include <math.h>
#include <sys/time.h>
#include <vector>
#include <cstdlib>
#include <chrono>
#include <mpi.h>

using namespace std;

class Grid { 
    private:
    	int x_size, y_size, z_size;
    public:
    	double* grid;
    
    	Grid(int _x_size, int _y_size, int _z_size) {
    		x_size = _x_size;
    		y_size = _y_size;
    		z_size = _z_size;
        	grid = (double*)calloc(x_size * y_size * z_size, sizeof(double));
    	}
    	
    	double& operator()(int i, int j, int k){
            return grid[i * z_size * y_size + j * z_size + k];
	}	
	
	double* operator[](int offset){
            return grid + offset;
	}
	    
    	~Grid() {
    	    free(grid);
    	}

};

void help() {
    cout << "Required arguments: N L dt Num_steps Num_threads" << endl;
}

double u_analytical(double x, double y, double z, double Lx, double Ly, double Lz, double t) {
    double a = M_PI * sqrt(9.0 / (Lx * Lx) + 4.0 / (Ly * Ly) + 4.0 / (Lz * Lz));
    return sin(3.0 * M_PI * x / Lx) * sin(2.0 * M_PI * y / Ly) * sin(2.0 * M_PI * z / Lz) * cos(a * t + 4.0 * M_PI);
}

double timer() {
    struct timeval tp;
    struct timezone tzp;
    gettimeofday(&tp, &tzp);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}


int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int rank, world_size;

    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (argc != 5 && argc != 6) {
        help();
        exit(0);
    }

    int N = atoi(argv[1]);
    int inputL = atoi(argv[2]);
    double L = inputL ? (double)inputL : M_PI;
    double Lx = L;
    double Ly = L;
    double Lz = L;
    double dt = stod(argv[3]);
    int n_steps = atoi(argv[4]);
    
    int num_threads = omp_get_max_threads();
    if (argc == 6)
    	num_threads = atoi(argv[5]);
    omp_set_num_threads(num_threads);
    
    double dx = Lx / (double)(N-1);
    double dy = Ly / (double)(N-1);
    double dz = Lz / (double)(N-1);
    double t = 0.0;
    
    int dims[3] = {0, 0, 0};
    MPI_Dims_create(world_size, 3, dims);
    
    if (N % dims[0] || N % dims[1] || N % dims[2]) {
        if (!rank)
            cerr << "Grid size and number of processes are incompatible" << endl;
        MPI_Finalize();
        return 1;
    }
    
    int size_x = N / dims[0];
    int size_y = N / dims[1];
    int size_z = N / dims[2];
    
    Grid u_prev = Grid(size_x, size_y, size_z);
    Grid u_curr = Grid(size_x, size_y, size_z);
    
    int periods[3] = {true, true, true};
    MPI_Comm communicator;
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, true, &communicator);

    int proc_rank;
    MPI_Comm_rank(communicator, &proc_rank);

    int proc_coords[3];
    MPI_Cart_coords(communicator, proc_rank, 3, proc_coords);
    
    double* yz_prev = (double*)calloc(size_y * size_z, sizeof(double));
    double* yz_next = (double*)calloc(size_y * size_z, sizeof(double));

    double* xz_prev = (double*)calloc(size_x * size_z, sizeof(double));
    double* xz_next = (double*)calloc(size_x * size_z, sizeof(double));

    double* xy_prev = (double*)calloc(size_x * size_y, sizeof(double));
    double* xy_next = (double*)calloc(size_x * size_y, sizeof(double));

    int back;
    if (proc_coords[1] == 0)
        back = rank + (dims[1]-1) * dims[2];
    else
    	back = rank - dims[2];

    int front;
    if (proc_coords[1] == dims[1]-1)
        front = rank - (dims[1]-1) * dims[2];
    else
    	front = rank + dims[2];
    	
    int left;
    if (proc_coords[0] == 0)
        left = rank + (dims[0]-1) * dims[1] * dims[2];
    else
    	left = rank - dims[1] * dims[2];

    int right;
    if (proc_coords[0] == dims[0]-1)
        right = rank - (dims[0]-1) * dims[1] * dims[2];
    else
    	right = rank + dims[1] * dims[2];

    int below;
    if (proc_coords[2] == 0)
        below = rank + (dims[2] - 1);
    else
    	below = rank - 1;

    int above;
    if (proc_coords[2] == dims[2]-1)
        above = rank - (dims[2] - 1);
    else
    	above = rank + 1;

    int y_offset_prev;
    if (proc_coords[1] == dims[1]-1)
        y_offset_prev = size_y - 2;
    else
    	y_offset_prev = size_y - 1;

    int y_offset_next;
    if (proc_coords[1] == 0)
        y_offset_next = 1;
    else
    	y_offset_next = 0;

    int z_offset_prev;
    if (proc_coords[2] == dims[2]-1)
        z_offset_prev = size_z - 2;
    else
    	z_offset_prev = size_z - 1;

    int z_offset_next;
    if (proc_coords[2] == 0)
        z_offset_next = 1;
    else
    	z_offset_next = 0;

    int from = 0;
    int to = size_x;

    if (proc_coords[0] == 0)
        from = 1;
    if (proc_coords[0] == dims[0]-1)
        to = size_x-1;
        
    MPI_Datatype MPI_XZ_PLANE;
    MPI_Type_vector(size_x, size_z, size_y * size_z, MPI_DOUBLE, &MPI_XZ_PLANE);
    MPI_Type_commit(&MPI_XZ_PLANE);

    MPI_Datatype MPI_XY_PLANE;
    MPI_Type_vector(size_y * size_x, 1, size_z, MPI_DOUBLE, &MPI_XY_PLANE);
    MPI_Type_commit(&MPI_XY_PLANE);

    MPI_Barrier(MPI_COMM_WORLD);
    
    double t1 = timer();
    
    #pragma omp parallel for
    for (int i = 0; i < size_x; i++) {
        for (int j = 0; j < size_y; j++) {
            for (int k = 0; k < size_z; k++) {
                int x_offset = proc_coords[0] * size_x + i;
                int y_offset = proc_coords[1] * size_y + j;
                int z_offset = proc_coords[2] * size_z + k;
                u_prev(i,j,k) = u_analytical(x_offset * dx, y_offset * dy, z_offset * dz, Lx, Ly, Lz, t);
                u_curr(i,j,k) = u_analytical(x_offset * dx, y_offset * dy, z_offset * dz, Lx, Ly, Lz, t+dt);
            }
        }
    }
    t += 2*dt;
    
    for (int step = 0; step < n_steps; step++) {
        MPI_Sendrecv(u_curr[(size_x - 1) * size_y * size_z], size_y*size_z, MPI_DOUBLE, right, 0, yz_prev, size_y*size_z, MPI_DOUBLE, left , 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        MPI_Sendrecv(u_curr[0], size_y*size_z, MPI_DOUBLE, left , 0, yz_next, size_y*size_z, MPI_DOUBLE, right, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        MPI_Sendrecv(u_curr[y_offset_prev * size_z], 1, MPI_XZ_PLANE, front, 0, xz_prev, size_x*size_z, MPI_DOUBLE, back , 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        MPI_Sendrecv(u_curr[y_offset_next * size_z], 1, MPI_XZ_PLANE, back,  0, xz_next, size_x*size_z, MPI_DOUBLE, front, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        MPI_Sendrecv(u_curr[z_offset_prev], 1, MPI_XY_PLANE, above, 0, xy_prev, size_x*size_y, MPI_DOUBLE, below, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        MPI_Sendrecv(u_curr[z_offset_next], 1, MPI_XY_PLANE, below, 0, xy_next, size_x*size_y, MPI_DOUBLE, above, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        #pragma omp parallel for
        for (int i = from; i < to; i++) {
            for (int j = 0; j < size_y; j++) {
                for (int k = 0; k < size_z; k++) {
                    double x_diff, y_diff, z_diff;

                    if (i == 0) {
                        x_diff = yz_prev[j*size_z + k] - 2 * u_curr(i,j,k) + u_curr(i+1,j,k);
                    } else if (i == size_x-1) {
                        x_diff = u_curr(i-1,j,k) - 2 * u_curr(i,j,k) + yz_next[j*size_z + k];
                    } else {
                        x_diff = u_curr(i-1,j,k) - 2 * u_curr(i,j,k) + u_curr(i+1,j,k);
                    }

                    if (j == 0) {
                        y_diff = xz_prev[i*size_z+k] - 2 * u_curr(i,j,k) + u_curr(i,j+1,k);
                    } else if (j == size_y-1) {
                        y_diff = u_curr(i,j-1,k) - 2 * u_curr(i,j,k) + xz_next[i*size_z+k];
                    } else {
                        y_diff = u_curr(i,j-1,k) - 2 * u_curr(i,j,k) + u_curr(i,j+1,k);
                    }

                    if (k == 0) {
                        z_diff = xy_prev[i*size_y+j] - 2 * u_curr(i,j,k) + u_curr(i,j,k+1);
                    } else if (k == size_z-1) {
                        z_diff = u_curr(i,j,k-1) - 2 * u_curr(i,j,k) + xy_next[i*size_y+j];
                    } else {
                        z_diff = u_curr(i,j,k-1) - 2 * u_curr(i,j,k) + u_curr(i,j,k+1);
                    }

                    double laplas = x_diff / (dx * dx) + y_diff / (dy * dy) + z_diff / (dz * dz);

                    u_prev(i,j,k) =  dt * dt * laplas - u_prev(i,j,k) + 2 * u_curr(i,j,k);
                }
            }
        }

        swap(u_prev.grid, u_curr.grid);
        t += dt;
    }
    
    double max_res = 0.0;
    
    #pragma omp parallel for reduction(max:max_res)
    for (int i = 0; i < size_x; i++) {
        for (int j = 0; j < size_y; j++) {
            for (int k = 0; k < size_z; k++) {
                int x_offset = proc_coords[0] * size_x + i;
                int y_offset = proc_coords[1] * size_y + j;
                int z_offset = proc_coords[2] * size_z + k;
                double u_anal_ijk = u_analytical(x_offset * dx, y_offset * dy, z_offset * dz, Lx, Ly, Lz, t-dt);
                double res = fabs(u_curr(i,j,k) - u_anal_ijk);
                max_res = max(res, max_res);
            }
        }
    }
    
    double residual;
    MPI_Allreduce(&max_res, &residual, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    
    MPI_Barrier(MPI_COMM_WORLD);
    double t2 = timer();
    
    if (!rank){
    	cout << "Number of threads: " << omp_get_max_threads() << endl;
    	cout << "Maximum residual: " << residual << endl;    
    	cout << "Time passed: " << t2 - t1 << endl;
    	cout << "Grid size: " << dims[0] << " " << dims[1] << " " << dims[2] << endl;
    }
    
    free(xy_prev);
    free(xy_next);
    free(xz_prev);
    free(xz_next);
    free(yz_prev);
    free(yz_next);

    MPI_Finalize();

    return 0;
}
