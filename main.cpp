#include <iostream>
#include "omp.h"
#include <math.h>
#include <sys/time.h>
#include <vector>
#include <cstdlib>
#include <chrono>

using namespace std;

void help() {
    cout << "Required arguments: N L dt Num_steps" << endl;
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

    if (argc != 5) {
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
    
    double dx = Lx / (double)(N-1);
    double dy = Ly / (double)(N-1);
    double dz = Lz / (double)(N-1);
    double t = 0.0;

    vector<vector<vector<double> > > u_prev(N, vector<vector<double> >(N, vector<double>(N, 0)));
    vector<vector<vector<double> > > u_curr(N, vector<vector<double> >(N, vector<double>(N, 0)));
    
    cout << "Number of threads: " << omp_get_max_threads() << endl;

    double t1 = timer();
    
    // Calculate u0 and u1
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                u_prev[i][j][k] = u_analytical(i*dx, j*dy, k*dz, Lx, Ly, Lz, t);
                u_curr[i][j][k] = u_analytical(i*dx, j*dy, k*dz, Lx, Ly, Lz, t+dt);
            }
        }
    }
    t += 2*dt;
    
    for (int step = 0; step < n_steps; step++) {
        #pragma omp parallel for
        for (int i = 1; i < N-1; i++) {
            for (int j = 0; j < N; j++) {
                for (int k = 0; k < N; k++) {
                    double x_d, y_d, z_d;
                    x_d = u_curr[i-1][j][k] - 2 * u_curr[i][j][k] + u_curr[i+1][j][k];

                    if (j == 0 || j == N-1) {
                        y_d = u_curr[i][N-2][k] - 2 * u_curr[i][j][k] + u_curr[i][1][k];
                    } 
                    else {
                        y_d = u_curr[i][j-1][k] - 2 * u_curr[i][j][k] + u_curr[i][j+1][k];
                    }

                    if (k == 0 || k == N-1) {
                        z_d = u_curr[i][j][N-2] - 2 * u_curr[i][j][k] + u_curr[i][j][1];
                    } 
                    else {
                        z_d = u_curr[i][j][k-1] - 2 * u_curr[i][j][k] + u_curr[i][j][k+1];
                    }

                    double laplas_op = x_d / (dx * dx) + y_d / (dy * dy) + z_d / (dz * dz);

                    u_prev[i][j][k] =  dt * dt * laplas_op - u_prev[i][j][k] + 2 * u_curr[i][j][k];
                }
            }
        }
	
        swap(u_prev, u_curr);
        t += dt;
    }
    
    double max_res = 0.0;
    
    #pragma omp parallel for reduction(max:max_res)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                double u_anal_ijk = u_analytical(i*dx, j*dy, k*dz, Lx, Ly, Lz, t-dt);
                double res = abs(u_curr[i][j][k] - u_anal_ijk);
                max_res = max(res, max_res);
            }
        }
    }
    
    cout << "Maximum residual: " << max_res << endl;

    double t2 = timer();
    cout << "Time passed: " << t2 - t1 << endl;

    return 0;
}
