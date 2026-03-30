#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <chrono>
#include <iomanip>
#include <omp.h>
#include <cmath>

using namespace std;
using namespace chrono;

typedef vector<vector<double>> Matrix;

bool readMatrix(const string& filename, Matrix& mat) {
    ifstream fin(filename.c_str());
    if (!fin) {
        cerr << "Error: cannot open file " << filename << endl;
        return false;
    }

    string line;
    while (getline(fin, line)) {
        istringstream iss(line);
        vector<double> row;
        double value;
        while (iss >> value)
            row.push_back(value);
        if (!row.empty())
            mat.push_back(row);
    }
    fin.close();
    return (!mat.empty() && mat.size() == mat[0].size());
}

void writeMatrix(const string& filename, const Matrix& mat) {
    ofstream fout(filename.c_str());
    for (size_t i = 0; i < mat.size(); i++) {
        for (size_t j = 0; j < mat[i].size(); j++) {
            fout << fixed << setprecision(6) << mat[i][j];
            if (j < mat[i].size() - 1) fout << " ";
        }
        fout << "\n";
    }
    fout.close();
}

Matrix multiplySequential(const Matrix& A, const Matrix& B) {
    size_t n = A.size();
    Matrix C(n, vector<double>(n, 0.0));
    for (size_t i = 0; i < n; ++i)
        for (size_t k = 0; k < n; ++k) {
            double aik = A[i][k];
            for (size_t j = 0; j < n; ++j)
                C[i][j] += aik * B[k][j];
        }
    return C;
}

Matrix multiplyParallel(const Matrix& A, const Matrix& B, int numThreads) {
    size_t n = A.size();
    Matrix C(n, vector<double>(n, 0.0));
    
    omp_set_num_threads(numThreads);
    
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < n; ++i) {
        for (size_t k = 0; k < n; ++k) {
            double aik = A[i][k];
            for (size_t j = 0; j < n; ++j) {
                C[i][j] += aik * B[k][j];
            }
        }
    }
    return C;
}

bool compareMatrices(const Matrix& C1, const Matrix& C2, double epsilon = 1e-6) {
    if (C1.size() != C2.size()) return false;
    size_t n = C1.size();
    for (size_t i = 0; i < n; i++) {
        if (C1[i].size() != C2[i].size()) return false;
        for (size_t j = 0; j < n; j++) {
            double diff = C1[i][j] - C2[i][j];
            if (diff < 0) diff = -diff;
            if (diff > epsilon) {
                return false;
            }
        }
    }
    return true;
}

int main(int argc, char* argv[]) {
    int threads[] = {1, 2, 4, 8};
    int numThreadConfigs = sizeof(threads) / sizeof(threads[0]);
    
    int sizes[] = {200, 400, 800, 1200, 1600, 2000};
    int numSizes = sizeof(sizes) / sizeof(sizes[0]);
    
    cout << "================================================================" << endl;
    cout << "Laboratory Work #2: Parallel Matrix Multiplication (OpenMP)" << endl;
    cout << "================================================================" << endl;
    
    #ifdef _OPENMP
        cout << "OpenMP supported. Max threads: " << omp_get_max_threads() << endl;
    #else
        cout << "OpenMP NOT supported. Check compilation flags." << endl;
        return 1;
    #endif
    
    cout << "\n=== EXPERIMENTS ===" << endl;
    
    vector<vector<double>> times(numSizes, vector<double>(numThreadConfigs + 1));
    vector<vector<double>> speedups(numSizes, vector<double>(numThreadConfigs));
    
    for (int s = 0; s < numSizes; s++) {
        int n = sizes[s];
        
        string fileA = "data/matrixA_" + to_string(n) + ".txt";
        string fileB = "data/matrixB_" + to_string(n) + ".txt";
        
        cout << "\n--- Matrix size: " << n << " x " << n << " ---" << endl;
        
        Matrix A, B;
        if (!readMatrix(fileA, A)) {
            cerr << "Skipping size " << n << ": file " << fileA << " not found" << endl;
            continue;
        }
        if (!readMatrix(fileB, B)) {
            cerr << "Skipping size " << n << ": file " << fileB << " not found" << endl;
            continue;
        }
        
        cout << "Sequential run..." << endl;
        auto start = high_resolution_clock::now();
        Matrix C_seq = multiplySequential(A, B);
        auto end = high_resolution_clock::now();
        double seqTime = duration<double>(end - start).count();
        times[s][0] = seqTime;
        cout << "  Time: " << fixed << setprecision(4) << seqTime << " sec" << endl;
        
        for (int t = 0; t < numThreadConfigs; t++) {
            int numThread = threads[t];
            cout << "Parallel run (" << numThread << " threads)..." << endl;
            
            start = high_resolution_clock::now();
            Matrix C_par = multiplyParallel(A, B, numThread);
            end = high_resolution_clock::now();
            double parTime = duration<double>(end - start).count();
            times[s][t + 1] = parTime;
            
            if (compareMatrices(C_seq, C_par)) {
                double speedup = seqTime / parTime;
                speedups[s][t] = speedup;
                cout << "  Time: " << fixed << setprecision(4) << parTime << " sec"
                     << " | Speedup: " << fixed << setprecision(2) << speedup << "x" << endl;
            } else {
                cout << "  ERROR: results do not match sequential!" << endl;
                speedups[s][t] = 0;
            }
        }
    }
    
    cout << "\n\n================================================================" << endl;
    cout << "SUMMARY TABLE - TIME (seconds)" << endl;
    cout << "================================================================" << endl;
    cout << "Size    | Sequential | ";
    for (int t = 0; t < numThreadConfigs; t++) {
        cout << threads[t] << " thr | ";
    }
    cout << endl;
    cout << "--------|------------|";
    for (int t = 0; t < numThreadConfigs; t++) {
        cout << "-------|";
    }
    cout << endl;
    
    for (int s = 0; s < numSizes; s++) {
        if (times[s][0] == 0) continue;
        cout << setw(6) << sizes[s] << " | ";
        cout << fixed << setprecision(4) << times[s][0] << " | ";
        for (int t = 0; t < numThreadConfigs; t++) {
            cout << fixed << setprecision(4) << times[s][t + 1] << " | ";
        }
        cout << endl;
    }
    
    cout << "\n\n================================================================" << endl;
    cout << "SPEEDUP TABLE (S = T_seq / T_par)" << endl;
    cout << "================================================================" << endl;
    cout << "Size    | ";
    for (int t = 0; t < numThreadConfigs; t++) {
        cout << threads[t] << " thr | ";
    }
    cout << endl;
    cout << "--------|";
    for (int t = 0; t < numThreadConfigs; t++) {
        cout << "-------|";
    }
    cout << endl;
    
    for (int s = 0; s < numSizes; s++) {
        if (times[s][0] == 0) continue;
        cout << setw(6) << sizes[s] << " | ";
        for (int t = 0; t < numThreadConfigs; t++) {
            if (speedups[s][t] > 0) {
                cout << fixed << setprecision(2) << speedups[s][t] << "x | ";
            } else {
                cout << "   --- | ";
            }
        }
        cout << endl;
    }
    
    cout << "\n================================================================" << endl;
    cout << "Experiments completed!" << endl;
    cout << "================================================================" << endl;
    
    return 0;
}