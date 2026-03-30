#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>

using namespace std;

void generateMatrix(int n, const string& filename) {
    ofstream fout(filename);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            fout << (rand() % 100) / 10.0;
            if (j < n - 1) fout << " ";
        }
        fout << "\n";
    }
    fout.close();
}

int main() {
    srand(time(0));
    
    int sizes[] = {200, 400, 800, 1200, 1600, 2000};
    int numSizes = sizeof(sizes) / sizeof(sizes[0]);
    
    // Create data folder (Windows)
    system("mkdir data 2>nul");
    // For Linux/Mac: system("mkdir -p data");
    
    for (int i = 0; i < numSizes; i++) {
        int n = sizes[i];
        string fileA = "data/matrixA_" + to_string(n) + ".txt";
        string fileB = "data/matrixB_" + to_string(n) + ".txt";
        
        cout << "Generating " << fileA << "..." << endl;
        generateMatrix(n, fileA);
        cout << "Generating " << fileB << "..." << endl;
        generateMatrix(n, fileB);
    }
    
    cout << "All matrices generated!" << endl;
    return 0;
}
