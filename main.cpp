#include <iostream>
#include <ctime>
#include <cstdlib>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <string>

using namespace std;
using namespace std::chrono;

const string fileName = "../results.csv";

int** getMatrixWithRandomValues(int size){
    int** matrix = new int*[size];
    for (int i=0; i<size; i++){
        matrix[i] = new int[size];
        for(int j=0; j<size; j++)
            matrix[i][j] = rand()%10 + 1;
    }
    return matrix;
}

int** zeros(int size){
    int** matrix = new int*[size];
    for (int i=0; i<size; i++){
        matrix[i] = new int[size];
        for(int j=0; j<size; j++)
            matrix[i][j] = 0;
    }
    return matrix;
}

void deleteMatrix(int size, int** matrix){
    for (int i = 0; i < size; ++i)
        delete [] matrix[i];
    delete [] matrix;
}

void appendResultToFile(int size, const string& functionName, milliseconds time){
    fstream file;
    file.open(fileName, ofstream::app);
    file << size << "," << functionName << "," << time.count() << "\n";
    file.close();
}

int** mul_ijk(int size, int** A, int** B){
    int** result = zeros(size);

    for(int i=0; i<size ;i++)
        for(int j=0; j<size; j++)
            for (int k = 0; k < size; k++)
                result[i][j] += A[i][k] * B[k][j];

    return result;
}

int** mul_ikj(int size, int** A, int** B){
    int** result = zeros(size);

    for(int i=0; i<size ;i++)
        for(int k=0; k<size; k++)
            for (int j = 0; j < size; j++)
                result[i][j] += A[i][k] * B[k][j];

    return result;
}

int** mul_jik(int size, int** A, int** B){
    int** result = zeros(size);

    for(int j=0; j < size ; j++)
        for(int i=0; i < size; i++)
            for (int k = 0; k < size; k++)
                result[i][j] += A[i][k] * B[k][j];

    return result;
}

int** mul_jki(int size, int** A, int** B){
    int** result = zeros(size);

    for(int j=0; j < size ; j++)
        for(int k=0; k < size; k++)
            for (int i = 0; i < size; i++)
                result[i][j] += A[i][k] * B[k][j];

    return result;
}

int** mul_kij(int size, int** A, int** B){
    int** result = zeros(size);

    for(int k=0; k < size ; k++)
        for(int i=0; i < size; i++)
            for (int j = 0; j < size; j++)
                result[i][j] += A[i][k] * B[k][j];

    return result;
}

int** mul_kji(int size, int** A, int** B){
    int** result = zeros(size);

    for(int k=0; k < size ; k++)
        for(int j=0; j<size; j++)
            for (int i = 0; i < size; i++)
                result[i][j] += A[i][k] * B[k][j];

    return result;
}

void execute(int size){
    int** (*functions[])(int, int**, int**)  = {&mul_ijk, &mul_ikj, &mul_jik, &mul_jki, &mul_kij, &mul_kji};
    string functionNames[] = {"ijk", "ikj", "jik", "jki", "kij", "kji"};

    auto A = getMatrixWithRandomValues(size);
    auto B = getMatrixWithRandomValues(size);

    for(int i=0; i < 6; i++){
        auto start = high_resolution_clock::now();

        auto result = functions[i](size, A, B);

        auto finish = high_resolution_clock::now();
        auto elapsed = duration_cast<milliseconds>(finish - start);
        cout <<  size << ", " << functionNames[i] << ", " << elapsed.count() <<  "ms\n";
        appendResultToFile(size, functionNames[i],elapsed);

        deleteMatrix(size, result);
    }

    deleteMatrix(size, A);
    deleteMatrix(size, B);
}

int main() {
    srand(time(0));

    int sizes[] = {10, 100, 1000};

    for(int size: sizes){
        execute(size);
    }

    return 0;
}
