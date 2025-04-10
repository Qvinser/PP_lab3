#include <iostream>
#include <fstream>
#include <math.h>
#include <omp.h>
#include <cstdlib> 
#include <random> 
using namespace std;


void Mul(int N, double** A, double* x, double* y)
{
#pragma omp parallel for
    for (int i = 0; i < N; i++)
    {
        y[i] = 0;
        for (int j = 0; j < N; j++)
            y[i] = y[i] + (A[i][j] * x[j]);
    }
}


void PrintMatrix(int N, double** A)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
            cout << A[i][j] << " ";
        cout << endl;
    }
}
void PrintVec(int N, double* A)
{
    for (int i = 0; i < N; i++)
    {
        cout << A[i] << " ";
    }
    cout << endl;
}
void Jacobi(int N, double** A, double* F, double* X)
{
    double eps = 0.001, norm, start, end;
    double* TempX = new double[N];
    //start=omp_get_wtime();
    for (int k = 0; k < N; k++)
        TempX[k] = X[k];
    int cnt = 0;
    start = omp_get_wtime();
    do {
    #pragma omp parallel for
        for (int i = 0; i < N; i++)
        {
            TempX[i] = F[i];
            for (int g = 0; g < N; g++)
                if (i != g)
                    TempX[i] -= A[i][g] * X[g];
            TempX[i] /= A[i][i];
        }
        norm = abs(X[0] - TempX[0]);
    #pragma omp parallel for reduction(min:norm)
        for (int h = 0; h < N; h++)
        {
            if (abs(X[h] - TempX[h]) > norm)
                norm = abs(X[h] - TempX[h]);
            X[h] = TempX[h];
        }
        cnt++;
    } while (norm > eps);
    end = omp_get_wtime();
    cout << "Количество итераций = " << cnt << endl;
    printf_s("Затрачено %f сек\n", (end - start));
    delete[] TempX;
}
void Load(int& N, double**& A, double*& F)
{
    ifstream fin;
    fin.open("input.txt");
    fin >> N;
    F = new double[N];
    A = new double* [N];
    for (int i = 0; i < N; i++)
        A[i] = new double[N];
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
            fin >> A[i][j];
        fin >> F[i];
    }
    fin.close();
}


// Генерация матрицы с диагональным преобладанием
void generateDiagonallyDominantMatrix(int n, double**& A) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist(-5.0, 5.0);
    A = new double*[n];
    for (int i = 0; i < n; ++i) {
        double sum = 0.0;
        A[i] = new double[n];
        for (int j = 0; j < n; ++j) {
            if (i != j) {
                A[i][j] = dist(gen);
                sum += std::abs(A[i][j]);
            }
        }
        // Обеспечим диагональное преобладание
        A[i][i] = sum + dist(gen) + 1.0; // гарантируем, что |a_ii| > сумма остальных
        if (dist(gen) < 0) A[i][i] *= -1;
    }
}

// Генерация вектора правых частей
void generateRHS(int n, double*& F) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist(-10.0, 10.0);
    F = new double[n];
    for (int i = 0; i < n; ++i)
        F[i] = dist(gen);
}

int main()
{
    double** Matrix, * b, * y, * x;
    int n = 5000;
    int thread_num = 1;
    omp_set_num_threads(thread_num);
    setlocale(0, "");
    generateDiagonallyDominantMatrix(n, Matrix);
    generateRHS(n, b);
    //Load(n, Matrix, b);
    if (n<=6) {
        cout << "Матрица:\n";
        PrintMatrix(n, Matrix);
        cout << "b: ";
        PrintVec(n, b);
    }
    x = new double[n];
    for (int i = 0; i < n; i++)
        x[i] = 20.0;
    y = new double[n];
    cout << endl << endl;
    Jacobi(n, Matrix, b, x);
    if (n <= 10) {
        cout << "Результат | x: ";
        PrintVec(n, x);
        cout << "A*x=b | b: ";
        Mul(n, Matrix, x, y);
        PrintVec(n, y);
    }
    delete x;
    delete y;
    for (int i = 0; i < n; i++)
    {
        delete[] Matrix[i];
    }
    delete[] Matrix;
    //system("pause");
    return 1;
}