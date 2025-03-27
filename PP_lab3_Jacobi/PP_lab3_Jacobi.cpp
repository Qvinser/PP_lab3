#include <iostream>
#include <fstream>
#include <math.h>
#include <omp.h>
#include <cstdlib> 
using namespace std;


void Mul(int N, double** A, double* x, double* y)
{
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
    double eps = 0.001, norm, * TempX = new double[N], start, end;
    //start=omp_get_wtime();
    for (int k = 0; k < N; k++)
        TempX[k] = X[k];
    int cnt = 0;
    start = omp_get_wtime();
    do {
        for (int i = 0; i < N; i++)
        {
            TempX[i] = F[i];
            for (int g = 0; g < N; g++)
                if (i != g)
                    TempX[i] -= A[i][g] * X[g];
            TempX[i] /= A[i][i];
        }
        norm = abs(X[0] - TempX[0]);
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
    setlocale(0, "");
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

int main()
{
    double** Matrix, * b, * y, * x;
    int n;
    Load(n, Matrix, b);
    cout << "Матрица:\n";
    PrintMatrix(n, Matrix);
    cout << "b: ";
    PrintVec(n, b);
    x = new double[n];
    for (int i = 0; i < n; i++)
        x[i] = 1.0;
    y = new double[n];
    cout << endl << b[1] << endl;
    Jacobi(n, Matrix, b, x);
    cout << "Результат | x: ";
    PrintVec(n, x);
    cout << "A*x=b | b: ";
    Mul(n, Matrix, x, y);
    PrintVec(n, y);
    delete x;
    delete y;
    system("pause");
    return 1;
}