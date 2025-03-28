#include <omp.h>
#include <iostream>
#include <chrono>
using namespace std;

int f(int n) {
	int ceil = (n+1) * 10;
	for (int i = 0; i < 100000000; i++) {
		n = (n * n) % ceil;
	}
	return n;
}

int main() {
	int a[100], b[100];
	// Инициализация массива b
	for (int i = 0; i < 100; i++)
		b[i] = i;
	// Директива OpenMP для распараллеливания цикла
	auto start = std::chrono::steady_clock::now();
#pragma omp parallel for
	for (int i = 0; i < 100; i++)
	{
		a[i] = f(b[i]);
		b[i] = 2 * a[i];
	}
	int result = 0;
	// Далее значения a[i] и b[i] используются, например, так:
#pragma omp parallel for reduction(+ : result)
	for (int i = 0; i < 100; i++)
		result += (a[i] + b[i]);
	std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - start;
	cout << "Time spent: " << elapsed.count() << " seconds." << endl;
	cout << "Result = " << result << endl;
	//
	return 0;
}