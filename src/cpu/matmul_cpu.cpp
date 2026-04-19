// Build: g++ -O3 -std=c++17 -o bin/cpu/matmul_cpu src/cpu/matmul_cpu.cpp
// Bench: ./bin/cpu/matmul_cpu --bench --n 512 --iters 3 --warmup 1
//
// Serial triple-loop reference. This is the "1 thread, no SIMD tuning" baseline.
// It is intentionally not OpenMP/AVX-tuned so the GPU rows read as honestly
// impressive; "tuned BLAS" is a different conversation.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <cmath>
#include <vector>

static void matmul(int n, const float* a, const float* b, float* c)
{
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			float s = 0.0f;
			for (int k = 0; k < n; k++) s += a[i*n + k] * b[k*n + j];
			c[i*n + j] = s;
		}
	}
}

int main(int argc, char** argv)
{
	bool bench = false;
	int n = 256;
	int iters = 3;
	int warmup = 1;
	for (int i = 1; i < argc; i++) {
		if (!strcmp(argv[i], "--bench"))       bench = true;
		else if (!strcmp(argv[i], "--n"))      n = atoi(argv[++i]);
		else if (!strcmp(argv[i], "--iters"))  iters = atoi(argv[++i]);
		else if (!strcmp(argv[i], "--warmup")) warmup = atoi(argv[++i]);
	}
	if (!bench) { fprintf(stderr, "cpu baseline: run with --bench\n"); return 1; }

	std::vector<float> a(n*n), b(n*n), c(n*n);
	for (int i = 0; i < n*n; i++) {
		a[i] = (float)((i * 1103515245 + 12345) & 0xff) / 255.0f;
		b[i] = a[i];
	}

	for (int i = 0; i < warmup; i++) matmul(n, a.data(), b.data(), c.data());

	double sum_ms = 0.0, sum_sq = 0.0;
	for (int it = 0; it < iters; it++) {
		auto t0 = std::chrono::high_resolution_clock::now();
		matmul(n, a.data(), b.data(), c.data());
		auto t1 = std::chrono::high_resolution_clock::now();
		double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
		sum_ms += ms;
		sum_sq += ms * ms;
	}

	double mean = sum_ms / iters;
	double var  = (sum_sq / iters) - mean * mean;
	if (var < 0) var = 0;
	double stddev = std::sqrt(var);
	double gflops = (2.0 * (double)n * n * n) / 1e9 / (mean / 1000.0);

	printf("{\"kernel\": \"matmul\", \"backend\": \"cpu\", \"n\": %d, "
	       "\"iters\": %d, \"ms_mean\": %.6f, \"ms_std\": %.6f, \"gflops\": %.3f}\n",
	       n, iters, mean, stddev, gflops);
	return 0;
}
