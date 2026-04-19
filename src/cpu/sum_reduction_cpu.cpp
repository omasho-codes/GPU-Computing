// Build: g++ -O3 -std=c++17 -o bin/cpu/sum_reduction_cpu src/cpu/sum_reduction_cpu.cpp
// Bench: ./bin/cpu/sum_reduction_cpu --bench --n 1048576 --iters 20 --warmup 3

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <cmath>
#include <vector>

static float reduce_sum(const float* a, int n)
{
	float s = 0.0f;
	for (int i = 0; i < n; i++) s += a[i];
	return s;
}

int main(int argc, char** argv)
{
	bool bench = false;
	int n = 1 << 20;
	int iters = 20;
	int warmup = 3;
	for (int i = 1; i < argc; i++) {
		if (!strcmp(argv[i], "--bench"))       bench = true;
		else if (!strcmp(argv[i], "--n"))      n = atoi(argv[++i]);
		else if (!strcmp(argv[i], "--iters"))  iters = atoi(argv[++i]);
		else if (!strcmp(argv[i], "--warmup")) warmup = atoi(argv[++i]);
	}
	if (!bench) { fprintf(stderr, "cpu baseline: run with --bench\n"); return 1; }

	std::vector<float> a(n);
	for (int i = 0; i < n; i++) a[i] = (float)((i % 4096) - 2048) * 0.001f;

	volatile float sink = 0.0f;
	for (int i = 0; i < warmup; i++) sink += reduce_sum(a.data(), n);

	double sum_ms = 0.0, sum_sq = 0.0;
	for (int it = 0; it < iters; it++) {
		auto t0 = std::chrono::high_resolution_clock::now();
		sink += reduce_sum(a.data(), n);
		auto t1 = std::chrono::high_resolution_clock::now();
		double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
		sum_ms += ms;
		sum_sq += ms * ms;
	}
	(void)sink;

	double mean = sum_ms / iters;
	double var  = (sum_sq / iters) - mean * mean;
	if (var < 0) var = 0;
	double stddev = std::sqrt(var);
	double gbps = ((double)n * sizeof(float)) / 1e9 / (mean / 1000.0);

	printf("{\"kernel\": \"sum_reduction\", \"backend\": \"cpu\", \"n\": %d, "
	       "\"iters\": %d, \"ms_mean\": %.6f, \"ms_std\": %.6f, \"gbps\": %.3f}\n",
	       n, iters, mean, stddev, gbps);
	return 0;
}
