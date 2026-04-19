// Build: g++ -O3 -std=c++17 -o bin/cpu/avg_image_blur_cpu src/cpu/avg_image_blur_cpu.cpp
// Bench: ./bin/cpu/avg_image_blur_cpu --bench --n 1024 --p 5 --iters 10 --warmup 2

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <cmath>
#include <vector>

static void blur(int n, int p, const float* a, float* b)
{
	int half = p / 2;
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			float sum = 0.0f;
			int pixels = 0;
			for (int pi = -half; pi <= half; pi++) {
				int ni = i + pi;
				if (ni < 0 || ni >= n) continue;
				for (int pj = -half; pj <= half; pj++) {
					int nj = j + pj;
					if (nj < 0 || nj >= n) continue;
					sum += a[ni*n + nj];
					pixels++;
				}
			}
			b[i*n + j] = sum / (float)pixels;
		}
	}
}

int main(int argc, char** argv)
{
	bool bench = false;
	int n = 512, p = 5;
	int iters = 10, warmup = 2;
	for (int i = 1; i < argc; i++) {
		if (!strcmp(argv[i], "--bench"))       bench = true;
		else if (!strcmp(argv[i], "--n"))      n = atoi(argv[++i]);
		else if (!strcmp(argv[i], "--p"))      p = atoi(argv[++i]);
		else if (!strcmp(argv[i], "--iters"))  iters = atoi(argv[++i]);
		else if (!strcmp(argv[i], "--warmup")) warmup = atoi(argv[++i]);
	}
	if (!bench) { fprintf(stderr, "cpu baseline: run with --bench\n"); return 1; }

	std::vector<float> a(n*n), b(n*n);
	for (int i = 0; i < n*n; i++) a[i] = (float)((i * 2654435761u) & 0xff) / 255.0f;

	for (int i = 0; i < warmup; i++) blur(n, p, a.data(), b.data());

	double sum_ms = 0.0, sum_sq = 0.0;
	for (int it = 0; it < iters; it++) {
		auto t0 = std::chrono::high_resolution_clock::now();
		blur(n, p, a.data(), b.data());
		auto t1 = std::chrono::high_resolution_clock::now();
		double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
		sum_ms += ms;
		sum_sq += ms * ms;
	}

	double mean = sum_ms / iters;
	double var  = (sum_sq / iters) - mean * mean;
	if (var < 0) var = 0;
	double stddev = std::sqrt(var);
	double gflops = ((double)n * n * (p * p + 1)) / 1e9 / (mean / 1000.0);

	printf("{\"kernel\": \"avg_image_blur\", \"backend\": \"cpu\", \"n\": %d, \"p\": %d, "
	       "\"iters\": %d, \"ms_mean\": %.6f, \"ms_std\": %.6f, \"gflops\": %.3f}\n",
	       n, p, iters, mean, stddev, gflops);
	return 0;
}
