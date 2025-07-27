#include "../../hnswlib/hnswlib.h"

#include <map>
#include <iomanip>
#include <iostream>
#include <vector>
#include <set>

#include <random>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <unordered_set>
#include <limits>

typedef std::chrono::high_resolution_clock Clock;
int main() {
    using namespace hnswlib;

    // --- Trade-off parameters ---
    size_t ef_search = 100;                     // higher ef => higher recall, slower search
    float recall     = 1.0f;                    // target recall in reverseSearchKnn
    double density_ratio_threshold = 0.5;       // keep candidates with density >= ratio * query_density

    // 1. Generate clustered data
    size_t dim = 15;
    size_t clusters = 4;
    size_t pts_per_cluster = 2500;
    size_t N = clusters * pts_per_cluster;

    std::mt19937 rng(25);
    std::uniform_real_distribution<float> noise(0.0f, 0.5f);
    std::vector<std::vector<float>> data(N, std::vector<float>(dim));
    std::vector<labeltype> labels(N);
    std::vector<std::vector<float>> centers = {{0,0}, {50,50}, {0,5}, {5,0}};

    for (size_t c = 0; c < clusters; ++c) {
        for (size_t i = 0; i < pts_per_cluster; ++i) {
            size_t idx = c * pts_per_cluster + i;
            data[idx][0] = centers[c][0] + noise(rng);
            data[idx][1] = centers[c][1] + noise(rng);
            labels[idx] = idx;
        }
    }

    // 2. Build HNSW index
    L2Space space(dim);
    size_t M = 6, ef_c = 100;
    HierarchicalNSW<float> hnsw(&space, N, M, ef_c, /*seed=*/1414);

    // Set search breadth for k-NN calls
    hnsw.setEf(ef_search);

    // uniform pr(h) distribution for hop-count estimation
    size_t H = (size_t)std::ceil(std::log((double)N) / std::log((double)M));
    std::vector<double> pr(H+1, 1.0/H);
    hnsw.setPrDistribution(pr);

    // density-based pruning threshold
    hnsw.setDensityRatioThreshold(density_ratio_threshold);

    // insert points
    for (size_t i = 0; i < N; ++i)
        hnsw.addPoint(data[i].data(), labels[i]);

    // 3. Precompute brute-force reverse thresholds (forward k-NN threshold)
    size_t k = 10;
    std::vector<double> thresholds(N);
    auto t0 = Clock::now();
    for (size_t o = 0; o < N; ++o) {
        std::vector<double> ds;
        ds.reserve(N-1);
        for (size_t j = 0; j < N; ++j) {
            if (j == o) continue;
            ds.push_back(space.get_dist_func()(data[o].data(), data[j].data(), space.get_dist_func_param()));
        }
        if (ds.size() < k) {
            thresholds[o] = std::numeric_limits<double>::infinity();
        } else {
            std::nth_element(ds.begin(), ds.begin() + (k), ds.end());
            thresholds[o] = ds[k-1];
        }
    }
    auto t1 = Clock::now();
    double thresh_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // 4. Select query points (one from each cluster)
    std::vector<size_t> queries = {10, 300,89,529,493,888,875,823,260, 510, 332,523,359,760,33,42,99};
    // std::vector<size_t> queries = {510, 760,33,42,99};

    // 5. Output CSV header including trade-off settings
    std::cout << "ef_search," << ef_search
              << ",recall," << recall
              << ",density_ratio," << density_ratio_threshold
              << ",precompute_threshold_ms," << thresh_ms << std::endl;
    std::cout << "query,method,size,time_ms,accuracy" << std::endl;

    // 6. Run queries
    for (auto ql : queries) {
        const auto &qv = data[ql];

        // 6a. Approximate reverse k-ANN
        t0 = Clock::now();
        auto approx = hnsw.reverseSearchKnn(qv.data(), k, recall);
        t1 = Clock::now();
        double t_approx = std::chrono::duration<double, std::milli>(t1 - t0).count();

        // 6b. Brute-force reverse k-ANN via forward threshold
        t0 = Clock::now();
        std::vector<labeltype> brute;
        brute.reserve(N);
        for (size_t o = 0; o < N; ++o) {
            if (o == ql) continue;
            double d_o_q = space.get_dist_func()(qv.data(), data[o].data(), space.get_dist_func_param());
            if (d_o_q <= thresholds[o]) brute.push_back(labels[o]);
        }
        t1 = Clock::now();
        double t_brute = std::chrono::duration<double, std::milli>(t1 - t0).count();

        // 6c. Compute accuracy = |approx ∩ brute| / |brute|
        std::unordered_set<labeltype> Sbr(brute.begin(), brute.end());
        size_t inter = 0;
        for (auto lbl : approx) if (Sbr.count(lbl)) ++inter;
        double accuracy = brute.empty() ? 1.0 : double(inter) / brute.size();

        // 6d. Print results
        std::cout << ql << ",approx ," << approx.size() << ", time(ms)="<< std::fixed << std::setprecision(2) << t_approx << "," << accuracy*100<<"%" << std::endl;
        std::cout << ql << ",brute  ,"  << brute.size()  << ", time(ms)=" << t_brute     << std::endl;
    }


return 0;
}
