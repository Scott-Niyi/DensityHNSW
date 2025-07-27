// test_serialize.cpp

#include <iostream>
#include <vector>
#include <random>
#include <cassert>
#include <chrono>
#include <algorithm>
#include <string>

#include "../../hnswlib/hnswlib.h"

using namespace hnswlib;
using Clock = std::chrono::high_resolution_clock;

// ----------------------------------------------------------------------------
// Convert a max-heap of (dist,label) into a sorted ascending-distance vector of (label,dist)
static std::vector<std::pair<labeltype,float>>
pqToSortedVector(std::priority_queue<std::pair<float,labeltype>> pq) {
    std::vector<std::pair<labeltype,float>> v;
    v.reserve(pq.size());
    while (!pq.empty()) {
        auto dl = pq.top();  // dl.first=dist, dl.second=label
        pq.pop();
        v.emplace_back(dl.second, dl.first);
    }
    std::reverse(v.begin(), v.end());
    return v;
}

// Print kNN results (vector of (label,dist))
static void printKnnResults(
    const std::vector<std::pair<labeltype,float>> &res,
    const std::string &tag,
    size_t max_print = 5
) {
    std::cout << tag
              << " (showing up to " << std::min(res.size(), max_print)
              << " of " << res.size() << ")\n";
    for (size_t i = 0; i < res.size() && i < max_print; ++i) {
        std::cout << "  label=" << res[i].first
                  << "  dist="  << res[i].second << "\n";
    }
    std::cout << "--------------------------------\n";
}

// Print reverse‑kNN results (vector of labels)
static void printRknnResults(
    const std::vector<labeltype> &res,
    const std::string &tag,
    size_t max_print = 5
) {
    std::cout << tag
              << " (showing up to " << std::min(res.size(), max_print)
              << " of " << res.size() << ")\n";
    for (size_t i = 0; i < res.size() && i < max_print; ++i) {
        std::cout << "  label=" << res[i] << "\n";
    }
    std::cout << "--------------------------------\n";
}

// ----------------------------------------------------------------------------
int main() {
    // parameters
    const int    dim        = 5;
    const size_t N          = 10000;     // thousands of points
    const size_t maxE       = N;
    const size_t M          = 16;
    const size_t efC        = 200;
    const size_t k          = 10;
    const std::string fname = "large_test_index.bin";

    // 1) build index
    L2Space space(dim);
    HierarchicalNSW<float> index(&space, maxE, M, efC);

    // 2) generate random data in [0,1]^5
    std::mt19937                    rng(234);
    std::uniform_real_distribution<float> ud(0.0f,1.0f);
    std::vector<std::vector<float>> data(N, std::vector<float>(dim));

    for (size_t i = 0; i < N; ++i) {
        for (int d = 0; d < dim; ++d)
            data[i][d] = ud(rng);
        index.addPoint(data[i].data(), i);
    }

    // 3) pick a random query
    std::uniform_int_distribution<size_t> pick(0, N-1);
    const size_t query_label = pick(rng);
    const float* query = data[query_label].data();
    std::cout << "Query label = " << query_label << "\n\n";

    // 4) BEFORE save/load: kNN and reverse‑kNN
    auto raw_knn_before  = index.searchKnn(query, k);           // priority_queue<dist,label>
    auto knn_before      = pqToSortedVector(raw_knn_before);    // vector<label,dist>
    auto rknn_before     = index.reverseSearchKnn(query, k);    // vector<label>

    printKnnResults(knn_before,  "kNN before save");
    printRknnResults(rknn_before,"reverse‑kNN before save");

    // 5) time saveIndex()
    auto t0 = Clock::now();
    index.saveIndex(fname);
    auto save_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                       Clock::now() - t0
                   ).count();
    std::cout << "[TIMING] saveIndex(): " << save_ms << " ms\n\n";

    // 6) time loadIndex() into a fresh object
    HierarchicalNSW<float> index2(&space, /*max_elems=*/0, /*M=*/0, /*efC=*/0);
    t0 = Clock::now();
    index2.loadIndex(fname, &space);
    auto load_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                       Clock::now() - t0
                   ).count();
    std::cout << "[TIMING] loadIndex(): " << load_ms << " ms\n\n";

    // 7) AFTER save/load: rerun searches
    auto knn_after  = pqToSortedVector(index2.searchKnn(query, k));
    auto rknn_after = index2.reverseSearchKnn(query, k);

    printKnnResults(knn_after,  "kNN after load");
    printRknnResults(rknn_after,"reverse‑kNN after load");

    // 8) sanity check: same labels in same order
    assert(knn_before.size()  == knn_after.size());
    assert(rknn_before.size() == rknn_after.size());
    for (size_t i = 0; i < k; ++i) {
        assert(knn_before[i].first  == knn_after[i].first);
        assert(rknn_before[i]       == rknn_after[i]);
    }

    std::cout << "✔️  All checks passed on " << N << " points.\n";
    return 0;
}
