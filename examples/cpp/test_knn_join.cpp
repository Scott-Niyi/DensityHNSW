// --------------------------------------------------------------
// test_knn_join.cpp
// --------------------------------------------------------------
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <iomanip>

#include "../../hnswlib/hnswlib.h"

using Clock = std::chrono::high_resolution_clock;
using hnswlib::HierarchicalNSW;
using hnswlib::labeltype;

template<typename T>
double l2sq(const T* a, const T* b, size_t dim) {
    double s = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        double d = double(a[i]) - double(b[i]);
        s += d * d;
    }
    return s;
}

int main(int argc, char** argv) {
    /* ---------- parameters ---------- */
    const size_t dim       = 40;
    size_t       nA        = (argc > 1) ? std::stoul(argv[1]) : 1000;
    size_t       nB        = (argc > 2) ? std::stoul(argv[2]) : 1000;
    const size_t k         = 10;
    const size_t nCheck    = 200;    // how many points to brute-force for recall

    const size_t M               = 16;
    const size_t efConstruction  = 200;
    const size_t randomSeed      = 42;

    std::mt19937  rng(randomSeed);
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);

    /* ---------- generate data ---------- */
    std::vector<std::vector<float>> A(nA, std::vector<float>(dim));
    std::vector<std::vector<float>> B(nB, std::vector<float>(dim));
    for (auto& v : A) for (auto& x : v) x = uni(rng);
    for (auto& v : B) for (auto& x : v) x = uni(rng);

    /* ---------- build indexes ---------- */
    hnswlib::L2Space space(dim);

    HierarchicalNSW<float> indexA(&space, nA, M, efConstruction, randomSeed);
    HierarchicalNSW<float> indexB(&space, nB, M, efConstruction, randomSeed+1);

    auto tic = Clock::now();
    for (size_t i = 0; i < nA; ++i)
        indexA.addPoint(A[i].data(), labeltype(i));
    auto tA = std::chrono::duration<double>(Clock::now() - tic).count();

    tic = Clock::now();
    for (size_t i = 0; i < nB; ++i)
        indexB.addPoint(B[i].data(), labeltype(i));
    auto tB = std::chrono::duration<double>(Clock::now() - tic).count();

    std::cout << "Built index A in " << tA << " s,  index B in " << tB << " s\n";

    /* ---------- run k-NN join ---------- */
    tic = Clock::now();
    auto joinRes = indexA.knnJoin(indexB, k);   // ef_join=3k, anchor_L=32 defaults
    auto tJoin = std::chrono::duration<double>(Clock::now() - tic).count();
    std::cout << "knnJoin completed in " << tJoin << " s\n";

    /* ---------- brute-force recall check ---------- */
    size_t verified = std::min(nCheck, nA);
    size_t correct  = 0;

    for (size_t i = 0; i < verified; ++i) {
        /* exact top-k from B */
        std::vector<std::pair<double,size_t>> bf;
        bf.reserve(nB);
        for (size_t j = 0; j < nB; ++j)
            bf.emplace_back(l2sq(A[i].data(), B[j].data(), dim), j);

        std::partial_sort(bf.begin(), bf.begin()+k, bf.end());

        const auto& approx = joinRes[i];        // k external labels from indexB
        std::unordered_set<labeltype> approxSet(approx.begin(), approx.end());

        for (size_t r = 0; r < k; ++r)
            if (approxSet.count(bf[r].second)) ++correct;
    }
    double recall = double(correct) / double(verified * k);

    std::cout << "Verified on " << verified << " points  →  recall@"
              << k << " = " << std::fixed << std::setprecision(3)
              << recall << '\n';

    /* ---------- sample output for one query ---------- */
    // size_t qid =94;
    // std::cout << "\nExample neighbourmonimomos for A[" << qid << "] :";
    // for (auto id : joinRes[qid]) std::cout << ' ' << id;
    // auto real_n = indexB.searchKnn(A[qid].data(),10);
    // std::cout << "\n The real one is ";
    // while (!real_n.empty()) {
    //     std::cout << real_n.top().second <<" ";
    //     real_n.pop();
    // }
    // std::cout << '\n';

    // tic = Clock::now();
    // for (size_t i = 0; i < nA; i++)
    // {
    //     indexB.searchKnn(A[i].data(),10);
    // }
        
    // auto tSeprt = std::chrono::duration<double>(Clock::now() - tic).count();
    // std::cout << "Separate indiviual Search completed in " << tJoin << " s\n";

    return 0;
}
