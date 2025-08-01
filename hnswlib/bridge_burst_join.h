#pragma once
/**
 * Bridge & Burst ANN-JOIN for two HierarchicalNSW indices that share
 * the same metric space.
 *
 * 1.  buildBridges()   – one-time pre-processing
 * 2.  joinAtoB()       – returns k approximate NNs in B for every point in A
 *
 * No exhaustive anchor × anchor loops, no density-hard pruning.
 */
#include "hnswalg.h"
#include <unordered_set>
#include <unordered_map>
#include <queue>
#include <chrono>

namespace hnswlib {

template<typename dist_t>
class BridgeBurstJoin {
    using HNSW = HierarchicalNSW<dist_t>;
    using Pair = std::pair<dist_t, labeltype>;

    
public:
    /** --------  phase-0 : create K_bridge outgoing links for every anchor in A  */
    static void buildBridges(HNSW& A,
                             HNSW& B,
                             size_t k_bridge = 4,
                             float  lambda   = 5,
                             size_t kprime   = 20)
    {
        /* gather anchor ids in each index */
        auto Now = std::chrono::high_resolution_clock::now;

        auto t0 =std::chrono::high_resolution_clock::now();

        std::vector<tableint> anchorsA, anchorsB;
        // anchorsA = A.getAnchors();
        // anchorsB = B.getAnchors();
        for (tableint id=0; id<A.cur_element_count; ++id)
        if (!A.anchor_connected_nodes_[id].empty())
        anchorsA.push_back(id);
        for (tableint id=0; id<B.cur_element_count; ++id)
        if (!B.anchor_connected_nodes_[id].empty())
        anchorsB.push_back(id);

        // auto anchorsA = A.get_anchor_list();
        // auto anchorsB = B.get_anchor_list();

        auto t1 =std::chrono::high_resolution_clock::now();
        auto t_approx = std::chrono::duration<double, std::milli>(t1 - t0).count();

        std::cout<<"Getting anchors takes "<<t_approx<<std::endl;          


        /* brute k-NN over anchorsB (small) – keep best k_bridge */
        std::vector<Pair> heap;
        heap.reserve(k_bridge+1);

        t0 = Now();
        for (tableint a : anchorsA) {
            const void* vecA = A.getDataByInternalId(a);
            float Ra  = A.getEdRadius(a,kprime);

            heap.clear();
            for (tableint b : anchorsB) {
                float dist = A.rawDist(vecA, B.getDataByInternalId(b));
                heap.emplace_back(dist, b);
            }
            std::partial_sort(heap.begin(),
                              heap.begin()+std::min(k_bridge,heap.size()),
                              heap.end());
            size_t kept = 0;
            A.bridge_out_[a].clear();
            for (auto& pr : heap) {
                if (kept==k_bridge) break;
                float Ra = A.getEdRadius(a, kprime);
                float dist = A.rawDist(vecA, B.getDataByInternalId(pr.second));
                float Rb = B.getEdRadius(pr.second,kprime);
                if (dist <= lambda * (Ra + Rb)) {      // ←  NEW  (was lambda * Ra)
                    A.bridge_out_[a].push_back(pr.second);
                    ++kept;
                }
            }
        }
        t1 = Now();   
        t_approx = std::chrono::duration<double, std::milli>(t1 - t0).count();

        std::cout<<"General join "<<t_approx<<std::endl;               
    }

    /** --------  phase-1 : JOIN – for every base point in A, return k NNs in B  */
    static std::unordered_map<labeltype, std::vector<Pair>>
    joinAtoB(const HNSW& A,
             const HNSW& B,
             size_t k          = 10,
             size_t ef_base    = 5,
            //  float  tau        = 1.1,
             size_t kprime     = 20)
    {
        using DistPair = std::pair<dist_t /*dist*/, tableint /*id*/>;

        std::unordered_map<labeltype, std::vector<Pair>> result;

        /* convenience lambdas */
        auto toLabel = [&](const HNSW& idx, tableint id) {
            return idx.getExternalLabel(id);
        };

        /* loop over every **live** point in A */
        for (tableint idA=0; idA<A.cur_element_count; ++idA) {
            if (A.isMarkedDeleted(idA)) continue;

            const void* qvec = A.getDataByInternalId(idA);
            tableint    ancA = A.anchor_labels_[idA];
            if (ancA==(tableint)-1 || A.bridge_out_[ancA].empty()) continue;

            /* density of source anchor */
            float LD_src = A.local_density_[ancA];
            if (LD_src==0.0) LD_src = 1e-6f;

            /* collect best candidates from all pivot anchors */
            std::priority_queue<Pair,
                                std::vector<Pair>,
                                std::greater<Pair>> topk;   // min-heap (greater)


            std::unordered_set<tableint> candidate_ids;
            std::priority_queue<DistPair> candidates_q;


            /* --------  burst for every bridge  -------- */
            for (tableint ancB : A.bridge_out_[ancA]) {
                /* bucket & 1-hop halo */
                // for (tableint nid : B.anchor_connected_nodes_[ancB])
                //     candidate_ids.insert(nid);
                    // dist_t curdist =
                    //         hnswlin::fstdistfunc_(A.getDataByInternalId(idA),
                    //                         B.getDataByInternalId(nid),
                    //                         dist_func_param_);
                    // candidates_q.emplace(nid,curdist );

                // auto halo = B.getConnectionsWithLock(ancB,0);  // 1-hop
                // for (auto nid : halo) candidate_ids.insert(nid);

                /* adaptive search budget */
                float LD_dst = B.local_density_[ancB];
                size_t ef    = ef_base * (LD_dst/LD_src);
                ef           = std::max<size_t>(ef_base, ef);

                /* restricted k-NN */
                // auto pq = B.searchKnnRestricted(qvec, k, candidate_ids, ef);
                auto top_candidates = B.searchBaseLayerST(
                            ancB, qvec, std::max(ef, k));  
                /* copy into output map */
                auto labA = A.getExternalLabel(idA);
                auto& vec = result[labA];
                while (!top_candidates.empty()) {
                    vec.emplace_back(top_candidates.top().first, B.getExternalLabel(top_candidates.top().second) );
                    top_candidates.pop();
                }
                candidate_ids.clear();
                // break;
            }
        }
        return result;
    }
};

} // namespace hnswlib
