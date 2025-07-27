#pragma once

#include "visited_list_pool.h"
#include "hnswlib.h"
#include <atomic>
#include <random>
#include <stdlib.h>
#include <assert.h>
#include <unordered_set>
#include <list>
#include <memory>
#include <set>
#include <chrono>

namespace hnswlib {
typedef unsigned int tableint;
typedef unsigned int linklistsizeint;

template<typename dist_t>
class HierarchicalNSW : public AlgorithmInterface<dist_t> {
 public:
    static const tableint MAX_LABEL_OPERATION_LOCKS = 65536;
    static const unsigned char DELETE_MARK = 0x01;

    size_t max_elements_{0};
    mutable std::atomic<size_t> cur_element_count{0};  // current number of elements
    size_t size_data_per_element_{0};
    size_t size_links_per_element_{0};
    mutable std::atomic<size_t> num_deleted_{0};  // number of deleted elements
    size_t M_{0};
    size_t maxM_{0};
    size_t maxM0_{0};
    size_t ef_construction_{0};
    size_t ef_{ 0 };

    double mult_{0.0}, revSize_{0.0};
    int maxlevel_{0};

    // --- anchor/density fields ---
    int   anchor_layer_ = 1;                      // which level to treat as "anchor"
    std::vector<tableint>               anchor_labels_;            // for each new node, which anchor node it pierced
    std::vector<std::vector<tableint>>  anchor_connected_nodes_;   // for each anchor node, list of new nodes
    std::unordered_set<tableint>        anchor_list_;              // a record list of all anchors 
    std::vector<double>                 local_density_;            // LD = k / (farthest‐KNN‐dist)
    std::vector<size_t>                 anchor_last_update_count_; // when we last recomputed density
    size_t                              density_k_     = 10;       // k in LD
    double                              density_thresh = 0.20;     // 20% growth threshold

    // for each anchor a, the set of (distance,node)
    // sorted ascending by distance
    std::vector<std::set<std::pair<dist_t, tableint> > > anchor_children_by_dist_;

    // for each node u, an iterator into whichever anchor_children_by_dist_ it lives in,
    std::vector<typename std::set<std::pair<dist_t,tableint>>::iterator> anchor_child_it_;

    // for each node u, its current distance to anchor_labels_[u]
    std::vector<dist_t> node_anchor_dist_;

    std::unique_ptr<VisitedListPool> visited_list_pool_{nullptr};

    // Locks operations with element by label value
    mutable std::vector<std::mutex> label_op_locks_;

    std::mutex global;
    std::vector<std::mutex> link_list_locks_;

    tableint enterpoint_node_{0};

    size_t size_links_level0_{0};
    size_t offsetData_{0}, offsetLevel0_{0}, label_offset_{ 0 };

    char *data_level0_memory_{nullptr};
    char **linkLists_{nullptr};
    std::vector<int> element_levels_;  // keeps level of each element

    size_t data_size_{0};

    DISTFUNC<dist_t> fstdistfunc_;
    void *dist_func_param_{nullptr};

    mutable std::mutex label_lookup_lock;  // lock for label_lookup_
    std::unordered_map<labeltype, tableint> label_lookup_;

    std::default_random_engine level_generator_;
    std::default_random_engine update_probability_generator_;

    mutable std::atomic<long> metric_distance_computations{0};
    mutable std::atomic<long> metric_hops{0};

    bool allow_replace_deleted_ = false;  // flag to replace deleted elements (marked as deleted) during insertions

    std::mutex deleted_elements_lock;  // lock for deleted_elements
    std::unordered_set<tableint> deleted_elements;  // contains internal ids of deleted elements

    /* ------------------------------------------------------------------ */
    /*  Bridge & Burst join support                                       */
    /* ------------------------------------------------------------------ */
    std::vector<std::vector<tableint>>  bridge_out_;        //  a-anchor → list of b-anchors

    
    HierarchicalNSW(SpaceInterface<dist_t> *s) {
    }


    HierarchicalNSW(
        SpaceInterface<dist_t> *s,
        const std::string &location,
        bool nmslib = false,
        size_t max_elements = 0,
        bool allow_replace_deleted = false)
        : allow_replace_deleted_(allow_replace_deleted) {
        loadIndex(location, s, max_elements);
    }


    HierarchicalNSW(
        SpaceInterface<dist_t> *s,
        size_t max_elements,
        size_t M = 16,
        size_t ef_construction = 200,
        size_t random_seed = 100,
        bool allow_replace_deleted = false)
        : label_op_locks_(MAX_LABEL_OPERATION_LOCKS),
            link_list_locks_(max_elements),
            element_levels_(max_elements),
            allow_replace_deleted_(allow_replace_deleted) {
        max_elements_ = max_elements;
        num_deleted_ = 0;
        data_size_ = s->get_data_size();
        fstdistfunc_ = s->get_dist_func();
        dist_func_param_ = s->get_dist_func_param();
        if ( M <= 10000 ) {
            M_ = M;
        } else {
            HNSWERR << "warning: M parameter exceeds 10000 which may lead to adverse effects." << std::endl;
            HNSWERR << "         Cap to 10000 will be applied for the rest of the processing." << std::endl;
            M_ = 10000;
        }
        maxM_ = M_;
        maxM0_ = M_ * 2;
        ef_construction_ = std::max(ef_construction, M_);
        ef_ = 10;

        level_generator_.seed(random_seed);
        update_probability_generator_.seed(random_seed + 1);

        size_links_level0_ = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);
        size_data_per_element_ = size_links_level0_ + data_size_ + sizeof(labeltype);
        offsetData_ = size_links_level0_;
        label_offset_ = size_links_level0_ + data_size_;
        offsetLevel0_ = 0;

        data_level0_memory_ = (char *) malloc(max_elements_ * size_data_per_element_);
        if (data_level0_memory_ == nullptr)
            throw std::runtime_error("Not enough memory");

        cur_element_count = 0;

        visited_list_pool_ = std::unique_ptr<VisitedListPool>(new VisitedListPool(1, max_elements));

        // initializations for special treatment of the first node
        enterpoint_node_ = -1;
        maxlevel_ = -1;

        linkLists_ = (char **) malloc(sizeof(void *) * max_elements_);
        if (linkLists_ == nullptr)
            throw std::runtime_error("Not enough memory: HierarchicalNSW failed to allocate linklists");
       
        // after linkLists_ = malloc(...)...
        anchor_labels_.assign      (max_elements_, (tableint)-1);
        anchor_connected_nodes_.resize(max_elements_);
        local_density_.assign      (max_elements_, 0.0);
        anchor_last_update_count_.assign(max_elements_, 0);
        anchor_children_by_dist_.resize(max_elements_);
        anchor_child_it_.resize      (max_elements_);
        node_anchor_dist_.assign     (max_elements_, std::numeric_limits<dist_t>::infinity());
        anchor_last_update_count_.assign(max_elements_, 0);

        /* Bridge table (initially empty) */
        bridge_out_.resize(max_elements_);

        size_links_per_element_ = maxM_ * sizeof(tableint) + sizeof(linklistsizeint);
        mult_ = 1 / log(1.0 * M_);
        revSize_ = 1.0 / mult_;
    }


    ~HierarchicalNSW() {
        clear();
    }

    void clear() {
        free(data_level0_memory_);
        data_level0_memory_ = nullptr;
        for (tableint i = 0; i < cur_element_count; i++) {
            if (element_levels_[i] > 0)
                free(linkLists_[i]);
        }
        free(linkLists_);
        linkLists_ = nullptr;
        cur_element_count = 0;
        visited_list_pool_.reset(nullptr);
    }


    struct CompareByFirst {
        constexpr bool operator()(std::pair<dist_t, tableint> const& a,
            std::pair<dist_t, tableint> const& b) const noexcept {
            return a.first < b.first;
        }
    };

    

    void setEf(size_t ef) {
        ef_ = ef;
    }


    inline std::mutex& getLabelOpMutex(labeltype label) const {
        // calculate hash
        size_t lock_id = label & (MAX_LABEL_OPERATION_LOCKS - 1);
        return label_op_locks_[lock_id];
    }


    inline labeltype getExternalLabel(tableint internal_id) const {
        labeltype return_label;
        memcpy(&return_label, (data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_), sizeof(labeltype));
        return return_label;
    }


    inline void setExternalLabel(tableint internal_id, labeltype label) const {
        memcpy((data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_), &label, sizeof(labeltype));
    }


    inline labeltype *getExternalLabeLp(tableint internal_id) const {
        return (labeltype *) (data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_);
    }


    inline char *getDataByInternalId(tableint internal_id) const {
        return (data_level0_memory_ + internal_id * size_data_per_element_ + offsetData_);
    }


    int getRandomLevel(double reverse_size) {
        std::uniform_real_distribution<double> distribution(0.0, 1.0);
        double r = -log(distribution(level_generator_)) * reverse_size;
        return (int) r;
    }

    size_t getMaxElements() {
        return max_elements_;
    }

    size_t getCurrentElementCount() {
        return cur_element_count;
    }

    size_t getDeletedCount() {
        return num_deleted_;
    }

    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
    searchBaseLayer(tableint ep_id, const void *data_point, int layer) {
        VisitedList *vl = visited_list_pool_->getFreeVisitedList();
        vl_type *visited_array = vl->mass;
        vl_type visited_array_tag = vl->curV;

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidateSet;

        dist_t lowerBound;
        if (!isMarkedDeleted(ep_id)) {
            dist_t dist = fstdistfunc_(data_point, getDataByInternalId(ep_id), dist_func_param_);
            top_candidates.emplace(dist, ep_id);
            lowerBound = dist;
            candidateSet.emplace(-dist, ep_id);
        } else {
            lowerBound = std::numeric_limits<dist_t>::max();
            candidateSet.emplace(-lowerBound, ep_id);
        }
        visited_array[ep_id] = visited_array_tag;

        while (!candidateSet.empty()) {
            std::pair<dist_t, tableint> curr_el_pair = candidateSet.top();
            if ((-curr_el_pair.first) > lowerBound && top_candidates.size() == ef_construction_) {
                break;
            }
            candidateSet.pop();

            tableint curNodeNum = curr_el_pair.second;

            std::unique_lock <std::mutex> lock(link_list_locks_[curNodeNum]);

            int *data;  // = (int *)(linkList0_ + curNodeNum * size_links_per_element0_);
            if (layer == 0) {
                data = (int*)get_linklist0(curNodeNum);
            } else {
                data = (int*)get_linklist(curNodeNum, layer);
//                    data = (int *) (linkLists_[curNodeNum] + (layer - 1) * size_links_per_element_);
            }
            size_t size = getListCount((linklistsizeint*)data);
            tableint *datal = (tableint *) (data + 1);
#ifdef USE_SSE
            _mm_prefetch((char *) (visited_array + *(data + 1)), _MM_HINT_T0);
            _mm_prefetch((char *) (visited_array + *(data + 1) + 64), _MM_HINT_T0);
            _mm_prefetch(getDataByInternalId(*datal), _MM_HINT_T0);
            _mm_prefetch(getDataByInternalId(*(datal + 1)), _MM_HINT_T0);
#endif

            for (size_t j = 0; j < size; j++) {
                tableint candidate_id = *(datal + j);
//                    if (candidate_id == 0) continue;
#ifdef USE_SSE
                _mm_prefetch((char *) (visited_array + *(datal + j + 1)), _MM_HINT_T0);
                _mm_prefetch(getDataByInternalId(*(datal + j + 1)), _MM_HINT_T0);
#endif
                if (visited_array[candidate_id] == visited_array_tag) continue;
                visited_array[candidate_id] = visited_array_tag;
                char *currObj1 = (getDataByInternalId(candidate_id));

                dist_t dist1 = fstdistfunc_(data_point, currObj1, dist_func_param_);
                if (top_candidates.size() < ef_construction_ || lowerBound > dist1) {
                    candidateSet.emplace(-dist1, candidate_id);
#ifdef USE_SSE
                    _mm_prefetch(getDataByInternalId(candidateSet.top().second), _MM_HINT_T0);
#endif

                    if (!isMarkedDeleted(candidate_id))
                        top_candidates.emplace(dist1, candidate_id);

                    if (top_candidates.size() > ef_construction_)
                        top_candidates.pop();

                    if (!top_candidates.empty())
                        lowerBound = top_candidates.top().first;
                }
            }
        }
        visited_list_pool_->releaseVisitedList(vl);

        return top_candidates;
    }


    // bare_bone_search means there is no check for deletions and stop condition is ignored in return of extra performance
    template <bool bare_bone_search = true, bool collect_metrics = false>
    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
    searchBaseLayerST(
        tableint ep_id,
        const void *data_point,
        size_t ef,
        BaseFilterFunctor* isIdAllowed = nullptr,
        BaseSearchStopCondition<dist_t>* stop_condition = nullptr) const {
        VisitedList *vl = visited_list_pool_->getFreeVisitedList();
        vl_type *visited_array = vl->mass;
        vl_type visited_array_tag = vl->curV;

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidate_set;

        dist_t lowerBound;
        if (bare_bone_search || 
            (!isMarkedDeleted(ep_id) && ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(ep_id))))) {
            char* ep_data = getDataByInternalId(ep_id);
            dist_t dist = fstdistfunc_(data_point, ep_data, dist_func_param_);
            lowerBound = dist;
            top_candidates.emplace(dist, ep_id);
            if (!bare_bone_search && stop_condition) {
                stop_condition->add_point_to_result(getExternalLabel(ep_id), ep_data, dist);
            }
            candidate_set.emplace(-dist, ep_id);
        } else {
            lowerBound = std::numeric_limits<dist_t>::max();
            candidate_set.emplace(-lowerBound, ep_id);
        }

        visited_array[ep_id] = visited_array_tag;

        while (!candidate_set.empty()) {
            std::pair<dist_t, tableint> current_node_pair = candidate_set.top();
            dist_t candidate_dist = -current_node_pair.first;

            bool flag_stop_search;
            if (bare_bone_search) {
                flag_stop_search = candidate_dist > lowerBound;
            } else {
                if (stop_condition) {
                    flag_stop_search = stop_condition->should_stop_search(candidate_dist, lowerBound);
                } else {
                    flag_stop_search = candidate_dist > lowerBound && top_candidates.size() == ef;
                }
            }
            if (flag_stop_search) {
                break;
            }
            candidate_set.pop();

            tableint current_node_id = current_node_pair.second;
            int *data = (int *) get_linklist0(current_node_id);
            size_t size = getListCount((linklistsizeint*)data);
//                bool cur_node_deleted = isMarkedDeleted(current_node_id);
            if (collect_metrics) {
                metric_hops++;
                metric_distance_computations+=size;
            }

#ifdef USE_SSE
            _mm_prefetch((char *) (visited_array + *(data + 1)), _MM_HINT_T0);
            _mm_prefetch((char *) (visited_array + *(data + 1) + 64), _MM_HINT_T0);
            _mm_prefetch(data_level0_memory_ + (*(data + 1)) * size_data_per_element_ + offsetData_, _MM_HINT_T0);
            _mm_prefetch((char *) (data + 2), _MM_HINT_T0);
#endif

            for (size_t j = 1; j <= size; j++) {
                int candidate_id = *(data + j);
//                    if (candidate_id == 0) continue;
#ifdef USE_SSE
                _mm_prefetch((char *) (visited_array + *(data + j + 1)), _MM_HINT_T0);
                _mm_prefetch(data_level0_memory_ + (*(data + j + 1)) * size_data_per_element_ + offsetData_,
                                _MM_HINT_T0);  ////////////
#endif
                if (!(visited_array[candidate_id] == visited_array_tag)) {
                    visited_array[candidate_id] = visited_array_tag;

                    char *currObj1 = (getDataByInternalId(candidate_id));
                    dist_t dist = fstdistfunc_(data_point, currObj1, dist_func_param_);

                    bool flag_consider_candidate;
                    if (!bare_bone_search && stop_condition) {
                        flag_consider_candidate = stop_condition->should_consider_candidate(dist, lowerBound);
                    } else {
                        flag_consider_candidate = top_candidates.size() < ef || lowerBound > dist;
                    }

                    if (flag_consider_candidate) {
                        candidate_set.emplace(-dist, candidate_id);
#ifdef USE_SSE
                        _mm_prefetch(data_level0_memory_ + candidate_set.top().second * size_data_per_element_ +
                                        offsetLevel0_,  ///////////
                                        _MM_HINT_T0);  ////////////////////////
#endif

                        if (bare_bone_search || 
                            (!isMarkedDeleted(candidate_id) && ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(candidate_id))))) {
                            top_candidates.emplace(dist, candidate_id);
                            if (!bare_bone_search && stop_condition) {
                                stop_condition->add_point_to_result(getExternalLabel(candidate_id), currObj1, dist);
                            }
                        }

                        bool flag_remove_extra = false;
                        if (!bare_bone_search && stop_condition) {
                            flag_remove_extra = stop_condition->should_remove_extra();
                        } else {
                            flag_remove_extra = top_candidates.size() > ef;
                        }
                        while (flag_remove_extra) {
                            tableint id = top_candidates.top().second;
                            top_candidates.pop();
                            if (!bare_bone_search && stop_condition) {
                                stop_condition->remove_point_from_result(getExternalLabel(id), getDataByInternalId(id), dist);
                                flag_remove_extra = stop_condition->should_remove_extra();
                            } else {
                                flag_remove_extra = top_candidates.size() > ef;
                            }
                        }

                        if (!top_candidates.empty())
                            lowerBound = top_candidates.top().first;
                    }
                }
            }
        }

        visited_list_pool_->releaseVisitedList(vl);
        return top_candidates;
    }


    void getNeighborsByHeuristic2(
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &top_candidates,
        const size_t M) {
        if (top_candidates.size() < M) {
            return;
        }

        std::priority_queue<std::pair<dist_t, tableint>> queue_closest;
        std::vector<std::pair<dist_t, tableint>> return_list;
        while (top_candidates.size() > 0) {
            queue_closest.emplace(-top_candidates.top().first, top_candidates.top().second);
            top_candidates.pop();
        }

        while (queue_closest.size()) {
            if (return_list.size() >= M)
                break;
            std::pair<dist_t, tableint> curent_pair = queue_closest.top();
            dist_t dist_to_query = -curent_pair.first;
            queue_closest.pop();
            bool good = true;

            for (std::pair<dist_t, tableint> second_pair : return_list) {
                dist_t curdist =
                        fstdistfunc_(getDataByInternalId(second_pair.second),
                                        getDataByInternalId(curent_pair.second),
                                        dist_func_param_);
                if (curdist < dist_to_query) {
                    good = false;
                    break;
                }
            }
            if (good) {
                return_list.push_back(curent_pair);
            }
        }

        for (std::pair<dist_t, tableint> curent_pair : return_list) {
            top_candidates.emplace(-curent_pair.first, curent_pair.second);
        }
    }


    linklistsizeint *get_linklist0(tableint internal_id) const {
        return (linklistsizeint *) (data_level0_memory_ + internal_id * size_data_per_element_ + offsetLevel0_);
    }


    linklistsizeint *get_linklist0(tableint internal_id, char *data_level0_memory_) const {
        return (linklistsizeint *) (data_level0_memory_ + internal_id * size_data_per_element_ + offsetLevel0_);
    }


    linklistsizeint *get_linklist(tableint internal_id, int level) const {
        return (linklistsizeint *) (linkLists_[internal_id] + (level - 1) * size_links_per_element_);
    }


    linklistsizeint *get_linklist_at_level(tableint internal_id, int level) const {
        return level == 0 ? get_linklist0(internal_id) : get_linklist(internal_id, level);
    }


    tableint mutuallyConnectNewElement(
        const void *data_point,
        tableint cur_c,
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &top_candidates,
        int level,
        bool isUpdate) {
        size_t Mcurmax = level ? maxM_ : maxM0_;
        getNeighborsByHeuristic2(top_candidates, M_);
        if (top_candidates.size() > M_)
            throw std::runtime_error("Should be not be more than M_ candidates returned by the heuristic");

        std::vector<tableint> selectedNeighbors;
        selectedNeighbors.reserve(M_);
        while (top_candidates.size() > 0) {
            selectedNeighbors.push_back(top_candidates.top().second);
            top_candidates.pop();
        }

        tableint next_closest_entry_point = selectedNeighbors.back();

        {
            // lock only during the update
            // because during the addition the lock for cur_c is already acquired
            std::unique_lock <std::mutex> lock(link_list_locks_[cur_c], std::defer_lock);
            if (isUpdate) {
                lock.lock();
            }
            linklistsizeint *ll_cur;
            if (level == 0)
                ll_cur = get_linklist0(cur_c);
            else
                ll_cur = get_linklist(cur_c, level);

            if (*ll_cur && !isUpdate) {
                throw std::runtime_error("The newly inserted element should have blank link list");
            }
            setListCount(ll_cur, selectedNeighbors.size());
            tableint *data = (tableint *) (ll_cur + 1);
            for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
                if (data[idx] && !isUpdate)
                    throw std::runtime_error("Possible memory corruption");
                if (level > element_levels_[selectedNeighbors[idx]])
                    throw std::runtime_error("Trying to make a link on a non-existent level");

                data[idx] = selectedNeighbors[idx];
            }
        }

        for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
            std::unique_lock <std::mutex> lock(link_list_locks_[selectedNeighbors[idx]]);

            linklistsizeint *ll_other;
            if (level == 0)
                ll_other = get_linklist0(selectedNeighbors[idx]);
            else
                ll_other = get_linklist(selectedNeighbors[idx], level);

            size_t sz_link_list_other = getListCount(ll_other);

            if (sz_link_list_other > Mcurmax)
                throw std::runtime_error("Bad value of sz_link_list_other");
            if (selectedNeighbors[idx] == cur_c)
                throw std::runtime_error("Trying to connect an element to itself");
            if (level > element_levels_[selectedNeighbors[idx]])
                throw std::runtime_error("Trying to make a link on a non-existent level");

            tableint *data = (tableint *) (ll_other + 1);

            bool is_cur_c_present = false;
            if (isUpdate) {
                for (size_t j = 0; j < sz_link_list_other; j++) {
                    if (data[j] == cur_c) {
                        is_cur_c_present = true;
                        break;
                    }
                }
            }

            // If cur_c is already present in the neighboring connections of `selectedNeighbors[idx]` then no need to modify any connections or run the heuristics.
            if (!is_cur_c_present) {
                if (sz_link_list_other < Mcurmax) {
                    data[sz_link_list_other] = cur_c;
                    setListCount(ll_other, sz_link_list_other + 1);
                } else {
                    // finding the "weakest" element to replace it with the new one
                    dist_t d_max = fstdistfunc_(getDataByInternalId(cur_c), getDataByInternalId(selectedNeighbors[idx]),
                                                dist_func_param_);
                    // Heuristic:
                    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidates;
                    candidates.emplace(d_max, cur_c);

                    for (size_t j = 0; j < sz_link_list_other; j++) {
                        candidates.emplace(
                                fstdistfunc_(getDataByInternalId(data[j]), getDataByInternalId(selectedNeighbors[idx]),
                                                dist_func_param_), data[j]);
                    }

                    getNeighborsByHeuristic2(candidates, Mcurmax);

                    int indx = 0;
                    while (candidates.size() > 0) {
                        data[indx] = candidates.top().second;
                        candidates.pop();
                        indx++;
                    }

                    setListCount(ll_other, indx);
                    // Nearest K:
                    /*int indx = -1;
                    for (int j = 0; j < sz_link_list_other; j++) {
                        dist_t d = fstdistfunc_(getDataByInternalId(data[j]), getDataByInternalId(rez[idx]), dist_func_param_);
                        if (d > d_max) {
                            indx = j;
                            d_max = d;
                        }
                    }
                    if (indx >= 0) {
                        data[indx] = cur_c;
                    } */
                }
            }
        }

        return next_closest_entry_point;
    }


    void resizeIndex(size_t new_max_elements) {
        if (new_max_elements < cur_element_count)
            throw std::runtime_error("Cannot resize, max element is less than the current number of elements");

        visited_list_pool_.reset(new VisitedListPool(1, new_max_elements));

        element_levels_.resize(new_max_elements);

        std::vector<std::mutex>(new_max_elements).swap(link_list_locks_);

        // Reallocate base layer
        char * data_level0_memory_new = (char *) realloc(data_level0_memory_, new_max_elements * size_data_per_element_);
        if (data_level0_memory_new == nullptr)
            throw std::runtime_error("Not enough memory: resizeIndex failed to allocate base layer");
        data_level0_memory_ = data_level0_memory_new;

        // Reallocate all other layers
        char ** linkLists_new = (char **) realloc(linkLists_, sizeof(void *) * new_max_elements);
        if (linkLists_new == nullptr)
            throw std::runtime_error("Not enough memory: resizeIndex failed to allocate other layers");
        linkLists_ = linkLists_new;

        max_elements_ = new_max_elements;
    }

    size_t indexFileSize() const {
        size_t size = 0;
        size += sizeof(offsetLevel0_);
        size += sizeof(max_elements_);
        size += sizeof(cur_element_count);
        size += sizeof(size_data_per_element_);
        size += sizeof(label_offset_);
        size += sizeof(offsetData_);
        size += sizeof(maxlevel_);
        size += sizeof(enterpoint_node_);
        size += sizeof(maxM_);

        size += sizeof(maxM0_);
        size += sizeof(M_);
        size += sizeof(mult_);
        size += sizeof(ef_construction_);

        size += cur_element_count * size_data_per_element_;

        for (size_t i = 0; i < cur_element_count; i++) {
            unsigned int linkListSize = element_levels_[i] > 0 ? size_links_per_element_ * element_levels_[i] : 0;
            size += sizeof(linkListSize);
            size += linkListSize;
        }
        return size;
    }

    void saveIndex(const std::string &location) {
        std::ofstream output(location, std::ios::binary);
        std::streampos position;

        writeBinaryPOD(output, offsetLevel0_);
        writeBinaryPOD(output, max_elements_);
        writeBinaryPOD(output, cur_element_count);
        writeBinaryPOD(output, size_data_per_element_);
        writeBinaryPOD(output, label_offset_);
        writeBinaryPOD(output, offsetData_);
        writeBinaryPOD(output, maxlevel_);
        writeBinaryPOD(output, enterpoint_node_);
        writeBinaryPOD(output, maxM_);

        writeBinaryPOD(output, maxM0_);
        writeBinaryPOD(output, M_);
        writeBinaryPOD(output, mult_);
        writeBinaryPOD(output, ef_construction_);

        output.write(data_level0_memory_, cur_element_count * size_data_per_element_);

        for (size_t i = 0; i < cur_element_count; i++) {
            unsigned int linkListSize = element_levels_[i] > 0 ? size_links_per_element_ * element_levels_[i] : 0;
            writeBinaryPOD(output, linkListSize);
            if (linkListSize)
                output.write(linkLists_[i], linkListSize);
        }

        // 1) anchor_layer_
        writeBinaryPOD(output, anchor_layer_);
        
        // 2) anchor_labels_
        writeBinaryPOD(output, anchor_labels_.size());
        for (auto &lbl : anchor_labels_) writeBinaryPOD(output, lbl);
        
        // 3) anchor_connected_nodes_
        writeBinaryPOD(output, anchor_connected_nodes_.size());
        for (auto &vec : anchor_connected_nodes_) {
            writeBinaryPOD(output, vec.size());
            for (auto &n : vec) writeBinaryPOD(output, n);
        }
        
        // 4) anchor_list_
        writeBinaryPOD(output, anchor_list_.size());
        for (auto &a : anchor_list_) writeBinaryPOD(output, a);
        
        // 5) local_density_
        writeBinaryPOD(output, local_density_.size());
        for (auto &d : local_density_) writeBinaryPOD(output, d);
        
        // 6) anchor_last_update_count_
        writeBinaryPOD(output, anchor_last_update_count_.size());
        for (auto &c : anchor_last_update_count_) writeBinaryPOD(output, c);
        
        // 7) density_k_ & density_thresh
        writeBinaryPOD(output, density_k_);
        writeBinaryPOD(output, density_thresh);
        
        // 8) anchor_children_by_dist_
        writeBinaryPOD(output, anchor_children_by_dist_.size());
        for (auto &childSet : anchor_children_by_dist_) {
            writeBinaryPOD(output, childSet.size());
            for (auto &pr : childSet) {
                writeBinaryPOD(output, pr.first);   // distance
                writeBinaryPOD(output, pr.second);  // node id
            }
        }
        
        // 9) node_anchor_dist_
        writeBinaryPOD(output, node_anchor_dist_.size());
        for (auto &d : node_anchor_dist_) writeBinaryPOD(output, d);
        
        // 10) bridge_out_
        writeBinaryPOD(output, bridge_out_.size());
        for (auto &vec : bridge_out_) {
            writeBinaryPOD(output, vec.size());
            for (auto &n : vec) writeBinaryPOD(output, n);
        }
        
        // 11) pr_dist_
        writeBinaryPOD(output, pr_dist_.size());
        for (auto &p : pr_dist_) writeBinaryPOD(output, p);
        
        // 12) density_ratio_threshold_
        writeBinaryPOD(output, density_ratio_threshold_);
        // … end of new fields …
                
        output.close();
    }


    void loadIndex(const std::string &location, SpaceInterface<dist_t> *s, size_t max_elements_i = 0) {
        std::ifstream input(location, std::ios::binary);

        if (!input.is_open())
            throw std::runtime_error("Cannot open file");

        clear();
        // get file size:
        input.seekg(0, input.end);
        std::streampos total_filesize = input.tellg();
        input.seekg(0, input.beg);

        readBinaryPOD(input, offsetLevel0_);
        readBinaryPOD(input, max_elements_);
        readBinaryPOD(input, cur_element_count);

        size_t max_elements = max_elements_i;
        if (max_elements < cur_element_count)
            max_elements = max_elements_;
        max_elements_ = max_elements;
        readBinaryPOD(input, size_data_per_element_);
        readBinaryPOD(input, label_offset_);
        readBinaryPOD(input, offsetData_);
        readBinaryPOD(input, maxlevel_);
        readBinaryPOD(input, enterpoint_node_);

        readBinaryPOD(input, maxM_);
        readBinaryPOD(input, maxM0_);
        readBinaryPOD(input, M_);
        readBinaryPOD(input, mult_);
        readBinaryPOD(input, ef_construction_);

        data_size_ = s->get_data_size();
        fstdistfunc_ = s->get_dist_func();
        dist_func_param_ = s->get_dist_func_param();

        auto pos = input.tellg();

        // /// Optional - check if index is ok:
        // input.seekg(cur_element_count * size_data_per_element_, input.cur);
        // for (size_t i = 0; i < cur_element_count; i++) {
        //     if (input.tellg() < 0 || input.tellg() >= total_filesize) {
        //         throw std::runtime_error("Index seems to be corrupted or unsupported");
        //     }

        //     unsigned int linkListSize;
        //     readBinaryPOD(input, linkListSize);
        //     if (linkListSize != 0) {
        //         input.seekg(linkListSize, input.cur);
        //     }
        // }

        // // throw exception if it either corrupted or old index
        // if (input.tellg() != total_filesize)
        //     throw std::runtime_error("Index seems to be corrupted or unsupported");

        // input.clear();
        // /// Optional check end

        input.seekg(pos, input.beg);

        data_level0_memory_ = (char *) malloc(max_elements * size_data_per_element_);
        if (data_level0_memory_ == nullptr)
            throw std::runtime_error("Not enough memory: loadIndex failed to allocate level0");
        input.read(data_level0_memory_, cur_element_count * size_data_per_element_);

        size_links_per_element_ = maxM_ * sizeof(tableint) + sizeof(linklistsizeint);

        size_links_level0_ = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);
        std::vector<std::mutex>(max_elements).swap(link_list_locks_);
        std::vector<std::mutex>(MAX_LABEL_OPERATION_LOCKS).swap(label_op_locks_);

        visited_list_pool_.reset(new VisitedListPool(1, max_elements));

        linkLists_ = (char **) malloc(sizeof(void *) * max_elements);
        if (linkLists_ == nullptr)
            throw std::runtime_error("Not enough memory: loadIndex failed to allocate linklists");
        element_levels_ = std::vector<int>(max_elements);
        revSize_ = 1.0 / mult_;
        ef_ = 10;
        for (size_t i = 0; i < cur_element_count; i++) {
            label_lookup_[getExternalLabel(i)] = i;
            unsigned int linkListSize;
            readBinaryPOD(input, linkListSize);
            if (linkListSize == 0) {
                element_levels_[i] = 0;
                linkLists_[i] = nullptr;
            } else {
                element_levels_[i] = linkListSize / size_links_per_element_;
                linkLists_[i] = (char *) malloc(linkListSize);
                if (linkLists_[i] == nullptr)
                    throw std::runtime_error("Not enough memory: loadIndex failed to allocate linklist");
                input.read(linkLists_[i], linkListSize);
            }
        }

        for (size_t i = 0; i < cur_element_count; i++) {
            if (isMarkedDeleted(i)) {
                num_deleted_ += 1;
                if (allow_replace_deleted_) deleted_elements.insert(i);
            }
        }


        // 1) anchor_layer_
        readBinaryPOD(input, anchor_layer_);
        
        // 2) anchor_labels_
        {
            size_t n; readBinaryPOD(input, n);
            anchor_labels_.resize(n);
            for (size_t i = 0; i < n; ++i) readBinaryPOD(input, anchor_labels_[i]);
        }
        
        // 3) anchor_connected_nodes_
        {
            size_t na; readBinaryPOD(input, na);
            anchor_connected_nodes_.resize(na);
            for (size_t i = 0; i < na; ++i) {
                size_t ni; readBinaryPOD(input, ni);
                anchor_connected_nodes_[i].resize(ni);
                for (size_t j = 0; j < ni; ++j)
                    readBinaryPOD(input, anchor_connected_nodes_[i][j]);
            }
        }
        
        // 4) anchor_list_
        {
            size_t ns; readBinaryPOD(input, ns);
            anchor_list_.clear();
            for (size_t i = 0; i < ns; ++i) {
                tableint v; readBinaryPOD(input, v);
                anchor_list_.insert(v);
            }
        }
        
        // 5) local_density_
        {
            size_t nl; readBinaryPOD(input, nl);
            local_density_.resize(nl);
            for (size_t i = 0; i < nl; ++i) readBinaryPOD(input, local_density_[i]);
        }
        
        // 6) anchor_last_update_count_
        {
            size_t nc; readBinaryPOD(input, nc);
            anchor_last_update_count_.resize(nc);
            for (size_t i = 0; i < nc; ++i)
                readBinaryPOD(input, anchor_last_update_count_[i]);
        }
        
        // 7) density_k_, density_thresh
        readBinaryPOD(input, density_k_);
        readBinaryPOD(input, density_thresh);
        
        // 8) anchor_children_by_dist_
        {
            size_t na; readBinaryPOD(input, na);
            anchor_children_by_dist_.clear();
            anchor_children_by_dist_.resize(na);
            for (size_t i = 0; i < na; ++i) {
                size_t sz; readBinaryPOD(input, sz);
                for (size_t j = 0; j < sz; ++j) {
                    dist_t d; tableint u;
                    readBinaryPOD(input, d);
                    readBinaryPOD(input, u);
                    anchor_children_by_dist_[i].insert({d,u});
                }
            }
        }
        
        // 9) node_anchor_dist_
        {
            size_t nn; readBinaryPOD(input, nn);
            node_anchor_dist_.resize(nn);
            for (size_t i = 0; i < nn; ++i)
                readBinaryPOD(input, node_anchor_dist_[i]);
        }
        
        // rebuild anchor_child_it_
        anchor_child_it_.clear();
        anchor_child_it_.resize(cur_element_count);
        for (size_t a = 0; a < anchor_children_by_dist_.size(); ++a) {
            for (auto it = anchor_children_by_dist_[a].begin();
                it != anchor_children_by_dist_[a].end(); ++it)
            {
                anchor_child_it_[ it->second ] = it;
            }
        }
        
        // 10) bridge_out_
        {
            size_t nb; readBinaryPOD(input, nb);
            bridge_out_.resize(nb);
            for (size_t i = 0; i < nb; ++i) {
                size_t ni; readBinaryPOD(input, ni);
                bridge_out_[i].resize(ni);
                for (size_t j = 0; j < ni; ++j)
                    readBinaryPOD(input, bridge_out_[i][j]);
            }
        }
        
        // 11) pr_dist_
        {
            size_t np; readBinaryPOD(input, np);
            pr_dist_.resize(np);
            for (size_t i = 0; i < np; ++i) readBinaryPOD(input, pr_dist_[i]);
        }
        
        // 12) density_ratio_threshold_
        readBinaryPOD(input, density_ratio_threshold_);

        input.close();

        return;
    }


    template<typename data_t>
    std::vector<data_t> getDataByLabel(labeltype label) const {
        // lock all operations with element by label
        std::unique_lock <std::mutex> lock_label(getLabelOpMutex(label));
        
        std::unique_lock <std::mutex> lock_table(label_lookup_lock);
        auto search = label_lookup_.find(label);
        if (search == label_lookup_.end() || isMarkedDeleted(search->second)) {
            throw std::runtime_error("Label not found");
        }
        tableint internalId = search->second;
        lock_table.unlock();

        char* data_ptrv = getDataByInternalId(internalId);
        size_t dim = *((size_t *) dist_func_param_);
        std::vector<data_t> data;
        data_t* data_ptr = (data_t*) data_ptrv;
        for (size_t i = 0; i < dim; i++) {
            data.push_back(*data_ptr);
            data_ptr += 1;
        }
        return data;
    }


    /*
    * Marks an element with the given label deleted, does NOT really change the current graph.
    */
    void markDelete(labeltype label) {
        // lock all operations with element by label
        std::unique_lock <std::mutex> lock_label(getLabelOpMutex(label));

        std::unique_lock <std::mutex> lock_table(label_lookup_lock);
        auto search = label_lookup_.find(label);
        if (search == label_lookup_.end()) {
            throw std::runtime_error("Label not found");
        }
        tableint internalId = search->second;
        lock_table.unlock();

        markDeletedInternal(internalId);
    }


    /*
    * Uses the last 16 bits of the memory for the linked list size to store the mark,
    * whereas maxM0_ has to be limited to the lower 16 bits, however, still large enough in almost all cases.
    */
    void markDeletedInternal(tableint internalId) {
        assert(internalId < cur_element_count);
        if (!isMarkedDeleted(internalId)) {
            unsigned char *ll_cur = ((unsigned char *)get_linklist0(internalId))+2;
            *ll_cur |= DELETE_MARK;
            num_deleted_ += 1;
            if (allow_replace_deleted_) {
                std::unique_lock <std::mutex> lock_deleted_elements(deleted_elements_lock);
                deleted_elements.insert(internalId);
            }
        } else {
            throw std::runtime_error("The requested to delete element is already deleted");
        }
    }


    /*
    * Removes the deleted mark of the node, does NOT really change the current graph.
    * 
    * Note: the method is not safe to use when replacement of deleted elements is enabled,
    *  because elements marked as deleted can be completely removed by addPoint
    */
    void unmarkDelete(labeltype label) {
        // lock all operations with element by label
        std::unique_lock <std::mutex> lock_label(getLabelOpMutex(label));

        std::unique_lock <std::mutex> lock_table(label_lookup_lock);
        auto search = label_lookup_.find(label);
        if (search == label_lookup_.end()) {
            throw std::runtime_error("Label not found");
        }
        tableint internalId = search->second;
        lock_table.unlock();

        unmarkDeletedInternal(internalId);
    }



    /*
    * Remove the deleted mark of the node.
    */
    void unmarkDeletedInternal(tableint internalId) {
        assert(internalId < cur_element_count);
        if (isMarkedDeleted(internalId)) {
            unsigned char *ll_cur = ((unsigned char *)get_linklist0(internalId)) + 2;
            *ll_cur &= ~DELETE_MARK;
            num_deleted_ -= 1;
            if (allow_replace_deleted_) {
                std::unique_lock <std::mutex> lock_deleted_elements(deleted_elements_lock);
                deleted_elements.erase(internalId);
            }
        } else {
            throw std::runtime_error("The requested to undelete element is not deleted");
        }
    }


    /*
    * Checks the first 16 bits of the memory to see if the element is marked deleted.
    */
    bool isMarkedDeleted(tableint internalId) const {
        unsigned char *ll_cur = ((unsigned char*)get_linklist0(internalId)) + 2;
        return *ll_cur & DELETE_MARK;
    }


    unsigned short int getListCount(linklistsizeint * ptr) const {
        return *((unsigned short int *)ptr);
    }


    void setListCount(linklistsizeint * ptr, unsigned short int size) const {
        *((unsigned short int*)(ptr))=*((unsigned short int *)&size);
    }


    /*
    * Adds point. Updates the point if it is already in the index.
    * If replacement of deleted elements is enabled: replaces previously deleted point if any, updating it with new point
    */
    void addPoint(const void *data_point, labeltype label, bool replace_deleted = false) {
        if ((allow_replace_deleted_ == false) && (replace_deleted == true)) {
            throw std::runtime_error("Replacement of deleted elements is disabled in constructor");
        }

        // lock all operations with element by label
        std::unique_lock <std::mutex> lock_label(getLabelOpMutex(label));
        if (!replace_deleted) {
            addPoint(data_point, label, -1);
            return;
        }
        // check if there is vacant place
        tableint internal_id_replaced;
        std::unique_lock <std::mutex> lock_deleted_elements(deleted_elements_lock);
        bool is_vacant_place = !deleted_elements.empty();
        if (is_vacant_place) {
            internal_id_replaced = *deleted_elements.begin();
            deleted_elements.erase(internal_id_replaced);
        }
        lock_deleted_elements.unlock();

        // if there is no vacant place then add or update point
        // else add point to vacant place
        if (!is_vacant_place) {
            addPoint(data_point, label, -1);
        } else {
            // we assume that there are no concurrent operations on deleted element
            labeltype label_replaced = getExternalLabel(internal_id_replaced);
            setExternalLabel(internal_id_replaced, label);

            std::unique_lock <std::mutex> lock_table(label_lookup_lock);
            label_lookup_.erase(label_replaced);
            label_lookup_[label] = internal_id_replaced;
            lock_table.unlock();

            unmarkDeletedInternal(internal_id_replaced);
            updatePoint(data_point, internal_id_replaced, 1.0);
        }
    }


    void updatePoint(const void *dataPoint, tableint internalId, float updateNeighborProbability) {
        // update the feature vector associated with existing point with new vector
        memcpy(getDataByInternalId(internalId), dataPoint, data_size_);

        int maxLevelCopy = maxlevel_;
        tableint entryPointCopy = enterpoint_node_;
        // If point to be updated is entry point and graph just contains single element then just return.
        if (entryPointCopy == internalId && cur_element_count == 1)
            return;

        int elemLevel = element_levels_[internalId];
        std::uniform_real_distribution<float> distribution(0.0, 1.0);
        for (int layer = 0; layer <= elemLevel; layer++) {
            std::unordered_set<tableint> sCand;
            std::unordered_set<tableint> sNeigh;
            std::vector<tableint> listOneHop = getConnectionsWithLock(internalId, layer);
            if (listOneHop.size() == 0)
                continue;

            sCand.insert(internalId);

            for (auto&& elOneHop : listOneHop) {
                sCand.insert(elOneHop);

                if (distribution(update_probability_generator_) > updateNeighborProbability)
                    continue;

                sNeigh.insert(elOneHop);

                std::vector<tableint> listTwoHop = getConnectionsWithLock(elOneHop, layer);
                for (auto&& elTwoHop : listTwoHop) {
                    sCand.insert(elTwoHop);
                }
            }

            for (auto&& neigh : sNeigh) {
                // if (neigh == internalId)
                //     continue;

                std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidates;
                size_t size = sCand.find(neigh) == sCand.end() ? sCand.size() : sCand.size() - 1;  // sCand guaranteed to have size >= 1
                size_t elementsToKeep = std::min(ef_construction_, size);
                for (auto&& cand : sCand) {
                    if (cand == neigh)
                        continue;

                    dist_t distance = fstdistfunc_(getDataByInternalId(neigh), getDataByInternalId(cand), dist_func_param_);
                    if (candidates.size() < elementsToKeep) {
                        candidates.emplace(distance, cand);
                    } else {
                        if (distance < candidates.top().first) {
                            candidates.pop();
                            candidates.emplace(distance, cand);
                        }
                    }
                }

                // Retrieve neighbours using heuristic and set connections.
                getNeighborsByHeuristic2(candidates, layer == 0 ? maxM0_ : maxM_);

                {
                    std::unique_lock <std::mutex> lock(link_list_locks_[neigh]);
                    linklistsizeint *ll_cur;
                    ll_cur = get_linklist_at_level(neigh, layer);
                    size_t candSize = candidates.size();
                    setListCount(ll_cur, candSize);
                    tableint *data = (tableint *) (ll_cur + 1);
                    for (size_t idx = 0; idx < candSize; idx++) {
                        data[idx] = candidates.top().second;
                        candidates.pop();
                    }
                }
            }
        }

        repairConnectionsForUpdate(dataPoint, entryPointCopy, internalId, elemLevel, maxLevelCopy);
    }


    void repairConnectionsForUpdate(
        const void *dataPoint,
        tableint entryPointInternalId,
        tableint dataPointInternalId,
        int dataPointLevel,
        int maxLevel) {
        tableint currObj = entryPointInternalId;
        if (dataPointLevel < maxLevel) {
            dist_t curdist = fstdistfunc_(dataPoint, getDataByInternalId(currObj), dist_func_param_);
            for (int level = maxLevel; level > dataPointLevel; level--) {
                bool changed = true;
                while (changed) {
                    changed = false;
                    unsigned int *data;
                    std::unique_lock <std::mutex> lock(link_list_locks_[currObj]);
                    data = get_linklist_at_level(currObj, level);
                    int size = getListCount(data);
                    tableint *datal = (tableint *) (data + 1);
#ifdef USE_SSE
                    _mm_prefetch(getDataByInternalId(*datal), _MM_HINT_T0);
#endif
                    for (int i = 0; i < size; i++) {
#ifdef USE_SSE
                        _mm_prefetch(getDataByInternalId(*(datal + i + 1)), _MM_HINT_T0);
#endif
                        tableint cand = datal[i];
                        dist_t d = fstdistfunc_(dataPoint, getDataByInternalId(cand), dist_func_param_);
                        if (d < curdist) {
                            curdist = d;
                            currObj = cand;
                            changed = true;
                        }
                    }
                }
            }
        }

        if (dataPointLevel > maxLevel)
            throw std::runtime_error("Level of item to be updated cannot be bigger than max level");

        for (int level = dataPointLevel; level >= 0; level--) {
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> topCandidates = searchBaseLayer(
                    currObj, dataPoint, level);

            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> filteredTopCandidates;
            while (topCandidates.size() > 0) {
                if (topCandidates.top().second != dataPointInternalId)
                    filteredTopCandidates.push(topCandidates.top());

                topCandidates.pop();
            }

            // Since element_levels_ is being used to get `dataPointLevel`, there could be cases where `topCandidates` could just contains entry point itself.
            // To prevent self loops, the `topCandidates` is filtered and thus can be empty.
            if (filteredTopCandidates.size() > 0) {
                bool epDeleted = isMarkedDeleted(entryPointInternalId);
                if (epDeleted) {
                    filteredTopCandidates.emplace(fstdistfunc_(dataPoint, getDataByInternalId(entryPointInternalId), dist_func_param_), entryPointInternalId);
                    if (filteredTopCandidates.size() > ef_construction_)
                        filteredTopCandidates.pop();
                }

                currObj = mutuallyConnectNewElement(dataPoint, dataPointInternalId, filteredTopCandidates, level, true);
            }
        }
    }


    std::vector<tableint> getConnectionsWithLock(tableint internalId, int level) const  {
        // std::unique_lock <std::mutex> lock(link_list_locks_[internalId]);
        unsigned int *data = get_linklist_at_level(internalId, level);
        int size = getListCount(data);
        std::vector<tableint> result(size);
        tableint *ll = (tableint *) (data + 1);
        memcpy(result.data(), ll, size * sizeof(tableint));
        return result;
    }


    tableint addPoint(const void *data_point, labeltype label, int level) {
        tableint cur_c = 0;
        {
            // Checking if the element with the same label already exists
            // if so, updating it *instead* of creating a new element.
            std::unique_lock <std::mutex> lock_table(label_lookup_lock);
            auto search = label_lookup_.find(label);
            if (search != label_lookup_.end()) {
                tableint existingInternalId = search->second;
                if (allow_replace_deleted_) {
                    if (isMarkedDeleted(existingInternalId)) {
                        throw std::runtime_error("Can't use addPoint to update deleted elements if replacement of deleted elements is enabled.");
                    }
                }
                lock_table.unlock();

                if (isMarkedDeleted(existingInternalId)) {
                    unmarkDeletedInternal(existingInternalId);
                }
                updatePoint(data_point, existingInternalId, 1.0);

                return existingInternalId;
            }

            if (cur_element_count >= max_elements_) {
                throw std::runtime_error("The number of elements exceeds the specified limit");
            }

            cur_c = cur_element_count;
            cur_element_count++;
            label_lookup_[label] = cur_c;
        }

        std::unique_lock <std::mutex> lock_el(link_list_locks_[cur_c]);
        int curlevel = getRandomLevel(mult_);
        if (cur_c == 0)
        {
            curlevel = 1;
        }
        
        if (level > 0)
            curlevel = level;

        element_levels_[cur_c] = curlevel;

        std::unique_lock <std::mutex> templock(global);
        int maxlevelcopy = maxlevel_;
        if (curlevel <= maxlevelcopy)
            templock.unlock();
        tableint currObj = enterpoint_node_;
        tableint enterpoint_copy = enterpoint_node_;

        memset(data_level0_memory_ + cur_c * size_data_per_element_ + offsetLevel0_, 0, size_data_per_element_);

        // Initialisation of the data and label
        memcpy(getExternalLabeLp(cur_c), &label, sizeof(labeltype));
        memcpy(getDataByInternalId(cur_c), data_point, data_size_);

        if (curlevel) {
            linkLists_[cur_c] = (char *) malloc(size_links_per_element_ * curlevel + 1);
            if (linkLists_[cur_c] == nullptr)
                throw std::runtime_error("Not enough memory: addPoint failed to allocate linklist");
            memset(linkLists_[cur_c], 0, size_links_per_element_ * curlevel + 1);
        }

        if ((signed)currObj != -1) {
            if (curlevel < maxlevelcopy) {
                dist_t curdist = fstdistfunc_(data_point, getDataByInternalId(currObj), dist_func_param_);
                for (int level = maxlevelcopy; level > curlevel; level--) {
                    bool changed = true;
                    while (changed) {
                        changed = false;
                        unsigned int *data;
                        std::unique_lock <std::mutex> lock(link_list_locks_[currObj]);
                        data = get_linklist(currObj, level);
                        int size = getListCount(data);

                        tableint *datal = (tableint *) (data + 1);
                        for (int i = 0; i < size; i++) {
                            tableint cand = datal[i];
                            if (cand < 0 || cand > max_elements_)
                                throw std::runtime_error("cand error");
                            dist_t d = fstdistfunc_(data_point, getDataByInternalId(cand), dist_func_param_);
                            if (d < curdist) {
                                curdist = d;
                                currObj = cand;
                                changed = true;
                            }
                        }
                    }
                    if (curlevel >= anchor_layer_ and level == anchor_layer_)
                    {
                        anchor_labels_[cur_c] = cur_c;
                        auto &vec = anchor_connected_nodes_[cur_c];
                        vec.push_back(cur_c);

                        // 1) compute and record its dist to its anchor
                        dist_t d = 0;
                        node_anchor_dist_[cur_c] = d;

                        // 2) insert into that anchor's sorted set
                        auto &S = anchor_children_by_dist_[cur_c];
                        // auto [it, ok] = S.emplace(d, cur_c);
                        auto insert_res = S.emplace(d, cur_c);
                        auto it         = insert_res.first;
                        // bool ok         = insert_res.second; 
                        anchor_child_it_[cur_c] = it;

                        anchor_list_.insert(cur_c);


                        continue;
                        
                    }
                    
                    if (level == anchor_layer_) {
                        // remember which node we pierced
                        anchor_labels_[cur_c] = currObj;
                        // attach this new node under that anchor
                        auto &vec = anchor_connected_nodes_[currObj];
                        vec.push_back(cur_c);

                        // 1) compute and record its dist to its anchor
                        dist_t d = fstdistfunc_(getDataByInternalId(cur_c),
                                                getDataByInternalId(currObj),
                                                dist_func_param_);
                        node_anchor_dist_[cur_c] = d;

                        // 2) insert into that anchor's sorted set
                        auto &S = anchor_children_by_dist_[currObj];
                        // auto [it, ok] = S.emplace(d, cur_c);
                        auto insert_res = S.emplace(d, cur_c);
                        auto it         = insert_res.first;
                        bool ok         = insert_res.second;                         
                        anchor_child_it_[cur_c] = it;


                    }                
                }
            }

            bool epDeleted = isMarkedDeleted(enterpoint_copy);
            for (int level = std::min(curlevel, maxlevelcopy); level >= 0; level--) {
                if (level > maxlevelcopy || level < 0)  // possible?
                    throw std::runtime_error("Level error");
                // --- anchor‐layer hook: record and maybe update density ---
                if (curlevel >= anchor_layer_ and level == anchor_layer_)
                {
                    anchor_labels_[cur_c] = cur_c;
                    auto &vec = anchor_connected_nodes_[cur_c];
                    vec.push_back(cur_c);
                    
                    // 1) compute and record its dist to its anchor
                    dist_t d = 0;
                    node_anchor_dist_[cur_c] = d;
                    
                    // 2) insert into that anchor's sorted set
                    auto &S = anchor_children_by_dist_[cur_c];
                    // auto [it, ok] = S.emplace(d, cur_c);
                    auto insert_res = S.emplace(d, cur_c);
                    auto it         = insert_res.first;
                    bool ok         = insert_res.second;                     
                    anchor_child_it_[cur_c] = it;
                    
                    anchor_list_.insert(cur_c);
                    
                }                
                if (curlevel < anchor_layer_ and level == anchor_layer_) {
                    // remember which node we pierced
                    anchor_labels_[cur_c] = currObj;
                    auto &vec = anchor_connected_nodes_[currObj];
                    vec.push_back(cur_c);
                    
                    // 1) compute and record its dist to its anchor
                    dist_t d = fstdistfunc_(getDataByInternalId(cur_c),
                                            getDataByInternalId(currObj),
                                            dist_func_param_);
                    node_anchor_dist_[cur_c] = d;

                    // 2) insert into that anchor's sorted set
                    auto &S = anchor_children_by_dist_[currObj];
                    auto insert_res = S.emplace(d, cur_c);
                    auto it         = insert_res.first;
                    bool ok         = insert_res.second; 
                    anchor_child_it_[cur_c] = it;

                }
                    
                std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates = searchBaseLayer(
                        currObj, data_point, level);
                if (epDeleted) {
                    top_candidates.emplace(fstdistfunc_(data_point, getDataByInternalId(enterpoint_copy), dist_func_param_), enterpoint_copy);
                    if (top_candidates.size() > ef_construction_)
                        top_candidates.pop();
                }
                auto temp_can = top_candidates;
                currObj = mutuallyConnectNewElement(data_point, cur_c, top_candidates, level, false);
                if (level == 0)
                {                    
                    auto & anchor_label = anchor_labels_[cur_c];
                    auto anchor_connected_nodes = anchor_connected_nodes_[anchor_label];
                    size_t newCount = anchor_connected_nodes.size();
                    if (newCount >= anchor_last_update_count_[anchor_label] * (1.0 + density_thresh)) {
                        auto anchor_data = getDataByInternalId(anchor_label);
                        while (temp_can.size()>10) temp_can.pop();
                        double farthest =temp_can.top().first;
                        // std::cout<<"update once ";
                        local_density_[anchor_label] = double(density_k_) / farthest;
                        anchor_last_update_count_[anchor_label] = newCount;
                    }     

                    while (temp_can.size()>0)
                    {
                        // Let u = selectedNeighbors[idx], v = cur_c
                        tableint u = temp_can.top().second;
                        temp_can.pop();
                        tableint v = cur_c;                        
                        // Try to re-anchor u under v’s anchor if closer:
                        tryReanchor(u, v);
                        tryReanchor(v, u);
                    }
                }                    
                    

                
            }
        } else {
            // Do nothing for the first element
            enterpoint_node_ = 0;
            // curlevel = 1;
            maxlevel_ = curlevel;
            anchor_layer_ = 1;
            if (curlevel == anchor_layer_) {
                // remember which node we pierced
                anchor_labels_[cur_c] = cur_c;
                // attach this new node under that anchor
                auto &vec = anchor_connected_nodes_[cur_c];
                vec.push_back(cur_c);

                // 1) compute and record its dist to its anchor
                dist_t d = 0;
                node_anchor_dist_[cur_c] = d;

                // 2) insert into that anchor's sorted set
                auto &S = anchor_children_by_dist_[0];
                auto insert_res = S.emplace(d, cur_c);
                auto it         = insert_res.first;
                bool ok         = insert_res.second; 
                anchor_child_it_[cur_c] = it;      

                anchor_list_.insert(cur_c);

            }            
        }

        // Releasing lock for the maximum level
        if (curlevel > maxlevelcopy) {
            enterpoint_node_ = cur_c;
            maxlevel_ = curlevel;
        }
        return cur_c;
    }


    std::priority_queue<std::pair<dist_t, labeltype >>
    searchKnn(const void *query_data, size_t k, BaseFilterFunctor* isIdAllowed = nullptr) const {
        std::priority_queue<std::pair<dist_t, labeltype >> result;
        if (cur_element_count == 0) return result;

        tableint currObj = enterpoint_node_;
        dist_t curdist = fstdistfunc_(query_data, getDataByInternalId(enterpoint_node_), dist_func_param_);

        for (int level = maxlevel_; level > 0; level--) {
            bool changed = true;
            while (changed) {
                changed = false;
                unsigned int *data;

                data = (unsigned int *) get_linklist(currObj, level);
                int size = getListCount(data);
                metric_hops++;
                metric_distance_computations+=size;

                tableint *datal = (tableint *) (data + 1);
                for (int i = 0; i < size; i++) {
                    tableint cand = datal[i];
                    if (cand < 0 || cand > max_elements_)
                        throw std::runtime_error("cand error");
                    dist_t d = fstdistfunc_(query_data, getDataByInternalId(cand), dist_func_param_);

                    if (d < curdist) {
                        curdist = d;
                        currObj = cand;
                        changed = true;
                    }
                }
            }
        }

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        bool bare_bone_search = !num_deleted_ && !isIdAllowed;
        if (bare_bone_search) {
            top_candidates = searchBaseLayerST<true>(
                    currObj, query_data, std::max(ef_, k), isIdAllowed);
        } else {
            top_candidates = searchBaseLayerST<false>(
                    currObj, query_data, std::max(ef_, k), isIdAllowed);
        }

        while (top_candidates.size() > k) {
            top_candidates.pop();
        }
        while (top_candidates.size() > 0) {
            std::pair<dist_t, tableint> rez = top_candidates.top();
            result.push(std::pair<dist_t, labeltype>(rez.first, getExternalLabel(rez.second)));
            top_candidates.pop();
        }
        return result;
    }


    std::vector<std::pair<dist_t, labeltype >>
    searchStopConditionClosest(
        const void *query_data,
        BaseSearchStopCondition<dist_t>& stop_condition,
        BaseFilterFunctor* isIdAllowed = nullptr) const {
        std::vector<std::pair<dist_t, labeltype >> result;
        if (cur_element_count == 0) return result;

        tableint currObj = enterpoint_node_;
        dist_t curdist = fstdistfunc_(query_data, getDataByInternalId(enterpoint_node_), dist_func_param_);

        for (int level = maxlevel_; level > 0; level--) {
            bool changed = true;
            while (changed) {
                changed = false;
                unsigned int *data;

                data = (unsigned int *) get_linklist(currObj, level);
                int size = getListCount(data);
                metric_hops++;
                metric_distance_computations+=size;

                tableint *datal = (tableint *) (data + 1);
                for (int i = 0; i < size; i++) {
                    tableint cand = datal[i];
                    if (cand < 0 || cand > max_elements_)
                        throw std::runtime_error("cand error");
                    dist_t d = fstdistfunc_(query_data, getDataByInternalId(cand), dist_func_param_);

                    if (d < curdist) {
                        curdist = d;
                        currObj = cand;
                        changed = true;
                    }
                }
            }
        }

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        top_candidates = searchBaseLayerST<false>(currObj, query_data, 0, isIdAllowed, &stop_condition);

        size_t sz = top_candidates.size();
        result.resize(sz);
        while (!top_candidates.empty()) {
            result[--sz] = top_candidates.top();
            top_candidates.pop();
        }

        stop_condition.filter_results(result);

        return result;
    }


    void checkIntegrity() {
        int connections_checked = 0;
        std::vector <int > inbound_connections_num(cur_element_count, 0);
        for (int i = 0; i < cur_element_count; i++) {
            for (int l = 0; l <= element_levels_[i]; l++) {
                linklistsizeint *ll_cur = get_linklist_at_level(i, l);
                int size = getListCount(ll_cur);
                tableint *data = (tableint *) (ll_cur + 1);
                std::unordered_set<tableint> s;
                for (int j = 0; j < size; j++) {
                    assert(data[j] < cur_element_count);
                    assert(data[j] != i);
                    inbound_connections_num[data[j]]++;
                    s.insert(data[j]);
                    connections_checked++;
                }
                assert(s.size() == size);
            }
        }
        if (cur_element_count > 1) {
            int min1 = inbound_connections_num[0], max1 = inbound_connections_num[0];
            for (int i=0; i < cur_element_count; i++) {
                assert(inbound_connections_num[i] > 0);
                min1 = std::min(inbound_connections_num[i], min1);
                max1 = std::max(inbound_connections_num[i], max1);
            }
            std::cout << "Min inbound: " << min1 << ", Max inbound:" << max1 << "\n";
        }
        std::cout << "integrity ok, checked " << connections_checked << " connections\n";
    }


    std::vector<labeltype> reverseSearchKnn(const void* query_data, size_t k, float recall = 1.0f) {
        auto t0 =std::chrono::high_resolution_clock::now();
        // 1. Determine hop count k' via Algorithm 1
        size_t kprime = getHopCountForQuery(k, recall);

        // 2. Insert query temporarily at level 0 to get its 1-hop neighbors
        std::vector<tableint> curHop;
        std::vector<tableint> candidateSet;

        auto t1 =std::chrono::high_resolution_clock::now();

        auto t_approx = std::chrono::duration<double, std::milli>(t1 - t0).count();

        std::cout<<"Initial takes "<<t_approx<<std::endl;          
        tableint qid;
        {
            // labeltype qlabel = std::numeric_limits<labeltype>::max();
            qid = this->searchKnn(query_data, kprime).top().second;
            curHop = getConnectionsWithLock(qid, 0);
            candidateSet = curHop;
            // markDelete(qlabel);
        }

        // 2.5 Compute query region density: k-NN around query
        auto qknn = this->searchKnn(query_data, density_k_);
        double far_q = qknn.top().first;
        double query_density = double(density_k_) / far_q;

        // 3. Estimate threshold ED_k'
        double ED = estimateED(query_data, kprime, curHop);
      


        t0 =std::chrono::high_resolution_clock::now();

        // 4. BFS outward up to k' hops with PS1 and density pruning
        for (size_t h = 2; h <= kprime; ++h) {
            std::vector<tableint> nextHop;
            for (auto pid : curHop) {
                auto neigh = getConnectionsWithLock(pid, 0);
                for (auto vid : neigh) {
                    // if (!prunePS1(query_data, vid, ED)) {
                    //     continue;}
                    // if (!pruneDensity(query_density, vid)) {
                    //     continue;}
                    candidateSet.push_back(vid);
                    nextHop.push_back(vid);
                }
            }
            curHop.swap(nextHop);
            if (curHop.empty()) break;
        }
        t1 =std::chrono::high_resolution_clock::now();
        t_approx = std::chrono::duration<double, std::milli>(t1 - t0).count();

        // std::cout<<"There used to be"<<candidateSet.size()<<"candidates\n";
        std::set<tableint> uni;
        for (auto i: candidateSet  )
        {
            uni.insert(i);
        }
        std::cout<<"There "<<uni.size()<<"unique candidates\n";
        
        
        
        // 5. Pruning strategy PS2 (plus density pruning)
        std::unordered_set<tableint> verifiedCandidates;
        for (auto oid : uni) {
            // if (!prunePS2(query_data, oid, kprime, k)) continue;
            if (!prunePS1(query_data, oid, ED)) {
                continue;}
            if (!pruneDensity(query_density, oid)) {
                continue;}            
            if (!pruneRadius(query_density,oid,query_data)){ continue; }
            verifiedCandidates.insert(oid);
        }
        
        std::cout<<"Prun2 takes "<<t_approx<<std::endl;
        std::cout<<"There "<<verifiedCandidates.size()<<" candidates left\n";
        // 6. Verification: keep those where query appears in their k-NN
        t0 =std::chrono::high_resolution_clock::now();

        std::unordered_set<labeltype> results;
        for (auto oid : verifiedCandidates) {
            char* data_o = getDataByInternalId(oid);
            // results.insert(getExternalLabel(oid));
            double ret_dis = fstdistfunc_(query_data, getDataByInternalId(oid), dist_func_param_);

            auto knn = this->searchKnn(data_o, k+3);
            auto p = knn.top();
            double bench_dis = p.first;
            if (getExternalLabel(oid) != std::numeric_limits<labeltype>::max() and ret_dis <= bench_dis) {
                results.insert(getExternalLabel(oid));
            }
        }
        std::vector<labeltype> final_results(results.begin(), results.end());
        t1 =std::chrono::high_resolution_clock::now();
        t_approx = std::chrono::duration<double, std::milli>(t1 - t0).count();

        std::cout<<"Verif takes "<<t_approx<<std::endl;
        return final_results;
    }

    /**
     * Set pr(h) distribution for hop-count calculation (Algorithm 1).
     * pr_dist_[h] = pr(p, h) as per the paper.
     */
    void setPrDistribution(const std::vector<double>& pr) {
        pr_dist_ = pr;
    }

    /**
     * Set the density ratio threshold: candidate density must be >=
     * (query_density * ratio) to be kept.
     */
    void setDensityRatioThreshold(double ratio) {
        density_ratio_threshold_ = ratio;
    }

    /* ------------------------------------------------------------------ */
    /*                    AUX-ILIARY HELPERS FOR JOIN                     */
    /* ------------------------------------------------------------------ */

    /** Return true if the node is an anchor (its own anchor-label). */
    bool isAnchor(tableint id) const noexcept {
        return anchor_labels_[id] == id;
    }

    /** List of all anchors in *this* index. */
    std::vector<tableint> getAnchors() const {
        std::vector<tableint> out;
        out.reserve(cur_element_count);
        for (tableint i = 0; i < cur_element_count; ++i)
            if (isAnchor(i)) out.push_back(i);
        return out;
    }

    std::unordered_set<tableint> get_anchor_list() const {
        return anchor_list_;
    }

    /** Constant reference to the cluster of an anchor. */
    const std::vector<tableint>& getCluster(tableint anchor) const {
        return anchor_connected_nodes_[anchor];
    }

    /** Geometric radius of an anchor’s cluster (k / LD). */
    double anchorRadius(tableint anchor, size_t k) const {
        double ld = local_density_[anchor];
        if (ld <= 0) return std::numeric_limits<double>::infinity();
        return static_cast<double>(k) / ld;
    }

    /* ------------------------------------------------------------------ */
    /*                    ENTRY-POINT OVERRIDE SEARCH                     */
    /* ------------------------------------------------------------------ */

    /** Same behaviour as searchKnn(), but the walk starts at |entry_id|. */
    std::priority_queue<std::pair<dist_t, tableint>,
                        std::vector<std::pair<dist_t, tableint>>,
                        CompareByFirst>
    searchKnnEP(const void* query_data,
                size_t k,
                tableint entry_id,
                size_t ef_override = 0) const
    {
        size_t old_ef = ef_;
        if (ef_override) const_cast<HierarchicalNSW*>(this)->ef_ = ef_override;

        std::priority_queue<std::pair<dist_t, tableint>,
                            std::vector<std::pair<dist_t, tableint>>,
                            CompareByFirst> result;

        if (cur_element_count == 0) { if (ef_override) const_cast<HierarchicalNSW*>(this)->ef_ = old_ef; return result; }

        tableint currObj = entry_id;
        dist_t curdist = fstdistfunc_(query_data, getDataByInternalId(currObj), dist_func_param_);

        // for (int level = maxlevel_; level > 0; --level) {
        //     bool changed = true;
        //     while (changed) {
        //         changed = false;
        //         linklistsizeint * data;
        //         data = (linklistsizeint *)get_linklist(currObj, level);
        //         int sz   = getListCount(data);
        //         tableint* datal = (tableint*)(data + 1);
        //         for (int i = 0; i < sz; ++i) {
        //             tableint cand = datal[i];
        //             dist_t d      = fstdistfunc_(query_data, getDataByInternalId(cand), dist_func_param_);
        //             if (d < curdist) { curdist = d; currObj = cand; changed = true; }
        //         }
        //     }
        // }

        bool bare = !num_deleted_;
        auto pq = bare ? searchBaseLayerST<true>(currObj, query_data,
                                                 std::max(ef_, k))
                       : searchBaseLayerST<false>(currObj, query_data,
                                                  std::max(ef_, k));

        // while (pq.size() > k) pq.pop();
        while (!pq.empty()) {
            result.emplace(pq.top().first, pq.top().second); pq.pop();
        }

        if (ef_override) const_cast<HierarchicalNSW*>(this)->ef_ = old_ef;
        return result;
    }

    /* ------------------------------------------------------------------ */
    /*                         APPROXIMATE k-NN JOIN                      */
    /* ------------------------------------------------------------------ */

    /**
     * Perform an approximate k-NN join:  for every point in *this* index (HA)
     * return its k nearest neighbours drawn from index HB.
     *
     * @param HB         the right-hand HNSW index
     * @param k          how many neighbours per point
     * @param ef_join    ef parameter used during each HB search (0 ⇒ 3·k)
     * @param anchor_L   how many nearest HB-anchors to probe per HA-anchor
     *
     * @return  result[a] = vector of k external labels from HB
     */
    std::vector<std::vector<labeltype>>
    knnJoin(HierarchicalNSW<dist_t>& HB,
            size_t k,
            size_t ef_join  = 0,
            size_t anchor_L = 32) const
    {
        if (ef_join == 0) ef_join = 3 * k;

        /* -------- pre-compute anchor lists & radii ------------------- */
        auto anchorsA = this->getAnchors();
        auto anchorsB = HB.getAnchors();

        std::vector<double> radA(this->cur_element_count, 0.0),
                            radB(HB.cur_element_count,      0.0);
        for (auto p : anchorsA) radA[p] = this->anchorRadius(p, k);
        for (auto q : anchorsB) radB[q] = HB.  anchorRadius(q, k);

        /* -------- initialise output structure ------------------------ */
        std::vector<std::vector<labeltype>>
            result(this->cur_element_count);   // by internal ID in *this*

        /* -------- process each anchor p ------------------------------ */
        for (tableint p : anchorsA) {

            /* 2A – gather at most anchor_L nearest HB anchors */
            auto pqB = HB.searchKnn(this->getDataByInternalId(p),
                                    anchor_L);   // ordinary search
            std::vector<tableint> candidateB;
            while (!pqB.empty()) {
                tableint q = pqB.top().second; pqB.pop();
                if (!HB.isAnchor(q)) continue;  // keep true anchors only

                dist_t d_c = this->fstdistfunc_(this->getDataByInternalId(p),
                                                HB.getDataByInternalId(q),
                                                this->dist_func_param_);
                if (d_c <= radA[p] + radB[q])
                    candidateB.push_back(q);
            }
            if (candidateB.empty()) continue;   // p’s cluster cannot match HB

            /* 3 – for every point a in cluster(p), search HB starting at q */
            for (tableint a : this->getCluster(p)) {
                const void* vecA = this->getDataByInternalId(a);


                /* collect best distance for every distinct ID */
                std::unordered_map<tableint, dist_t> best;                
                for (tableint q : candidateB) {
                    auto knnB = HB.searchKnnEP(vecA, k+3, q, ef_join);

                    while (!knnB.empty()) {

                        tableint id   = knnB.top().second;
                        dist_t   dist = knnB.top().first;
                        auto it = best.find(id);
                        if (it == best.end() || dist < it->second)
                            best[id] = dist;                        
                        knnB.pop();
                    }
                }
                /* build min-heap from the unique set, keep k closest */
                std::priority_queue<std::pair<dist_t, tableint>,
                                    std::vector<std::pair<dist_t, tableint>>,
                                    CompareByFirst> topk;
                for (const auto& kv : best) {
                    topk.emplace(kv.second, kv.first);
                    if (topk.size() > k) topk.pop();
                }

                /* store external labels in ascending-distance order */
                std::vector<labeltype>& out = result[a];
                // std::cout<<a<<" ";
                out.resize(topk.size());
                for (size_t idx = topk.size(); idx > 0; --idx) {
                    out[idx-1] = HB.getExternalLabel(topk.top().second);
                    topk.pop();
                }
            } // end for each a∈cluster(p)
        }     // end for each anchor p
        return result;
    }
    /****************  Bridge & Burst helpers  ****************/

    /** quick ED-radius:  R = (k′/LD)·d_k   with LD = k / d_k  */
    float getEdRadius(tableint anchor, size_t kprime) const {
        if (local_density_[anchor] == 0.0) return std::numeric_limits<float>::max();
        float dk = static_cast<float>(density_k_) / static_cast<float>(local_density_[anchor]);
        return dk * static_cast<float>(kprime) / static_cast<float>(density_k_);
    }

    /** raw vector distance between two data pointers (same metric) */
    dist_t rawDist(const void* a, const void* b) const {
        return fstdistfunc_(a, b, dist_func_param_);
    }

    /** distance between internal ids that may live in different indices */
    dist_t distById(tableint ida, const HierarchicalNSW& other, tableint idb) const {
        return fstdistfunc_(getDataByInternalId(ida),
                            other.getDataByInternalId(idb),
                            dist_func_param_);
    }

    /**
     * Restricted k-NN: only nodes whose **internal id** is in
     * `allowed_ids` are eligible (uses the label-filter hook internally).
     */
    template<typename IdSet>
    std::priority_queue<std::pair<dist_t, labeltype>>
    searchKnnRestricted(const void* query_data,
                        size_t k,
                        const IdSet& allowed_ids,
                        size_t ef_override = 0) const
    {
        struct Allowed : public BaseFilterFunctor {
            const HierarchicalNSW* idx;
            const IdSet*           allowed;
            Allowed(const HierarchicalNSW* i, const IdSet* s)
                : idx(i), allowed(s) {}
            bool operator()(labeltype lab) const  {
                auto it = idx->label_lookup_.find(lab);
                if (it == idx->label_lookup_.end()) return false;
                return allowed->count(it->second) != 0;
            }
        } functor(this,&allowed_ids);

        size_t old = ef_;
        if (ef_override) const_cast<HierarchicalNSW*>(this)->setEf(ef_override);
        auto pq = searchKnn(query_data, k, &functor);
      
        if (ef_override) const_cast<HierarchicalNSW*>(this)->setEf(old);
        return pq;
    }    
private:
    // pr(p, h) distribution for h = 1..H
    std::vector<double> pr_dist_;

    // density-based pruning parameters
    // size_t density_k_ = 20;
    double density_ratio_threshold_ = 0.6;  // keep candidates whose region density >= 0.5 * query_density

    /** Algorithm 1: choose hop count k' */
    size_t getHopCountForQuery(size_t k, float recall) const {
        size_t H = std::min(k, (size_t)std::ceil(std::log((double)max_elements_) / std::log((double)M_)));
        size_t kprime = 0;
        double acc = 0.0;
        while (acc < recall && kprime < H) {
            ++kprime;
            if (kprime >= pr_dist_.size()) break;
            acc += pr_dist_[kprime];
        }
        return std::min(kprime, H);
    }

    /** Parzen estimator ED(k') = ED1 * (ω * k')^(1/d) */
    double estimateED(const void* query_data, size_t kprime,
                      const std::vector<tableint>& oneHop) const {
        double ED1 = 0.0;
        for (auto pid : oneHop) {
            double d = fstdistfunc_(query_data, getDataByInternalId(pid), dist_func_param_);
            ED1 = std::max(ED1, d);
        }
        double sum_deg = 0.0;
        for (size_t i = 0; i < cur_element_count; ++i)
            sum_deg += getConnectionsWithLock(i, 0).size();
        double omega = sum_deg / (double)cur_element_count;
        size_t dim = *((size_t*)dist_func_param_);
        double factor = std::pow(omega * kprime, 1.0 / dim);
        return ED1 * factor;
    }

    /** PS1: prune if distance > ED threshold */
    bool prunePS1(const void* query_data, tableint node_id, double ED) const {
        double d = fstdistfunc_(query_data, getDataByInternalId(node_id), dist_func_param_);
        return d <= ED;
    }

    /** PS2: prune if d(q,o) > l-th neighbor distance of o */
    bool prunePS2(const void* query_data, tableint node_id,
                  size_t kprime, size_t k) const {
        double d_o_q = fstdistfunc_(query_data, getDataByInternalId(node_id), dist_func_param_);
        // auto neigh = getConnectionsWithLock(node_id, 0);
        auto neigh = this->searchKnn(query_data,k);
        if (neigh.empty()) return false;
        std::vector<double> ds;
        ds.reserve(neigh.size());
        auto neigh_temp = neigh;
        while (!neigh_temp.empty())
        {
            tableint nid = neigh_temp.top().second;
            ds.push_back(fstdistfunc_(getDataByInternalId(nid), getDataByInternalId(node_id), dist_func_param_));
            neigh_temp.pop();
        }
        
        size_t l = std::min(k, neigh.size());
        std::nth_element(ds.begin(), ds.begin() + l - 1, ds.end());
        return d_o_q <= ds[l - 1];
    }

    /** Density-based prune: keep if region density >= ratio * query density */
    bool pruneDensity(double query_density, tableint node_id) const {
        int a = anchor_labels_[node_id];
        if (a < 0) return true;  // no anchor info, do not prune
        double dens = local_density_[a];
        return dens <= 2*query_density ;
    }
    bool pruneRadius(double query_density, tableint node_id, const void * query_data) const {
        int a = anchor_labels_[node_id];
        if (a < 0) return true;  // no anchor info, do not prune
        double r1 = 40/ local_density_[a] ;
        // double r2 = query_density * density_k_;
        dist_t dist = fstdistfunc_(query_data, getDataByInternalId(node_id), dist_func_param_);
        // std::cout<<"dist = "<<r1;
        return dist < r1;
    }        


    // If u is now linked at level 0 to v, we call
    //   tryReanchor(u, v);
    // which tests whether v's anchor is *closer* to u
    // than u's current anchor is, and if so migrates u.
    void tryReanchor(tableint u, tableint v) {
        // std::cout<<"How are ";

        // 1) bounds safety
        assert(u < cur_element_count);
        assert(v < cur_element_count);

        // 2) our anchor vectors must be sized for all elements
        assert(anchor_children_by_dist_.size() == max_elements_);
        assert(anchor_child_it_.size()            == max_elements_);
        assert(node_anchor_dist_.size()          == max_elements_); 

        
        tableint oldA = anchor_labels_[u];
        tableint newA = anchor_labels_[v];
        // 3) old anchor must be valid, and iterator must live in that set
        assert(oldA < anchor_children_by_dist_.size());
        auto &oldSet = anchor_children_by_dist_[oldA];
  
        if (newA == oldA) return;
        
        // compute dist(u, newA)
        char *pu = getDataByInternalId(u);
        char *pa = getDataByInternalId(newA);
        dist_t dnew = fstdistfunc_(pu, pa, dist_func_param_);

        // compare to u’s current anchor‐distance
        if (dnew < node_anchor_dist_[u]) {
            // 1) remove u from old anchor set
            anchor_children_by_dist_[oldA].erase(anchor_child_it_[u]);
        {
            auto &oldVec = anchor_connected_nodes_[oldA];
            oldVec.erase(std::remove(oldVec.begin(), oldVec.end(), u),
                         oldVec.end());
        }            

            // 2) insert into new anchor set
            auto &S = anchor_children_by_dist_[newA];
            // auto [it,ok] = S.emplace(dnew, u);
            auto insert_res = S.emplace(dnew, u);
            auto it         = insert_res.first;
            bool ok         = insert_res.second;             
            anchor_child_it_[u] = it;

            // 3) update state
            anchor_labels_[u]    = newA;
            node_anchor_dist_[u] = dnew;
        }
    }

};
}  // namespace hnswlib
