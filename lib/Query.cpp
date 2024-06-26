#ifndef QUERY_CPP
#define QUERY_CPP
#include "DataSet.cpp"
#include "Config.cpp"
#include "Recall.cpp"
#include "../res/kmeans/kmeans.cpp"
using namespace std::literals;

/**
 * @brief 从一个图中查询 ANN
*/
std::vector<int> GraphQuery(DataSet<FILETYPE> *dataSet, Item<FILETYPE> &query, int k, std::vector<int> beginVector = std::vector<int>()) {
    std::priority_queue<std::pair<double, int>, std::vector<std::pair<double, int>>, std::greater<std::pair<double, int>>> candidate;
    std::priority_queue<std::pair<double, int>, std::vector<std::pair<double, int>>, std::less<std::pair<double, int>>> topK;
    std::unordered_set<int> visited;
    // 初始化候选者
    if (beginVector.empty()) {
        candidate.push(std::make_pair(query - dataSet->baseData[0], 0));
    } else {
        for (auto & item : beginVector) {
            candidate.push(std::make_pair(query - dataSet->baseData[item], item));
        }
    }
    while (!candidate.empty()) {
        // 每次取出一个
        auto item = candidate.top();
        candidate.pop();
        // 将候选者插入到候选集里
        if (topK.size() == k && item.first > topK.top().first) {
            break;
        } else {
            topK.push(item);
            // 如果超过了 K 个,则删除最大的一个
            if (topK.size() > k) {
                topK.pop();
            }
        }
        if (CREATEGRAPH == 1) mtx[item.second].lock();
        // 遍历所有的邻居
        for (int i = 0; i < dataSet->baseData[item.second].edge.size(); i++) {
            int indexValue = dataSet->baseData[item.second].edge[i];
            // 如果没有插入过,则加入到候选者中
            if (visited.find(indexValue) == visited.end()) {
                double distance = query - dataSet->baseData[indexValue];
                if (distance > topK.top().first) {
                    continue;
                }
                candidate.push(std::make_pair(distance, indexValue));
                visited.insert(indexValue);
            }
        }
        if (CREATEGRAPH == 1) mtx[item.second].unlock();
    }
    // 将结果转换为 vector
    std::vector<int> ans;
    while (!topK.empty()) {
        ans.push_back(topK.top().second);
        topK.pop();
    }
    return ans;
}

/**
 * @brief 从一个图中查询 ANN
*/
void queryAnn(DataSet <FILETYPE> *dataSet, int K) {
    std::cout << "========queryAnn============" << std::endl;
    std::cout << "query size: " << dataSet->queryData.size() << std::endl;
    int index = 0;

    // 遍历所有查询
    std::vector<QUERYANS> queryAns;
    std::mutex insertQueryItemMtx;
    #pragma omp parallel for num_threads(THREAD_CONFIG) 
    for (int i = 0; i < dataSet->queryData.size(); i++) {
        auto start = std::chrono::steady_clock::now();
        std::vector<int> ans = GraphQuery(dataSet, dataSet->queryData[i], K);
        auto end = std::chrono::steady_clock::now();
        double recall = 0;
        for (auto & item : ans) {
            for (auto & ansItem : dataSet->ansData[i].vectors) {
                if (item == ansItem) {
                    recall++;
                    break;
                }
            }
        }
        recall = recall / K;
        insertQueryItemMtx.lock();
        queryAns.emplace_back((end-start) / 1us, recall);
        insertQueryItemMtx.unlock();
    }
    solveQueryAns(queryAns, dataSet->queryData.size());
}
#endif