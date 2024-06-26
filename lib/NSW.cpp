#ifndef __GraphQuery_CPP__
#define __GraphQuery_CPP__

#include "DataSet.cpp"
#include "Config.cpp"
#include <algorithm>
#include <queue>
#include <unordered_set>
#include <vector>
#include <cmath>
#include <cstdio>
#include <mutex>


/**
 * @brief 从一个图中查询 KNN
 * 输入：图，查询点，K
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

void WriteNSWGraph(DataSet<FILETYPE> *dataSet) {
    // 存储到文件中
    std::ofstream file("./dataset/index/NSW" + std::to_string(DatabaseSelect) + ".edge");
    for (auto & item : dataSet->baseData) {
        for (auto & edge : item.edge) file << edge << " ";
        file << std::endl;
    }
    file.close();
}
void createNSWGraph(DataSet<FILETYPE> *dataSet, int M) {
    // 如果文件存在，直接读取
    std::ifstream file("./dataset/index/NSW" + std::to_string(DatabaseSelect) + ".edge");
    if (file.is_open()) {
        std::cout << "NSW.edge is exist" << std::endl;
        // 从文件中读取
        std::string line;
        int index = 0;
        while (std::getline(file, line)) {
            std::stringstream ss(line);
            int edge;
            while (ss >> edge) {
                dataSet->baseData[index].edge.push_back(edge);
            }
            index++;
        }
        std::cout << "NSW.edge is loaded" << std::endl;
    } else {
        // 从头开始构建
        mtx = new std::mutex[dataSet->baseData.size()];
        std::cout << "NSW.edge is not exist" << std::endl;
        CREATEGRAPH = 1;
        #pragma omp parallel for num_threads(THREAD_CONFIG) 
        for (int i = 0; i < dataSet->baseData.size(); i++) {
            if (i % 100 == 0)  std::cout << "createNSWGraph: " << (i + 0.0)/ dataSet->baseData.size() << std::endl;
            auto ans = GraphQuery(dataSet, dataSet->baseData[i], M);
            for (auto & item : ans) {
                if (item != i) {
                    insertEdge(dataSet, i, item, M);
                }
            }
        }
        std::cout << "NSW.edge is created" << std::endl;
        CREATEGRAPH = 0;
        WriteNSWGraph(dataSet);
        delete [] mtx;
        std::cout << "NSW.edge is saved" << std::endl;
    }
    file.close();
    return;
}

#endif