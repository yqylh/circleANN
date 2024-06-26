#ifndef Circle_CPP
#define Circle_CPP
#include "DataSet.cpp"
#include "Config.cpp"
#include "Recall.cpp"
#include "../res/kmeans/kmeans.cpp"
using namespace std::literals;

struct Node{
    int id;
    Item<FILETYPE> *v;
    std::vector<Node *> edge;
    Node() {}
    Node(int id, Item<FILETYPE> *v) : id(id), v(v) {}
};
struct cmpGreater {
    bool operator()(std::pair<double, Node *> &a, std::pair<double, Node *> &b) {
        return a.first > b.first;
    }
};
struct cmpLess {
    bool operator()(std::pair<double, Node *> &a, std::pair<double, Node *> &b) {
        return a.first < b.first;
    }
};
struct Circle {
    // value
    DataSet<FILETYPE> *dataSet;
    int K;
    std::vector<Node *> nodes;
    // func
    std::vector<int> GraphQuery(Item<float> &query, std::vector<int> beginVector = std::vector<int>());
    void queryAnn();
    // 构造和析构
    void setNodes() {
        nodes = std::vector<Node *>(dataSet->baseData.size());
        for (int i = 0; i < dataSet->baseData.size(); i++)
            nodes[i] = new Node(i, &dataSet->baseData[i]);
    }
    Circle(DataSet<FILETYPE> *dataSet, int K) : dataSet(dataSet), K(K) {
        setNodes();
    }
    ~Circle() {
        for (auto & item : nodes) delete item;
    }
};


/**
 * @brief 从一个图中查询 ANN
*/
std::vector<int> Circle::GraphQuery(Item<float> &query, std::vector<int> beginVector) {
    std::priority_queue<std::pair<double, Node*>, std::vector<std::pair<double, Node*>>, cmpGreater> candidate; // 小顶堆
    std::priority_queue<std::pair<double, Node*>, std::vector<std::pair<double, Node*>>, cmpLess> topK; // 大顶堆

    std::unordered_set<int> visited;
    // 初始化候选者
    if (beginVector.empty()) {
        candidate.push(std::make_pair(query - *nodes[0]->v, nodes[0]));
    } else {
        for (auto & item : beginVector) {
            candidate.push(std::make_pair(query - *nodes[item]->v, nodes[item]));
        }
    }
    while (!candidate.empty()) {
        // 每次取出一个
        auto item = candidate.top();
        candidate.pop();
        // 将候选者插入到候选集里
        if (topK.size() == K && item.first > topK.top().first) {
            break;
        } else {
            topK.push(item);
            // 如果超过了 K 个,则删除最大的一个
            if (topK.size() > K) topK.pop();
        }
        // 遍历所有的邻居
        for (auto & toId : item.second->edge) {
            // 如果没有插入过,则加入到候选者中
            if (visited.find(toId->id) == visited.end()) {
                double distance = query - *toId->v;
                if (distance > topK.top().first) continue;
                candidate.push(std::make_pair(distance, toId));
                visited.insert(toId->id);
            }
        }
    }
    // 将结果转换为 vector
    std::vector<int> ans;
    while (!topK.empty()) {
        ans.push_back(topK.top().second->id);
        topK.pop();
    }
    return ans;
}

/**
 * @brief 从一个图中查询 ANN
*/
void Circle::queryAnn() {
    std::cout << "========queryAnn============" << std::endl;
    std::cout << "query size: " << dataSet->queryData.size() << std::endl;
    int index = 0;

    // 遍历所有查询
    std::vector<QUERYANS> queryAns;
    std::mutex insertQueryItemMtx;
    #pragma omp parallel for num_threads(THREAD_CONFIG) 
    for (int i = 0; i < dataSet->queryData.size(); i++) {
        auto start = std::chrono::steady_clock::now();
        std::vector<int> ans = GraphQuery(dataSet->queryData[i]);
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