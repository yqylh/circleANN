#ifndef Circle_CPP
#define Circle_CPP
#include "DataSet.cpp"
#include "Config.cpp"
#include "Recall.cpp"
#include "../res/kmeans/kmeans.cpp"
using namespace std::literals;

struct Node {
    int id;
    Item<FILETYPE> *v;
    std::vector<Node *> edge;
    Node() {}
    Node(int id, Item<FILETYPE> *v) : id(id), v(v) {}
};

struct cmpGreater {
    bool operator()(std::pair<float, Node *> &a, std::pair<float, Node *> &b) {
        return a.first > b.first;
    }
};

struct cmpLess {
    bool operator()(std::pair<float, Node *> &a, std::pair<float, Node *> &b) {
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
    void solveEdge();
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
    std::priority_queue<std::pair<float, Node*>, std::vector<std::pair<float, Node*>>, cmpGreater> candidate; // 小顶堆
    std::priority_queue<std::pair<float, Node*>, std::vector<std::pair<float, Node*>>, cmpLess> topK; // 大顶堆

    bool *visited = new bool[nodes.size()];
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
            if (visited[toId->id] == false) {
                float distance = query - *toId->v;
                if (distance > topK.top().first) continue;
                candidate.push(std::make_pair(distance, toId));
                visited[toId->id] = true;
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
        float recall = 0;
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

void Circle::solveEdge() {
    // 按照凸包分成多个圈
    std::vector<std::vector<Node*>> circles;
    bool *visited = new bool[nodes.size() + 10];
    for (int i = 0; i <= nodes.size(); i++) visited[i] = false;
    int insertCount = 0;
    std::vector< std::priority_queue<std::pair<float, Node*>, std::vector<std::pair<float, Node*>>, cmpLess> > min(D);
    std::vector< std::priority_queue<std::pair<float, Node*>, std::vector<std::pair<float, Node*>>, cmpGreater> > max(D);
    for (auto & node : nodes) {
        for (int i = 0; i < D; i++) {
            min[i].push(std::make_pair(node->v->vectors[i], node));
            max[i].push(std::make_pair(node->v->vectors[i], node));
        }
    }
    while (insertCount < nodes.size()) {
        std::vector<Node *> circle;
        if (nodes.size() - insertCount < D * 2) {
            for (int i = 0; i < nodes.size(); i++) {
                if (visited[nodes[i]->id] == false) {
                    circle.push_back(nodes[i]);
                    insertCount++;
                }
            }
            circles.push_back(circle);
            break;
        }
        for (int i = 0; i < D; i++) {
            bool insertMin = false;
            while (insertMin == false) {
                if (min[i].empty()) break;
                auto minItem = min[i].top();
                if (visited[minItem.second->id] == false) {
                    circle.push_back(minItem.second);
                    visited[minItem.second->id] = true;
                    insertCount++;
                    insertMin = true;
                }
                min[i].pop();
            }
            bool insertMax = false;
            while (insertMax == false) {
                if (max[i].empty()) break;
                auto maxItem = max[i].top();
                if (visited[maxItem.second->id] == false) {
                    circle.push_back(maxItem.second);
                    visited[maxItem.second->id] = true;
                    insertCount++;
                    insertMax = true;
                }
                max[i].pop();
            }
        }
        circles.push_back(circle);
    }
    std::cout << "circle size: " << circles.size() << std::endl;
    #pragma omp parallel for num_threads(THREAD_CONFIG)
    for (auto & circle : circles) {
        for (auto & node : circle) {
            std::vector<std::pair<float, Node *>> distance;
            for (auto & item : circle) {
                if (item == node) continue;
                distance.push_back(std::make_pair(*item->v - *node->v, item));
            }
            std::sort(distance.begin(), distance.end(), cmpLess());
            for (int i = 0; i < D*2/LAYER_P; i++) {
                node->edge.push_back(distance[i].second);
            }
        }
    }
    delete[] visited;
}
#endif