#include "DataSet.cpp"
#include "Config.cpp"
#include "NSW.cpp"
#include "../res/kmeans/kmeans.cpp"
#include <algorithm>
#include <map>
#include <unordered_map>
#include <cmath>
using namespace std::literals;

class QUERYANS {
public:
    int us;
    double recall;
    QUERYANS(int us, double recall) : us(us), recall(recall) {}
};

void solveQueryAns(std::vector<QUERYANS> &queryAns, int queryNum) {
    double avgRecall = 0;
    double avgTime = 0;
    for (auto & item : queryAns) {
        avgRecall += item.recall;
        avgTime += item.us;
    }
    avgRecall /= queryAns.size();
    avgTime /= queryAns.size();
    std::cout << "avgRecall: " << avgRecall * 100 << "% , avgTime:" << avgTime << " us" << std::endl;
}

/**
 * @brief 从一个图中查询 KNN
 * 输入：图，K
*/
void queryAnn(DataSet <FILETYPE> *dataSet) {
    std::cout << "========queryAnn============" << std::endl;
    std::cout << "query size: " << dataSet->queryData.size() << std::endl;
    int index = 0;
    createNSWGraph(dataSet, M);
    /**
     * 遍历所有查询
    */
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
    #endif
}