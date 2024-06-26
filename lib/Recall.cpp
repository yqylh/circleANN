#ifndef __recall_CPP__
#define __recall_CPP__

#include "Config.cpp"

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

#endif