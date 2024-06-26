/**
 * DataSet.cpp
 * 用于处理数据集的输入
 * 作者：yqy
 * 日期：2022.11.8
 * 主体结构:Item记录每个向量的信息，DataSet记录数据集的信息
 * DataSet 是一个抽象类，其派生类有：表示对于不同数据集的处理,目标是将不同的数据都处理为统一格式
 * 使用时将派生类的对象转换成 DataSet 类
 * 
*/

#ifndef __DATASET_H__
#define __DATASET_H__
#include "Config.cpp"
#include "HDF5read.cpp"
/**
 * Item
 * 用于记录每个向量的信息
 * vector:向量
 * edge : 该向量的KNN/NSW图
 * 提供重载的运算符 : [] -(欧氏距离的平方)
*/
template <typename T>
class Item {
public: 
    Item(){}
    Item(int lenth) : vectors(lenth){}
    ~Item(){}
    std::vector<T> vectors; // 向量
    std::vector<int> edge; // 边
    T &operator[](int i) { return vectors[i]; }
    // cluster
    int clusterId; // 聚类的id

    // 重载运算符，用于计算两个向量的距离(仅计算欧式距离的平方)
    double operator-(Item<T> &item) {
        double sum = 0;
        // openmp simd
        #pragma omp simd
        for (int i = 0; i < vectors.size(); i++) {
            sum += (vectors[i] - item.vectors[i]) * (vectors[i] - item.vectors[i]);
        }
        return sum;
    }
    double operator*(Item<T> &item) {
        double sum = 0;
        for (int i = 0; i < vectors.size(); i++) {
            sum += vectors[i] * item.vectors[i];
        }
        return sum;
    }
    double length() {
        double sum = 0;
        for (int i = 0; i < vectors.size(); i++) {
            sum += vectors[i] * vectors[i];
        }
        return sqrt(sum);
    }
    inline void print() {
        for (auto & item : vectors) {
            std::cout << item << " ";
        }
        std::cout << std::endl;
    }
};

template <typename T>
class DataSet{
public:
    DataSet() {
        dimension = -1;
    }
    ~DataSet() {}
    int dimension; // 向量维度
    std::vector<Item<T>> baseData; // 向量库
    std::vector<Item<int>> ansData; // 答案
    std::vector<Item<T>> queryData; // 查询
};

template <typename T>
class SIFTDataSet: public DataSet<T>{
public:
    SIFTDataSet(std::string baseFile, std::string queryFile, std::string ansFile) : DataSet<T>(), baseFileName(baseFile), queryFileName(queryFile), ansFileName(ansFile) {
        readBaseData();
        readQueryData();
        readAnsData();
    }
    ~SIFTDataSet(){}
private:
    std::string baseFileName;
    std::string queryFileName;
    std::string ansFileName;
    // 读入向量库
    void readBaseData() {
        FILE *fp = fopen(baseFileName.c_str(), "rb");
        if (fp == NULL) {
            std::cout << "open file error" << std::endl;
            return;
        }
        int num = 0;
        int length;
        while (fread(&length, sizeof(int), 1, fp)){
            this->baseData.emplace_back(length);
            fread(this->baseData.back().vectors.data(),  sizeof(T), length, fp);
            this->baseData.back().vectors.shrink_to_fit();
            #if (DatabaseSelect > 2 && DatabaseSelect < 7)
                if (++num == maxbaseNum) break;
            #endif
        }
        this->baseData.shrink_to_fit();
        if (this->dimension == -1) {
            this->dimension = length;
        } else if (this->dimension != length) {
            std::cout << "dimension error" << std::endl;
            return;
        }
        fclose(fp);
    }
    // 读入查询数据
    void readQueryData() {
        FILE *fp = fopen(queryFileName.c_str(), "rb");
        if (fp == NULL) {
            std::cout << "open file error" << std::endl;
            return;
        }
        int length;
        while (fread(&length, sizeof(int), 1, fp)){
            this->queryData.emplace_back(length);
            fread(this->queryData.back().vectors.data(),  sizeof(T), length, fp);
        }
        fclose(fp);
    }
    // 读入答案
    void readAnsData() {
        FILE *fp = fopen(ansFileName.c_str(), "rb");
        if (fp == NULL) {
            std::cout << "open file error" << std::endl;
            return;
        }
        int length;
        while (fread(&length, sizeof(int), 1, fp)){
            this->ansData.emplace_back(length);
            fread(this->ansData.back().vectors.data(),  sizeof(int), length, fp);
        }
        fclose(fp);
    }
};

template <typename T>
class HDF5DataSet: public DataSet<T>{
public:
    HDF5DataSet(std::string baseFile) : DataSet<T>(), baseFileName(baseFile) {
        readBaseData();
        readQueryData();
        readAnsData();
    }
    ~HDF5DataSet(){}
private:
    std::string baseFileName;
    // 读入向量库
    void readBaseData() {
        auto base = read_dataset<T>(baseFileName, "train");
        for (auto & item : base) {
            this->baseData.emplace_back(item.size());
            this->baseData.back().vectors = item;
        }
        this->baseData.shrink_to_fit();
        this->dimension = D;
    }
    // 读入查询数据
    void readQueryData() {
        auto test = read_dataset<T>(baseFileName, "test");
        for (auto & item : test) {
            this->queryData.emplace_back(item.size());
            this->queryData.back().vectors = item;
        }
        this->queryData.shrink_to_fit();
    }
    // 读入答案
    void readAnsData() {
        auto neighbors = read_dataset<int>(baseFileName, "neighbors");
        for (auto & item : neighbors) {
            this->ansData.emplace_back(item.size());
            this->ansData.back().vectors = item;
        }
        this->ansData.shrink_to_fit();
    }
};

#endif