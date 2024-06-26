#ifndef __config_CPP__
#define __config_CPP__
#include <cstdio>
#include <vector>
#include <cmath>
#include <iostream>
#include <string>
#include <fstream>
#include <cstdlib>
#include <sstream>
#include <thread>
#include <chrono>
#include <mutex>

#define uchar unsigned char
// 查询配置
int CREATEGRAPH = 0;
std::mutex *mtx;
#define THREAD_CONFIG 1

// 选择数据集
#define SIFT1M 1
#define FashionMNIST 2
#define MNIST 3
#define GIST 4
#define LASTFM 5
#define NYTIMES 6
#define GLOVE25 7
#define GLOVE100 8

// 数据集配置
/**
 * D: 每个向量的维度
 * M: 每个向量的最大边数
 * K: 每个向量的查询邻居数
*/
#if DatabaseSelect == SIFT1M
    const int K = 100;
    const int D = 128;
    const int maxbaseNum = 1000000;
    #define FILETYPE float
    std::string baseFileName = "./dataset/sift1M/sift_base.fvecs";
    std::string queryFileName = "./dataset/sift1M/sift_query.fvecs";
    std::string ansFileName = "./dataset/sift1M/sift_groundtruth.ivecs";
    bool HDF5 = false;
#endif

#if DatabaseSelect == FashionMNIST
    const int K = 100;
    const int D = 784;
    #define FILETYPE float
    int maxbaseNum = 60000;
    std::string baseFileName = "./dataset/hdf5/fashion-mnist-784-euclidean.hdf5";
    bool HDF5 = true;
#endif

#if DatabaseSelect == MNIST
    const int K = 100;
    const int D = 784;
    #define FILETYPE float
    int maxbaseNum = 60000;
    std::string baseFileName = "./dataset/hdf5/mnist-784-euclidean.hdf5";
    bool HDF5 = true;
#endif

#if DatabaseSelect == GIST
    const int K = 100;
    const int D = 960;
    #define FILETYPE float
    int maxbaseNum = 1000000;
    std::string baseFileName = "./dataset/hdf5/gist-960-euclidean.hdf5";
    bool HDF5 = true;
#endif

#if DatabaseSelect == LASTFM
    const int K = 100;
    const int D = 65;
    #define FILETYPE float
    int maxbaseNum = 292385;
    std::string baseFileName = "./dataset/hdf5/lastfm-64-dot.hdf5";
    bool HDF5 = true;
#endif

#if DatabaseSelect == NYTIMES
    const int K = 100;
    const int D = 256;
    #define FILETYPE float
    int maxbaseNum = 290000;
    std::string baseFileName = "./dataset/hdf5/nytimes-256-angular.hdf5";
    bool HDF5 = true;
#endif

#if DatabaseSelect == GLOVE25
    const int K = 100;
    const int D = 25;
    #define FILETYPE float
    int maxbaseNum = 1183514;
    std::string baseFileName = "./dataset/hdf5/glove-25-angular.hdf5";
    bool HDF5 = true;
#endif 

#if DatabaseSelect == GLOVE100
    const int K = 100;
    const int D = 100;
    #define FILETYPE float
    int maxbaseNum = 1183514;
    std::string baseFileName = "./dataset/hdf5/glove-100-angular.hdf5";
    bool HDF5 = true;
#endif

#endif