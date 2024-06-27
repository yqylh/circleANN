#include "lib/Circle.cpp"

/**
我在构图的时候，可以把图分成 N/D 个多层凸包（维护每个维度的 list，结合凸包的特点），然后我画出来了一个多层高维球体。这个球体的每一层一定是子层的凸包。这样我们就对高维空间的点进行了划分
我们可以在每个球面上建立 NSW，在球面之间建立少量的边的联系（也就是在每个球面上进行 NSW 搜索）。我们一共执行了 N/D个 D 大小的 NSW。2N 个 D 大小的 NSW。
对于每一层球面进行随机采样选点，表示对球面的随机采样。例如采样 C 个，每一层采样率为C/D, 整体采样了 N * C / D，表示对每个球面随机划分成了 C 块，在每一块里随机选择了一个中心（可以用聚类替代）

对于这S =  N * C / D 个元素可以再次建立球体（因为凸包是不准确的），重复上面的过程，选出来了 S * C / D个元素。建立 i 次球体，我们的索引数量为 N * (C / D) ^ i。

在最高一层，我们对于N * (C / D) ^ i个元素建立NSW，从上一个球体中最内部的一个元素开始搜索。

我们可以理解为我们每层会较为准确的将图分为了S=N/D*C 个区域，这S个表示对于空间的计算距离的均匀采样。每一次我们都精准的按照 C/D的比例去降低的数量。

相比之下我们的优点：

1. 稳定对空间进行 N/D*C的采样
2. 稳定均匀的按照 log 级别的降低搜索难度
3. 因为邻居足够准确，所以邻居数量可以降低，极限情况可以降低到邻居=4
4. 在搜索时保持为对空间的均匀采样进行遍历
5. 搜索为空间中心为起点，更为精准，方向指向性比较强
6. 好并行，构建多个 NSW 时可以无损并行
7. 对于多中心的稠密集群可以保留内部关系和相邻关系。

或者说利用紧密子图 和 聚类的概念来构造hnsw


我突然在想，我们是否可以将 N 个点，构造成 M*L 个序列，每个序列稳定 M~2M 个元素

序列之间越远，边越多；越近（实际上更靠近），边越少

序列与序列之间，应该可以可以考虑这个关系，越远的🔗越多，越近的，连接越少。可以随机也可以选定一些点当做主 Key 衡量距离。

最后就有点像神经网络了，越靠后越稠密（一定）

idea：
1. 按照凸包，分成 C= N /( D*2 )组，每组 D*2 个元素，表示一层
2. 每组元素相当于一个圆。在元素内部，计算 TOP p个邻居，建立边
3. 每层c内的元素，向外扩展，连接 log_3^C层，也就是c+3层，c+3*3 层，第一层增加 p / log_3^3个邻居，第二层增加p / log_3^3^2= p/2个邻居。
4. 将上述元素每三层保留一份，也就是概率为1/3，然后构建一个大小只有1/3的图。继续执行，直到图的大小不足D*2, 构建一个 ANN 索引

对于 1M 数据集，128 维。p = D* 2 / 8 = 32个。指数概率为 3
第一维凸包：
10^6个点，每一层 256 个，一共3900层。每个节点内部邻居 32 个，第一层邻居 32*2 个。一共七层，由于对称，所以：
32 * 1的有两个  
32 / 3的有两个
32/9的有两个
32/81的又一个
一共 32 + 32*2 + 32/3*2 + 32/9*2 + 1 = 125 个

如果p = 5%，节点就只有10+10*2+3 了
一共会建立33*1e6条边。
第二维：
1e6 个点，由于三层留一个，所以还剩下 3e5 个点，一共 1300 层。
33*1e6/3
一共有 八层
也就是(33+11+4+1.3+0.45+0.15...=50*1e6 条边)

可以对比同样 m 下的Recall差异
设定参数：
层内邻居数量概率p, 基础邻居数=D*2/p
向外扩展的速率v1，会对外扩展到 c + 1, c + 1 + 1*v1, ..., c + 1 + n*v1。以及反过来
基础邻居递减速率v2，每扩展一层，邻居数量/v2，p, p/v2, p/v2^2...最小为 1。如果设定为 -1，则按照p/1 p/2 p/3 p/4的顺序缩减
上层衰减率t, 每隔 t 层保留一层内的元素，重复执行邻居边的策略（更快地到达内部）也可以每层随机筛选t/层数的元素
*/
void init() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(0);
}
/**
 * input value
 * LAYER_P: 层内邻居数量概率 double
 * LAYER_ADD: 向外扩展的速率 int
 * LAYER_DEC: 每层邻居递减速率 int
 * LAYER_T: 上层衰减率 int
 * input format : main LAYER_P LAYER_ADD LAYER_DEC LAYER_T
*/
void parseArgs(int argc, char *argv[]) {
    init();
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " LAYER_P LAYER_ADD LAYER_DEC LAYER_T" << std::endl;
        exit(1);
    }
    LAYER_P = atof(argv[1]);
    LAYER_ADD = atof(argv[2]);
    LAYER_DEC = atof(argv[3]);
    LAYER_T = atof(argv[4]);
} 

int main(int argc, char *argv[]) {
    parseArgs(argc, argv);
    DataSet<FILETYPE> *ds;
    if (HDF5) ds = static_cast<DataSet<FILETYPE> *>(new HDF5DataSet<FILETYPE>(baseFileName));
    else ds = static_cast<DataSet<FILETYPE> *>(new SIFTDataSet<FILETYPE>(baseFileName, queryFileName, ansFileName));
    Circle circle(ds, K);
    circle.solveEdge();
    circle.queryAnn();
    delete ds;
    return 0;
}