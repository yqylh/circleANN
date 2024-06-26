#include "lib/DataSet.cpp"
#include "lib/Config.cpp"
#include "lib/Query.cpp"
int main(){
    std::ios::sync_with_stdio(false);
    std::cin.tie(0);
    HDF5DataSet<FILETYPE> *ds;
    if (HDF5) ds = new HDF5DataSet<FILETYPE>(baseFileName);
    else ds = new SIFTDataSet<FILETYPE>(baseFileName, queryFileName, ansFileName);
    DataSet<FILETYPE> *dataSet = ds;
    queryAnn(dataSet);
    delete ds;
    return 0;
}
