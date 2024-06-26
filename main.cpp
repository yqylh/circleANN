#include "lib/DataSet.cpp"
#include "lib/Config.cpp"
#include "lib/Query.cpp"
int main(){
    std::ios::sync_with_stdio(false);
    std::cin.tie(0);
    DataSet<FILETYPE> *ds;
    if (HDF5) ds = static_cast<DataSet<FILETYPE> *>(new HDF5DataSet<FILETYPE>(baseFileName));
    else ds = static_cast<DataSet<FILETYPE> *>(new SIFTDataSet<FILETYPE>(baseFileName, queryFileName, ansFileName));
    queryAnn(ds, 100);
    delete ds;
    return 0;
}