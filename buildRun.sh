# -DZERO -DCLUSTER 表示NSW 多查询
# -DZERO 表示NSW 单查询
# null 表示 HNSW 单查询
if g++ main.cpp -o main -g -std=c++17 -O3 -DDatabaseSelect=1 -DTHREAD_CONFIG=64 -fopenmp -pthread -w \
    -I /usr/include/hdf5/serial \
    -I ./res/hdf5/HighFive/include/ \
    -lhdf5_cpp -lhdf5 -L /usr/lib/x86_64-linux-gnu/hdf5/serial ; then
    gdb ./main 0.05 3 3 3
    if [ -f "main" ]; then
        rm main
    fi
    if [ -d "main.dSYM" ]; then
        rm -r ./main.dSYM
    fi
else 
    echo "some error occured"
fi

