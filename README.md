hip exo


mkdir build
cd build
cmake ..
make -j16

./render/render ../data/metadata.txt ../nn/new_new4/5.pt ../nn/new_new4/5_human.pt
