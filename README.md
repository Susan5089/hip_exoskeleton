hip exo

```bash
source  ~/py_env/MASS/bin/activate
mkdir build
cd build
cmake ..
make -j16
```
```bash
./render/render ../data/metadata.txt ../nn/new_new4/5.pt ../nn/new_new4/5_human.pt
```