export NVCCFLAGS=-arch=sm_70
cd raymarching
pip install .
cd ..

cd gridencoder
pip install .
cd ..

cd shencoder
pip install .
cd .. 

cd ffmlp
pip install .
cd ..