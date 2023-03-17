CFLAGS = -std=c++17 `pkg-config --cflags --libs opencv4` -fopenmp -lfftw3f -O3
PROGRAM = Excuse.cpp
OBJECT = a.out
NVCCFLAGS = -L/usr/local/cuda/lib64 -I/usr/local/cuda/include -lcusolver -lcudart -lcublas -lnppc -lnppial -lnppicc -lnppidei -lnppif -lnppig -lnppim -lnppist -lnppisu -lnppitc -lnpps -Xcompiler -fopenmp -O3 -std=c++17 `pkg-config --cflags --libs opencv4`

build:
	/usr/local/cuda-12.0/bin/nvcc EllipseFit.cu $(NVCCFLAGS) -o $(OBJECT)
elipseFit:
	/usr/local/cuda-12.0/bin/nvcc EllipseFit.cu $(NVCCFLAGS) -o $(OBJECT) && ./$(OBJECT)

