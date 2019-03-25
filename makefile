CC=g++
CXX=/usr/local/cuda-10.1/bin/nvcc
CXXFLAGS= -O3 -lm -lstdc++
NVCCFLAGS= -O3 -lm -lstdc++ --expt-relaxed-constexpr -arch sm_30 
DEPS = bvh.hpp Ray.hpp stb_image_write.h SceneObjects.hpp ./glm/glm/glm.hpp ./glm/glm/gtx/io.hpp isect.hpp
OBJ = bvh.o 

#all: ray

#Ray.cu.o: Ray.cu Ray.hpp
#	$(CXX) -c -O3 -arch=sm_37 -std=c++11 -cudart=shared -rdc=true Ray.cu

#%.o: %.c $(DEPS)
#	$(CXX) -c -o $@ $< $(CXXFLAGS)

#ray: $(OBJ)
#	$(CXX) -c -o $@ $^ Ray.cu.o $(CXXFLAGS)

all: $(DEPS)
	$(CXX) $(NVCCFLAGS) main.cu bvh.cu Ray.cu -o ray
 




