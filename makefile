CXX=/usr/local/cuda-10.2/bin/nvcc
NVCCFLAGS= -O3 -lglfw -lGL -lGLEW  -lm -lstdc++ --expt-relaxed-constexpr -arch sm_30 
DEPS = bvh.hpp Ray.hpp Shader.hpp stb_image_write.h SceneObjects.hpp ./glm/glm/glm.hpp ./glm/glm/gtx/io.hpp isect.hpp objloader.hpp

all: $(DEPS)
	$(CXX) $(NVCCFLAGS) main.cu bvh.cu Ray.cu Shader.cpp objloader.cpp -o ray

clean:
	rm -f *.o ray
 




