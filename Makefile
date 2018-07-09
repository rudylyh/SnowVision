INCLUDE_FLAGS = -I/usr/include/pcl-1.8/ -I/usr/include/eigen3/ -I/usr/include/vtk-6.2/

LIB_FLAGS = -L/usr/pcl/lib/

LD_FLAGS = -lpcl_kdtree -lpcl_segmentation -lopencv_highgui -lopencv_core

all: sherd2depth.cpp
	g++ sherd2depth.cpp -fPIC -shared -o sherd2depth.so ${INCLUDE_FLAGS} ${LIB_FLAGS} ${LD_FLAGS}

clean:
	rm sherd2depth
