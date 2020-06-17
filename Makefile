INCLUDE_FLAGS = -I/usr/include/pcl-1.8/ -I/usr/include/eigen3/ -I/usr/include/vtk-6.3/

LD_FLAGS = -lpcl_common -lpcl_kdtree -lpcl_segmentation -lpcl_visualization -lvtkRenderingCore-6.3 -lvtkCommonDataModel-6.3 -lvtkCommonMath-6.3 -lvtkCommonCore-6.3 -lboost_system -lboost_thread -lopencv_highgui -lopencv_core

all: xyz_proc.cpp
	g++ xyz_proc.cpp -fPIC -shared -o libxyz_proc.so ${INCLUDE_FLAGS} ${LD_FLAGS} `pkg-config --cflags --libs opencv`

clean:
	rm libxyz_proc.so
