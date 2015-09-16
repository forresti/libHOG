#BUILD_DIR = ./build #FIXME: send the .o files to ./build
#BUILD_DIR = ./
INC_OBJS = helpers.o libHOG.o libHOG_kernels.o voc5_reference_kernels.o
EXE_OBJS = libHOG_cli.o test_libHOG.o test_SimpleImg.o 
OBJS = $(INC_OBJS) $(EXE_OBJS)
EXE_LIST = libHOG_cli test_libHOG test_SimpleImg

#on sandy bridge, -march=native gives me -mavx -msse2avx
CC = g++
CCOPTS = -c -O3 -fPIC `pkg-config opencv --cflags` -msse -ffast-math -fopenmp -mssse3 -I./include
#CCOPTS += -mavx -msse2avx
LINK = g++
LINKOPTS = `pkg-config opencv --libs` 
#LINKOPTS += -lgomp #FIXME: doesn't seem to work on Mac. (may be required for OpenMP)

#link
#TODO: create the following binaries, in ./build:
# [done] libHOG.so
# [done] libHOG_cli (executable)
# [done] test_libHOG (executable)
# [TODO] test_SimpleImg (executable)

all: $(EXE_LIST) libHOG.so

libHOG_cli : $(INC_OBJS) libHOG_cli.o
	$(LINK) -o libHOG_cli $(INC_OBJS) libHOG_cli.o $(LINKOPTS)

test_libHOG : $(INC_OBJS) test_libHOG.o
	$(LINK) -o test_libHOG $(INC_OBJS) test_libHOG.o $(LINKOPTS)

test_SimpleImg : $(INC_OBJS) test_SimpleImg.o
	$(LINK) -o test_SimpleImg $(INC_OBJS) test_SimpleImg.o $(LINKOPTS)

libHOG.so : $(INC_OBJS)
	$(LINK) -shared -o libHOG.so $(INC_OBJS) $(LINKOPTS)

#  TODO: do I really have to put 'src/' or 'include/' before each file? (-I./include didn't help)
libHOG.o : src/libHOG.cpp include/libHOG.h include/libHOG_kernels.h include/voc5_reference_kernels.h include/helpers.h include/SimpleImg.hpp 
	$(CC) $(CCOPTS) src/libHOG.cpp #-o build/libHOG.o #TODO: custom build dir

libHOG_cli.o : src/libHOG_cli.cpp include/helpers.h include/SimpleImg.hpp include/libHOG.h 
	$(CC) $(CCOPTS) src/libHOG_cli.cpp

helpers.o : src/helpers.cpp include/helpers.h 
	$(CC) $(CCOPTS) src/helpers.cpp

libHOG_kernels.o : src/libHOG_kernels.cpp include/libHOG_kernels.h include/helpers.h include/SimpleImg.hpp
	$(CC) $(CCOPTS) src/libHOG_kernels.cpp

voc5_reference_kernels.o : src/voc5_reference_kernels.cpp include/voc5_reference_kernels.h include/helpers.h include/SimpleImg.hpp
	$(CC) $(CCOPTS) src/voc5_reference_kernels.cpp

test_libHOG.o : tests/test_libHOG.cpp include/test_libHOG.h include/helpers.h include/SimpleImg.hpp include/libHOG.h include/libHOG_kernels.h include/voc5_reference_kernels.h 
	$(CC) $(CCOPTS) tests/test_libHOG.cpp

test_SimpleImg.o : tests/test_SimpleImg.cpp include/helpers.h include/SimpleImg.hpp 
	$(CC) $(CCOPTS) tests/test_SimpleImg.cpp


clean : 
	rm -f *.o *.so $(EXE_LIST) 2>/dev/null

