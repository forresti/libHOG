# libHOG

initial configuration:
0. install OpenCV
0. point Makefile to OpenCV
0. mkdir tests_output; mkdir build;
0. make

basic example on how to use libHOG in C++:
```
//computing HOG features on my_image.jpg
cv::Mat img = cv::imread("my_image.jpg");
libHOG lHOG();
lHOG.compute_pyramid(img);
//HOG pyramid is located here: lHOG.hogBuffer_blocks[scale]
//pyramid dimensions: lHOG.hogHeight[scale], lHOG.hogWidth[scale]
//number of scales: lHOG.hogHeight.size()

```

see src/libHOG_cli.cpp for an additional example of how to use libHOG
 

linking your project with libHOG:
 include libHOG.h, and dynamically link with libHOG.so 

planned functionality:
0. Matlab API (similar to the interface in voc-release5 featpyramid.m)



