# libHOG

## About

libHOG is a library that computes [Histogram of Oriented Gradient](https://en.wikipedia.org/wiki/Histogram_of_oriented_gradients) features. The benefit of libHOG over other HOG implementations is speed. On an Intel i7-3930k 6-core CPU, we measured the following results:

|                 | Frame Rate  | Speedup | Energy per frame (Joules)|
| :------------- |:-------------:| :-----:| :-----:|
| [voc-release5][1] | 2.44 fps | 1.0x | 57.4 J/frame |
| [Dollar][2]       | 5.88 fps | 2.4x | 26.4 J/frame |
| [FFLD-serial][3] | 4.59 fps | 1.9x | 29.9 J/frame |
| [FFLD-OpenMP][3] | 19.6 fps | 8.0x | 9.44 J/frame | 
| libHOG (this codebase) | **58.8 fps** | **24.0x** | **3.15 J/frame** |

The output of libHOG is numerically equivalent to the HOG features in the Deformable Model Parts model ([voc-release5][1]) codebase, with a 24x speedup. For further speedups (over 70fps), we also offer the option to use L1-norm (instead of the traditional L2-norm) when calculating the gradients. If you find libHOG useful, please consider citing the [libHOG paper](http://forrestiandola.com/publications/libHOG_ITSC15.pdf):

    @inproceedings{libHOG,
        Author = {Forrest N. Iandola and Matthew W. Moskewicz and Kurt Keutzer},
        Title = {libHOG: Energy-Efficient Histogram of Oriented Gradient Computation},
        Booktitle = {ITSC},
        Year = {2015}
    }
## Getting Started

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

[1]: https://github.com/rbgirshick/voc-dpm
[2]: https://pdollar.github.io/toolbox/channels/fhog.html
[3]: https://github.com/fanxu/ffld
