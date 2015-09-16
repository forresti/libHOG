#ifndef __HELPERS_H__
#define __HELPERS_H__
#include <opencv2/opencv.hpp>
#include <math.h>
#include <sys/time.h>
#include <stdint.h> //for uintptr_t
#include <immintrin.h> //256-bit AVX
#include <xmmintrin.h> //for other SSE-like stuff
#include <string>

using namespace std;
using namespace cv;

//macros from voc-release5 fconvsse.cc
#define IS_ALIGNED(ptr) ((((uintptr_t)(ptr)) & 0xF) == 0) 

#if !defined(__APPLE__)
#include <malloc.h>
#define malloc_aligned(a,b) memalign(a,b)
#else
#define malloc_aligned(a,b) malloc(b)
#endif

void print_epi16(__m128i vec_sse, string vec_name);
double read_timer();
//std::string forrestGetImgType(int imgTypeInt);
int compute_stride(int width, int size_per_element, int ALIGN_IN_BYTES);
float* allocate_hist(int in_imgHeight, int in_imgWidth, int sbin, int &out_hogHeight, int &out_hogWidth);
int16_t* allocate_hist_16bit(int in_imgHeight, int in_imgWidth, int sbin,
                   int &out_hogHeight, int &out_hogWidth);

Mat downsampleWithOpenCV(Mat img, double scale);
#endif

