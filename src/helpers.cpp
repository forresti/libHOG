#include "helpers.h"
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

double read_timer(){
    struct timeval start;
    gettimeofday( &start, NULL );
    return (double)((start.tv_sec) + 1.0e-6 * (start.tv_usec)) * 1000; //in milliseconds
}

void print_epi16(__m128i vec_sse, string vec_name){
    int16_t vec_scalar[8];
    _mm_store_si128((__m128i*)vec_scalar, vec_sse);

    printf("    %s: ", vec_name.c_str());
    for(int i=0; i<8; i++){
        printf("%d, ", vec_scalar[i]);
    }
    printf("\n");
}

//size_per_element = sizeof(float), sizeof(char), or whatever data type you're using.
int compute_stride(int width, int size_per_element, int ALIGN_IN_BYTES){
    //thanks: http://stackoverflow.com/questions/2403631

    //int ALIGN_IN_BYTES=32; //for AVX
    int stride = (width*size_per_element + (ALIGN_IN_BYTES - (width*size_per_element)%ALIGN_IN_BYTES))/size_per_element;
    return stride;
}

//TODO: put this into a class (like PgHogContainer or streamHog), once I decide what data types to use
//@output-by-ref out_hogWidth out_hogHeight
//@return hogWidth = memory aligned vector for storing HOG histogram 
float* allocate_hist(int in_imgHeight, int in_imgWidth, int sbin,
                   int &out_hogHeight, int &out_hogWidth){

    //out_hogHeight = round(in_imgHeight/sbin);
    //out_hogWidth = round(in_imgWidth/sbin);
    out_hogHeight = (int)round((double)in_imgHeight/(double)sbin);
    out_hogWidth = (int)round((double)in_imgWidth/(double)sbin);
    const int hogDepth = 32;

    float* hogBuffer = (float*)malloc_aligned(32, out_hogWidth * out_hogHeight * hogDepth * sizeof(float));
    memset(hogBuffer, 0, out_hogWidth * out_hogHeight * hogDepth * sizeof(float));
    return hogBuffer;
}

//TODO: make allocate_hist() into a template that can work on any data type.
int16_t* allocate_hist_16bit(int in_imgHeight, int in_imgWidth, int sbin,
                   int &out_hogHeight, int &out_hogWidth){

    //out_hogHeight = round(in_imgHeight/sbin);
    //out_hogWidth = round(in_imgWidth/sbin);
    out_hogHeight = (int)round((double)in_imgHeight/(double)sbin);
    out_hogWidth = (int)round((double)in_imgWidth/(double)sbin);
    const int hogDepth = 32;

    int16_t* hogBuffer = (int16_t*)malloc_aligned(32, out_hogWidth * out_hogHeight * hogDepth * sizeof(int16_t));
    memset(hogBuffer, 0, out_hogWidth * out_hogHeight * hogDepth * sizeof(int16_t));
    return hogBuffer;
}


//use OpenCV's bilinear filter downsampling
Mat downsampleWithOpenCV(Mat img, double scale){
    int inWidth = img.cols;
    int inHeight = img.rows;
    assert(img.type() == CV_8UC3);
    int nChannels = 3;

    int outWidth = round(inWidth * scale);
    int outHeight = round(inHeight * scale);
    Mat outImg(outHeight, outWidth, CV_8UC3); //col-major for OpenCV 
    Size outSize = outImg.size();

    cv::resize(img,
               outImg,
               outSize,
               0, //scaleX -- default = outSize.width / img.cols
               0, //scaleY -- default = outSize.height / img.rows
               INTER_LINEAR /* use bilinear interpolation */);

    return outImg;
}


