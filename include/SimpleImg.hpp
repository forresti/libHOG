#ifndef __SIMPLEIMG_H__
#define __SIMPLEIMG_H__
#include "helpers.h"
#include <opencv2/opencv.hpp> //only used for file I/O
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
using namespace std;

template<class pixel_t>
class SimpleImg{
  public:
    pixel_t* data;
    int width;
    int stride; //width+padding. note that this is row-major.
    int height;
    int n_channels;
    static const int ALIGN_IN_BYTES=32;

    SimpleImg(int in_height, int in_width, int in_n_channels){
        height = in_height;
        width = in_width;
        //stride = compute_stride(width, sizeof(pixel_t), ALIGN_IN_BYTES); //defined in helpers.cpp -- uses ALIGN_IN_BYTES=32 by default.
        stride = compute_stride(width, sizeof(uint8_t), ALIGN_IN_BYTES); 
        n_channels = in_n_channels;

        data = (pixel_t*)malloc_aligned(ALIGN_IN_BYTES, height * stride * n_channels * sizeof(pixel_t));
        memset(this->data, 0, height * stride * n_channels * sizeof(pixel_t));
    }

    //constructor 'from file'
    SimpleImg(string fname){
        cv::Mat img = cv::imread(fname); //.c_str()?
        Mat_to_SimpleImg_ctor(img); 
    }

    //constructor 'from cv::Mat'
    SimpleImg(cv::Mat img){
        Mat_to_SimpleImg_ctor(img);
    }

    //cv::Mat -> instantiate myself (a SimpleImg) -- TODO: make this private 
    void Mat_to_SimpleImg_ctor(cv::Mat img)
    {
        height = img.rows;
        width = img.cols;
        //stride = compute_stride(width, sizeof(pixel_t), ALIGN_IN_BYTES); //defined in helpers.cpp
        stride = compute_stride(width, sizeof(uint8_t), ALIGN_IN_BYTES);
        //printf("    stride = %d \n", stride);
        n_channels = 3;
        assert(img.type() == CV_8UC3); //require that input img is 3-channel, uchar words. 
        assert(sizeof(pixel_t) == 1); //uchar single-byte words
        this->data = (pixel_t*)malloc_aligned(32, height * stride * n_channels * sizeof(pixel_t)); //factor of 3 for 3 ch
        memset(this->data, 0, height * stride * n_channels * sizeof(pixel_t));

        //TODO: consider OpenCV copyMakeBorder() for padding http://docs.opencv.org/modules/imgproc/doc/filtering.html?#copymakeborder

        double start_time = read_timer();

        //this might be slow ... don't worry, it's just a test:
        //#pragma omp parallel for
        for(int y=0; y<height; y++){
            for(int x=0; x<width; x++){
                for(int channel = 0; channel < 3; channel++){
                    //this->data[y*stride*3 + x*3 + channel] = img.at<cv::Vec3b>(y,x)[channel];  //channels as inner dimension
                    this->data[y*stride + x + channel*stride*height] = img.at<cv::Vec3b>(y,x)[channel]; //channels as outer dimension 
                    //this->data[y*stride + x + channel*stride*height] = img.data[y*width*3 + x*width + channel];
                }
            }
        }

        double response_time = read_timer() - start_time;
        //printf("transposed img in %f ms \n", response_time);
    }

    //write the current image object's data out to file.
    void simple_imwrite(string fname){
        //assuming we're using uchar images.
        assert(n_channels == 1 || n_channels == 3);
        cv::Mat* out_img;
        assert(sizeof(pixel_t) == 1 || sizeof(pixel_t) == 2); //require 1-byte (uint8_t) or 2-byte (int16_t) pixel type
        //printf("sizeof(pixel_t) = %d \n", sizeof(pixel_t));
       
        if(sizeof(pixel_t) == 1){ //unsigned 8-bit 
            if(this->n_channels == 1){
                out_img = new cv::Mat(this->height, this->stride, CV_8UC1, this->data); //cv::Mat(int rows, int cols, int type, char* preAllocatedPointerToData)
            }
            else if(this->n_channels == 3){
                out_img = new cv::Mat(this->height, this->stride, CV_8UC3, this->data); 
            }
        }
        else if(sizeof(pixel_t) == 2){ //signed 16-bit
            if(this->n_channels == 1){
                out_img = new cv::Mat(this->height, this->stride, CV_16SC1, this->data); //cv::Mat(int rows, int cols, int type, char* preAllocatedPointerToData)
            }
            else if(this->n_channels == 3){
                out_img = new cv::Mat(this->height, this->stride, CV_16SC3, this->data);
            }
        }

        cv::imwrite(fname, *out_img);

        //TODO: delete out_img
    }

    void simple_csvwrite(string fname){
        assert(n_channels == 1); //might add 3-channel support later
        assert(sizeof(pixel_t) == 1); //need unsigned char pixels for this to work (because of typecasting below)

        FILE* pFile = fopen (fname.c_str(),"w");
        for(int row=0; row < height; row++){
            for(int col=0; col < (width - 1); col++){
                fprintf(pFile, "%u,", data[row*stride + col]); //%u = unsigned
            }
            fprintf(pFile, "%u\n", data[row*stride + (width - 1)]);
        }        
        fclose (pFile);
    }

    ~SimpleImg(){
        free(data);
    }
};
#endif
