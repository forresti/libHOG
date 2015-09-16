#ifndef __LIBHOG_H__
#define __LIBHOG_H__
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <opencv2/opencv.hpp> //only used for file I/O
#include <SimpleImg.hpp>
using namespace std;

//TODO: define class for data to return... like a Caffe blob.

class libHOG{

  public:
    libHOG(int nLevels, int interval, bool use_voc5_impl);
    libHOG(int nLevels, int interval);
    libHOG(); //empty constructor w/ default settings
    ~libHOG();
    void compute_pyramid(cv::Mat img); //output is in this->hogBuffer_blocks.
    void reset_timers();

    //timer variables:
    double img_pyra_time;
    double alloc_time;
    double grad_time;
    double hist_time;
    double norm_time;

    //user-viewable output.
    // TODO: free & reallocate these at the BEGINNING of the next compute_pyramid() call.
    //  (of course, only free if it's not null....) 
    vector< float* > hogBuffer_blocks; //this is the output HOG pyramid <TODO: rename to 'hogPyra'>
    vector< int > hogHeight;
    vector< int > hogWidth;

  private:
    int get_sbin_for_scale(int scaleIdx, int interval);
    void gen_img_pyra(cv::Mat img_Mat);
    void allocate_bufs();
    //void free_img_pyra();
    void free_bufs();

    //buffers
    vector< SimpleImg<uint8_t>* > imgPyramid;
    vector< SimpleImg<uint8_t>* > ori; //(img.height, img.stride, 1); //out img has just 1 channel
    vector< SimpleImg<int16_t>* > mag; //(img.height, img.stride, 1); //out img has just 1 channel
    vector< float* > hogBuffer; //TODO: rename to hogCells
    vector< float* > normImg;
    float sc; //for calculating pyramid scales

    //user-selected in constructor:
    int nLevels;
    int interval;
    bool use_voc5_impl;
    //TODO: handle user-selected sbin?
    //TODO: handle user-selected padding?
    //TODO: user-selected ffld or voc5 compatibility
    //TODO: user-selected L1 or L2 norm


    //for freeing imgPyramid, ori, and mag
    template<class t>
    void delete_vec_of_obj(vector<t> &vec){
        for(int s=0; s<vec.size(); s++){
            delete vec[s];
        }
        vec.resize(0);
    }

    //it is safe to call free_vec_of_ptr() even if vec[:] isn't allocated yet
    template<class t>
    void free_vec_of_ptr(vector<t> &vec){
        for(int s=0; s<vec.size(); s++){
            if(vec[s] != NULL){
                free(vec[s]);
            }
        }
        vec.resize(0);
    }
};

#endif

