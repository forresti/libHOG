#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cassert>
#include <xmmintrin.h>
#include <pmmintrin.h> //for _mm_hadd_pd()
#include <opencv2/opencv.hpp> //only used for file I/O

#include "SimpleImg.hpp"
#include "libHOG.h"
#include "helpers.h"
#include "libHOG_kernels.h"
#include "voc5_reference_kernels.h"
using namespace std;

//@param use_voc5_impl: if true, voc5-based impl. else, use libHOG native impl
libHOG::libHOG(int nLevels_, int interval_, bool use_voc5_impl_){
    use_voc5_impl = use_voc5_impl_;
    nLevels = nLevels_;
    interval = interval_;
    reset_timers();
}

//TODO: take sbin, padding, <ffld or voc5>, <L1 or L2>
libHOG::libHOG(int nLevels_, int interval_){
    use_voc5_impl = false; //use libHOG's native impl
    nLevels = nLevels_;
    interval = interval_;
    reset_timers(); 
}

//empty constructor w/ default settings
libHOG::libHOG(){
    use_voc5_impl = false; //use libHOG's native impl
    nLevels = 40;
    interval = 10;
    reset_timers();
}

libHOG::~libHOG(){
    free_vec_of_ptr(hogBuffer_blocks); //free any leftovers from final call to compute_pyramid()
}

//number of pixels (per channel)
int get_imgPyra_num_px(vector< SimpleImg<uint8_t>* > imgPyramid){
    int num_px = 0;
    for(int s=0; s<imgPyramid.size(); s++){
        num_px += imgPyramid[s]->height * imgPyramid[s]->width;
    }
    return num_px;
}

//GB of data output from histogram
double get_hist_gb(vector< int > hogHeight, vector< int > hogWidth){
    int hist_bytes = 0;
    for(int s=0; s<hogHeight.size(); s++){
        hist_bytes += hogHeight[s] * hogWidth[s] * 18 * 4; //18 bins, 4-byte float data
    }
    double hist_gb = ((double)hist_bytes) / 1e9;
    return hist_gb;
}

//hand-coded impl of pyramid. (will modularize it better eventually)
// typical: img pyramid: 16.901926 ms, gradients: 4.586978 ms, hist: 9.008650 ms, norm: 6.529126 ms
//@param img_Mat = image in OpenCV data format
void libHOG::compute_pyramid(cv::Mat img_Mat){

    //cleanup results from previous call to compute_pyramid()
    free_vec_of_ptr(hogBuffer_blocks); //it's ok if hogBuffer isn't allocated yet
    hogHeight.resize(0);
    hogWidth.resize(0);    

    int n_iter = 1; //not really "iterating" -- just number of times to run the experiment
#if 0
    if(n_iter < 10){
        printf("    WARNING: n_iter = %d. For statistical significance, we recommend n_iter=10 or greater. \n", n_iter);
    }
#endif

    libHOG_kernels lHog; //libHOG_kernels constructor initializes lookup tables & constants (mostly for orientation bins)
    voc5_reference_kernels voc5;
    sc = pow(2, 1 / (float)interval);

    //vectors of buffers (one pointer per HOG scale)
    imgPyramid.resize(nLevels);
    ori.resize(nLevels); 
    mag.resize(nLevels); 
    hogBuffer.resize(nLevels);
    normImg.resize(nLevels);

    hogBuffer_blocks.resize(nLevels); //this is the output HOG pyramid
    hogHeight.resize(nLevels); //TODO: reset to zeros
    hogWidth.resize(nLevels);

    int num_px = 0;
    double hist_gb = 0;

    double start_time = read_timer();

    //TODO: put n_iter OUTSIDE of this function. 
    //for(int iter=0; iter<n_iter; iter++){ //do several runs, take the avg time

//step 1: image pyramid
        double img_pyra_start = read_timer();
          gen_img_pyra(img_Mat);
        img_pyra_time += (read_timer() - img_pyra_start);
 
        num_px = get_imgPyra_num_px(imgPyramid);

//step 1.1: now that we know img dimensions, allocate memory for HOG stuff
        double alloc_start = read_timer();
          allocate_bufs();
        alloc_time += (read_timer() - alloc_start);

        hist_gb = get_hist_gb(hogHeight, hogWidth);

#if 1

//step 2: gradients
        double grad_start = read_timer();

        #pragma omp parallel for
        for(int s=0; s<nLevels; s++){

            int sbin = get_sbin_for_scale(s, interval);

            //[mag, ori] = gradient(img)
            if(use_voc5_impl){
                voc5.gradient(imgPyramid[s]->height, imgPyramid[s]->width, imgPyramid[s]->stride, 
                              imgPyramid[s]->n_channels, ori[s]->n_channels, imgPyramid[s]->data, ori[s]->data, mag[s]->data);
            }
            else{
                lHog.gradient(imgPyramid[s]->height, imgPyramid[s]->width, imgPyramid[s]->stride, 
                              imgPyramid[s]->n_channels, ori[s]->n_channels, imgPyramid[s]->data, ori[s]->data, mag[s]->data);
            }
        }

        grad_time += (read_timer() - grad_start);

//step 3: histogram cells
        double hist_start = read_timer();

        #pragma omp parallel for
        for(int s=0; s<nLevels; s++){

            int sbin = get_sbin_for_scale(s, interval);
            if(use_voc5_impl){
                voc5.computeCells(imgPyramid[s]->height, imgPyramid[s]->width, imgPyramid[s]->stride, sbin,
                                  ori[s]->data, mag[s]->data, hogHeight[s], hogWidth[s], hogBuffer[s]);
            }
            else{ 
                lHog.computeCells_gather(imgPyramid[s]->height, imgPyramid[s]->width, imgPyramid[s]->stride, sbin,
                                         ori[s]->data, mag[s]->data, hogHeight[s], hogWidth[s], hogBuffer[s]);
            }
        }

        hist_time += (read_timer() - hist_start);

//step 4: normalize cells into blocks
        double norm_start = read_timer();

        #pragma omp parallel for
        for(int s=0; s<nLevels; s++){

            //normImg(x,y) = sum( hist(x,y,0:17) )
            lHog.hogCell_gradientEnergy(hogBuffer[s], hogHeight[s], hogWidth[s], normImg[s]); // no substantial differences from voc5 version

            //blocks = normalizeCells(hist, normImg)
            if(use_voc5_impl){
                voc5.normalizeCells(hogBuffer[s], normImg[s], hogBuffer_blocks[s], hogHeight[s], hogWidth[s]);
            }
            else{
                lHog.normalizeCells(hogBuffer[s], normImg[s], hogBuffer_blocks[s], hogHeight[s], hogWidth[s]); 
            }
        }

        norm_time += (read_timer() - norm_start);

        delete_vec_of_obj(imgPyramid);
        delete_vec_of_obj(ori);
        delete_vec_of_obj(mag);
        free_vec_of_ptr(hogBuffer);
        // don't free hogBuffer_blocks() until the beginning of the next call to compute_pyramid(). 
        // ... this allows the user to use the data in hogBuffer_blocks().
#endif
    //}
    double end_timer = read_timer() - start_time;
    printf("  avg time for multiscale = %f ms \n", end_timer/n_iter);
    printf("    img pyramid: %f ms, malloc: %f ms, gradients: %f ms, hist: %f ms, norm: %f ms\n", img_pyra_time/n_iter, alloc_time/n_iter, grad_time/n_iter, hist_time/n_iter, norm_time/n_iter);
    
    double px_gb = ((double)num_px * 3) / 1e9;
    printf("    img pyramid: %d pixels = %f GB, hist output: %f GB \n", num_px, px_gb, hist_gb);
    printf("    img pyramid: %f GB/s, gradients: %f GB/s, hist: %f GB/s, norm: %f GB/s \n", 
           px_gb/(img_pyra_time/(n_iter * 1000)),
           px_gb/(grad_time/(n_iter * 1000)),
           px_gb/(hist_time/(n_iter * 1000)),
           hist_gb/(norm_time/(n_iter * 1000)) ); //w.r.t GB of input data. and ms to sec.
    //note: input to grad = num_px*3 channels. input to hist = num_px * (1 byte ori + 2 byte mag). input to hist and grad are same size.
    //note: output from hist and output from norm are approx the same size
}

void libHOG::gen_img_pyra(cv::Mat img_Mat){
    double img_pyra_start = read_timer();

    assert( nLevels == 4*interval ); //TODO: relax this.
    #pragma omp parallel for
    for(int i=0; i<interval; i++){
        float downsampleFactor = 1/pow(sc, i);
        //printf("downsampleFactor = %f \n", downsampleFactor);

        //top 10 scales
        cv::Mat img_scaled = downsampleWithOpenCV(img_Mat, downsampleFactor);
        imgPyramid[i] = new SimpleImg<uint8_t>(img_scaled); //use w/ sbin=4

        imgPyramid[i + interval] = new SimpleImg<uint8_t>(img_scaled); //use w/ sbin=8

        //next 10 scales
        img_scaled = downsampleWithOpenCV(img_Mat, downsampleFactor/2);
        imgPyramid[i + 2*interval] = new SimpleImg<uint8_t>(img_scaled); //use w/ sbin=8

        //bottom 10 scales
        img_scaled = downsampleWithOpenCV(img_Mat, downsampleFactor/4);
        imgPyramid[i + 3*interval] = new SimpleImg<uint8_t>(img_scaled); //use w/ sbin=8
    }
}

//assumes gen_img_pyra() has been called.
void libHOG::allocate_bufs(){
    for(int s=0; s<nLevels; s++){
        int sbin = get_sbin_for_scale(s, interval);

        ori[s] = new SimpleImg<uint8_t>(imgPyramid[s]->height, imgPyramid[s]->stride, 1);
        mag[s] = new SimpleImg<int16_t>(imgPyramid[s]->height, imgPyramid[s]->stride, 1);

        hogBuffer[s] = allocate_hist(imgPyramid[s]->height, imgPyramid[s]->width, sbin,
                                         hogHeight[s], hogWidth[s]); //hog{Height,Width} are passed by ref.
        hogBuffer_blocks[s] = allocate_hist(imgPyramid[s]->height, imgPyramid[s]->width, sbin,
                                                hogHeight[s], hogWidth[s]); //for normalized result
        normImg[s] = (float*)malloc_aligned(32, hogWidth[s] * hogHeight[s] * sizeof(float));
    }
}

void libHOG::reset_timers(){
    img_pyra_time = 0;
    alloc_time = 0;
    grad_time = 0;
    hist_time = 0;
    norm_time = 0;
}

//HACK: assumes we're using the protocol "sbin=4 for top octave; sbin=8 otherwise"
int libHOG::get_sbin_for_scale(int scaleIdx, int interval){
    int sbin;
    if(scaleIdx < interval) sbin=4; //top octave
    else sbin = 8; //other octaves
    return sbin;
}


