#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cassert>
#include <xmmintrin.h>
#include <pmmintrin.h> //for _mm_hadd_pd()
#include <opencv2/opencv.hpp> //only used for file I/O

#include "SimpleImg.hpp"
#include "voc5_reference_kernels.h"
#include "libHOG_kernels.h"
#include "libHOG.h"
#include "helpers.h"
#include "test_libHOG.h"
using namespace std;

bool isVerbose = false;

//test template for: 
//  bestChannel = argmax(mag[0,1,2]); 
//  [gradX, gradY] = gradX[bestChannel], gradY[bestChannel]
//@in-out gradX_max[8], gradY_max[8] = result from this channel's iteration
//@return bool = "pass/fail"
bool test_ori_argmax(int16_t magChannel[8],    int16_t old_magMax[8],
                    int16_t gradX_channel[8], int16_t gradY_channel[8],
                    int16_t* gradX_max,     int16_t* gradY_max,
                    int16_t gold_gradX_max[8], int16_t gold_gradY_max[8]) //gold_* = expected output
{

    libHOG_kernels lHog_k;

    //copy to SSE registers
    __m128i magChannel_sse = _mm_load_si128((__m128i*)magChannel); //current channel's mag
    __m128i old_magMax_sse = _mm_load_si128((__m128i*)old_magMax); //max mag found in previous channels
    __m128i gradX_channel_sse = _mm_load_si128((__m128i*)gradX_channel); //current channel's grad{X,Y}
    __m128i gradY_channel_sse = _mm_load_si128((__m128i*)gradY_channel);
    __m128i gradX_max_sse = _mm_load_si128((__m128i*)gradX_max);
    __m128i gradY_max_sse = _mm_load_si128((__m128i*)gradY_max);

    lHog_k.select_epi16(magChannel_sse, old_magMax_sse, 
                      gradX_channel_sse, gradY_channel_sse,
                      gradX_max_sse, gradY_max_sse); //grad{X,Y}_max_sse are passed by ref, so they get updated.
   
    //print_epi16(gradY_max_sse, "gradY_max"); //TODO: get this to work w/o segfault
    //print_epi16(gradY_max_sse, "gradY_max");
 
    //copy back passed-by-ref grad{X,Y}_max_sse
    _mm_store_si128((__m128i*)gradX_max, gradX_max_sse); //write SSE result back to scalar grad{X,Y}_max
    _mm_store_si128((__m128i*)gradY_max, gradY_max_sse);

    //check correctness
    bool isGood = true;
    for(int i=0; i<8; i++){
        if(gradY_max[i] != gold_gradX_max[i]){
            isGood = false;
            printf("    gradX_max[%d]. expected:%d, got:%d \n", i, gold_gradX_max[i], gradX_max[i]);
        }
        if(gradY_max[i] != gold_gradY_max[i]){
            isGood = false;
            printf("    gradY_max[%d]. expected:%d, got:%d \n\n", i, gold_gradY_max[i], gradY_max[i]);
        }
    }
    return isGood;
}

//scalar reference impl.
//you call this ONCE per channel. you externally update old_magMax.
//@in-out gradX_max[8], gradY_max[8]
void reference_ori_argmax(int16_t magChannel[8],    int16_t old_magMax[8],
                          int16_t gradX_channel[8], int16_t gradY_channel[8],
                          int16_t* gradX_max,     int16_t* gradY_max)
{

    for(int i=0; i<8; i++){ //iterate over sse-style vector
        if(magChannel[i] > old_magMax[i]){
            gradX_max[i] = gradX_channel[i];
            gradY_max[i] = gradY_channel[i];    
        }
    }
    //now, YOU update old_magMax and call again on next channel.
}

//test libHOG::select_epi16(), which gets the argmax mag channel, 
//     and stores the gradX,gradY of that argmax channel.
void run_tests_ori_argmax(){
    printf("run_tests_ori_argmax() \n");
    int numFailed=0;

  //test suite with 3 subtests; 1 per channel
    int16_t  magChannel0[8] = {1,2,3,4,5,6,7,8};
    int16_t  magChannel1[8] = {110,120,130,140,150,160,1,180};
    int16_t  magChannel2[8] = {0,200,50,0,0,0,0,0};
    int16_t* magChannel[3] = {magChannel0, magChannel1, magChannel2}; 
    int16_t  old_magMax[8] = {0,0,0,0,0,0,0,0};   

    int16_t gradX_channel[3][8]; 
    int16_t gradY_channel[3][8];
    int16_t gradX_max[8] = {0,0,0,0,0,0,0,0};
    int16_t gradY_max[8] = {0,0,0,0,0,0,0,0};

    //initialize gradX, gradY with some arbitrary values...
    // gradX[ch=0][:] = 1,...,8. gradX[ch=1][:] = 11...18, gradX[ch=2][:] = 21...28 
    for(int ch=0; ch<3; ch++){
        for(int i=0; i<8; i++){
            gradX_channel[ch][i] = ch*10 + i + 1;
            gradY_channel[ch][i] = ch*10 + i + 1;
        }
    }
    
    int16_t gold_gradX_max[8];
    int16_t gold_gradY_max[8];

    //test grad{X,Y}_max calculation. (this is the gradX, gradY of the argmax magnitude channel)
    for(int ch=0; ch<3; ch++){
        //printf("channel %d \n", ch);

        //calculate 'gold' for this channel's iteration
        reference_ori_argmax(magChannel[ch], old_magMax, 
                             gradX_channel[ch], gradY_channel[ch],
                             gold_gradX_max, gold_gradY_max); //updates gold_grad{X,Y}_max  

        bool isGood = test_ori_argmax(magChannel[ch], old_magMax, 
                                      gradX_channel[ch], gradY_channel[ch], //TODO: &gradX_channel[ch][0], if needed
                                      gradX_max, gradY_max,
                                      gold_gradX_max, gold_gradY_max);

        for(int i=0; i<8; i++){ //emulate _mm_max_epi16(magChannel[ch], old_magMax)
            old_magMax[i] = max(magChannel[ch][i], old_magMax[i]); 
        }
        if(!isGood){ numFailed++; }
   }

    printf("    number of select_epi16 tests failed: %d \n", numFailed); 
}

//@return number of elements that are different.
template<class my_pixel_t>
int diff_imgs(my_pixel_t* img_gold, my_pixel_t* img_test, int imgHeight, int imgWidth, int imgDepth,
               string img_gold_name, string img_test_name){

    int num_diff = 0;
    for(int y=0; y<imgHeight; y++){
        for(int x=0; x<imgWidth; x++){
            for(int d=0; d<imgDepth; d++){
                my_pixel_t gold_element = img_gold[x*imgDepth + y*imgWidth*imgDepth + d];
                my_pixel_t test_element = img_test[x*imgDepth + y*imgWidth*imgDepth + d];
                if(test_element != gold_element){
                    num_diff++;

                    //e.g. x=1, y=1, d=31, voc5=..., libHOG=...
                    if(isVerbose){
                        printf("      x=%d, y=%d, d=%d. %s=%d, %s=%d \n", x, y, d, 
                                img_gold_name.c_str(), gold_element, img_test_name.c_str(), test_element);
                    }
                }
            }
        }
    }
    return num_diff;
}

int diff_hogs(float* hog_gold, float* hog_test, int hogHeight, int hogWidth, int hogDepth,
               string hog_gold_name, string hog_test_name){
    float eps_diff = 0.01;
    int num_diff = 0;

    for(int y=0; y<hogHeight; y++){
        for(int x=0; x<hogWidth; x++){
            for(int d=0; d<hogDepth; d++){
                float gold_element = hog_gold[x*hogDepth + y*hogWidth*hogDepth + d];
                float test_element = hog_test[x*hogDepth + y*hogWidth*hogDepth + d];
                if( fabs(test_element - gold_element) > eps_diff){
                    num_diff++;
                    if(isVerbose){
                        //e.g. x=1, y=1, d=31, voc5=..., libHOG=...
                        printf("      x=%d, y=%d, d=%d. %s=%f, %s=%f \n", x, y, d, 
                                hog_gold_name.c_str(), gold_element, hog_test_name.c_str(), test_element);
                    }
                }
            }
        }
    }
    return num_diff;
}

//correctness check
void test_gradients_voc5_vs_libHOG(){
    printf("test_gradients_voc5_vs_libHOG()\n");
    libHOG_kernels lHog_k; //libHOG constructor initializes lookup tables & constants (mostly for orientation bins)
    voc5_reference_kernels voc5;
    int sbin = 4;

    SimpleImg<uint8_t>img("images_640x480/carsgraz_001.image.jpg");

//TODO: use STRIDE instead of WIDTH for the following:
    SimpleImg<uint8_t> ori_libHOG(img.height, img.width, 1); //out img has just 1 channel
    SimpleImg<uint8_t> ori_voc5(img.height, img.width, 1);
    SimpleImg<int16_t> mag_libHOG(img.height, img.width, 1); //out img has just 1 channel
    SimpleImg<int16_t> mag_voc5(img.height, img.width, 1); 

  //[mag, ori] = gradient_libHOG(img)
    voc5.gradient(img.height, img.width, img.stride, img.n_channels, ori_voc5.n_channels, img.data, ori_voc5.data, mag_voc5.data);
    lHog_k.gradient(img.height, img.width, img.stride, img.n_channels, ori_libHOG.n_channels, img.data, ori_libHOG.data, mag_libHOG.data); 

    mag_voc5.simple_imwrite("tests_output/mag_voc5.jpg");
    mag_libHOG.simple_imwrite("tests_output/mag_libHOG.jpg");
    //ori_voc5.simple_imwrite("tests_output/ori_voc5.jpg");
    //ori_libHOG.simple_imwrite("tests_output/ori_libHOG.jpg");
    ori_voc5.simple_csvwrite("tests_output/ori_voc5.csv");
    ori_libHOG.simple_csvwrite("tests_output/ori_libHOG.csv");

    int num_diff;

    //orientation diff
    if(isVerbose)  printf("    diff ori:\n");

    num_diff = diff_imgs<uint8_t>(ori_voc5.data, ori_libHOG.data, img.height, img.width, 1, "ori_voc5", "ori_libHOG");
    printf("    orientation: %d / %d elements differ \n", num_diff, (img.height * img.width * 3));

    //magnitude diff
    if(isVerbose) printf("    diff mag:\n");

    num_diff = diff_imgs<int16_t>(mag_voc5.data, mag_libHOG.data, img.height, img.width, 1, "mag_voc5", "mag_libHOG");
    printf("    magnitude: %d / %d elements differ \n", num_diff, (img.height * img.width * 3));
}

//correctness check
void test_computeCells_voc5_vs_libHOG(){
    printf("test_computeCells_voc5_vs_libHOG() \n");
    libHOG_kernels lHog_k; //libHOG constructor initializes lookup tables & constants (mostly for orientation bins)
    voc5_reference_kernels voc5;
    int sbin = 4;

    //SimpleImg<uint8_t> img("./carsgraz001_goofySize_539x471.jpg");
    //SimpleImg<uint8_t> img("./carsgraz001_goofySize_641x480.jpg");
    SimpleImg<uint8_t>img("images_640x480/carsgraz_001.image.jpg");

//TODO: use STRIDE instead of WIDTH for the following:
    SimpleImg<uint8_t> ori_libHOG(img.height, img.width, 1); //out img has just 1 channel
    SimpleImg<uint8_t> ori_voc5(img.height, img.width, 1);
    SimpleImg<int16_t> mag_libHOG(img.height, img.width, 1); //out img has just 1 channel
    SimpleImg<int16_t> mag_voc5(img.height, img.width, 1); 
    int hogWidth, hogHeight;
    float* hogBuffer_voc5 = allocate_hist(img.height, img.width, sbin,
                                          hogHeight, hogWidth); //hog{Height,Width} are passed by ref.
    float* hogBuffer_libHOG = allocate_hist(img.height, img.width, sbin,
                                               hogHeight, hogWidth); //hog{Height,Width} are passed by ref.
    int hogStride = hogWidth; //TODO: change this?

  //[mag, ori] = gradient(img)
    voc5.gradient(img.height, img.width, img.stride, img.n_channels, ori_voc5.n_channels, img.data, ori_voc5.data, mag_voc5.data);
    lHog_k.gradient(img.height, img.width, img.stride, img.n_channels, ori_libHOG.n_channels, img.data, ori_libHOG.data, mag_libHOG.data); 

  //hist = computeCells(mag, ori, sbin)
    voc5.computeCells(img.height, img.width, img.stride, sbin,
                                     ori_libHOG.data, mag_libHOG.data, 
                                     hogHeight, hogWidth, hogBuffer_voc5); 
    lHog_k.computeCells_gather(img.height, img.width, img.stride, sbin,
                             ori_libHOG.data, mag_libHOG.data,
                             hogHeight, hogWidth, hogBuffer_libHOG);
    int hogDepth = 32;
    int num_diff = diff_hogs(hogBuffer_voc5, hogBuffer_libHOG, hogHeight, hogWidth, hogDepth, "voc5_cells", "libHOG_cells");
    printf("    HOG cells: %d / %d elements differ \n", num_diff, (hogHeight * hogWidth * hogDepth) ); 
}

//correctness check
void test_normalizeCells_voc5_vs_libHOG(){
    printf("test_normalizeCells_voc5_vs_libHOG() \n");
    libHOG_kernels lHog_k; //libHOG constructor initializes lookup tables & constants (mostly for orientation bins)
    voc5_reference_kernels voc5;
    int sbin = 4;

    SimpleImg<uint8_t>img("images_640x480/carsgraz_001.image.jpg");
    SimpleImg<uint8_t> ori_voc5(img.height, img.width, 1);
    SimpleImg<int16_t> mag_voc5(img.height, img.width, 1); 
    int hogWidth, hogHeight;
    float* hogBuffer_voc5 = allocate_hist(img.height, img.width, sbin,
                                          hogHeight, hogWidth); //hog{Height,Width} are passed by ref.
    int hogStride = hogWidth; //TODO: change this?
    float* normImg = (float*)malloc_aligned(32, hogWidth * hogHeight * sizeof(float));

    float* hogBuffer_blocks_voc5 = allocate_hist(img.height, img.width, sbin,
                                                 hogHeight, hogWidth);
    float* hogBuffer_blocks_libHOG = allocate_hist(img.height, img.width, sbin,
                                                   hogHeight, hogWidth);

  //[mag, ori] = gradient(img)
    voc5.gradient(img.height, img.width, img.stride, img.n_channels, ori_voc5.n_channels, img.data, ori_voc5.data, mag_voc5.data);

  //hist = computeCells(mag, ori, sbin)
    voc5.computeCells(img.height, img.width, img.stride, sbin,
                                     ori_voc5.data, mag_voc5.data, 
                                     hogHeight, hogWidth, hogBuffer_voc5);

    lHog_k.hogCell_gradientEnergy(hogBuffer_voc5, hogHeight, hogWidth, normImg);

  //blocks = normalizeCells(hist, normImg)
    voc5.normalizeCells(hogBuffer_voc5, normImg, hogBuffer_blocks_voc5, hogHeight, hogWidth); //populate hogBuffer_blocks_voc5
    lHog_k.normalizeCells(hogBuffer_voc5, normImg, hogBuffer_blocks_libHOG, hogHeight, hogWidth); //populate hogBuffer_blocks_libHOG
 
    int hogDepth = 32;
    int num_diff = diff_hogs(hogBuffer_blocks_voc5, hogBuffer_blocks_libHOG, hogHeight, hogWidth, hogDepth, "voc5_blocks", "libHOG_blocks");
    printf("    HOG blocks: %d / %d elements differ \n", num_diff, (hogHeight * hogWidth * hogDepth) ); 

}

//correctness check
// using high-level libHOG interface, instead of low-level libHOG_kernels interface
void test_endToEnd_voc5_vs_libHOG(){
    printf("test_endToEnd_voc5_vs_libHOG() \n");
    cv::Mat img_Mat = cv::imread("images_640x480/carsgraz_001.image.jpg");

    //setup HOG objects
    int hogDepth = 32; //assumed everywhere
    int nLevels = 40;
    int interval = 10;
    bool use_voc5 = 0;
    libHOG lHOG_native(nLevels, interval, use_voc5);

    use_voc5 = 1;
    libHOG lHOG_voc5(nLevels, interval, use_voc5);

    //compute HOG pyramids
    if(isVerbose) printf("  voc5: \n");
    lHOG_voc5.compute_pyramid(img_Mat);

    if(isVerbose) printf("  libHOG native: \n");
    lHOG_native.compute_pyramid(img_Mat);

    //tally up diffs
    int num_diff = 0;
    int num_elem = 0;

    for(int s=0; s < lHOG_native.hogBuffer_blocks.size(); s++){

        assert(lHOG_voc5.hogHeight[s] == lHOG_native.hogHeight[s]); 
        assert(lHOG_voc5.hogWidth[s] == lHOG_native.hogWidth[s]);

        int num_elem_curr = lHOG_native.hogHeight[s] * lHOG_native.hogWidth[s] * hogDepth;
        num_elem += num_elem_curr;

        if(isVerbose) printf("    scale %d: \n", s);
         
        int num_diff_curr = diff_hogs(lHOG_voc5.hogBuffer_blocks[s], lHOG_native.hogBuffer_blocks[s], 
                                      lHOG_native.hogHeight[s], lHOG_native.hogWidth[s], 
                                      hogDepth, "voc5_blocks", "libHOG_blocks");
        num_diff += num_diff_curr;

        if(isVerbose){
            printf("      s=%d, num_elem_curr=%d, num_diff_curr=%d \n", s, num_elem_curr, num_diff_curr);
        }
    }

    float pct_diff = ((float)num_diff / (float)num_elem) * 100;
    printf("    HOG blocks in full pyra: %d / %d elements differ (%f percent) \n", num_diff, num_elem, pct_diff); //TODO: figure out how to put '%' instead of 'percent' without being confused as a format identifier 
}

int main (int argc, char **argv)
{
    //TODO: use a better option parser (if we plan to add more options)
    if(argc > 1){
        if(strcmp(argv[1], "--verbose") == 0){
            isVerbose = true; //global variable in this file
        }
    }
    printf("verbose mode: %d \n", isVerbose);

    //unit tests
    run_tests_ori_argmax();
    test_gradients_voc5_vs_libHOG(); 
    test_computeCells_voc5_vs_libHOG();
    test_normalizeCells_voc5_vs_libHOG();
    test_endToEnd_voc5_vs_libHOG();
}

