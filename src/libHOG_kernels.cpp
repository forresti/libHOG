#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cassert>
#include <xmmintrin.h>
#include <pmmintrin.h> //for _mm_hadd_pd()

#include "libHOG_kernels.h"
#include "helpers.h"
using namespace std;

#define eps 0.0001

//#define L2_GRAD
//#define SCALE_ORI //if defined, scale up the orientation (1 to 18) to make it more visible in output images for debugging

//constructor
libHOG_kernels::libHOG_kernels(){
    init_atan2_constants(); //easier to vectorize alternative to lookup table. (similar to VOC5 hog)
    init_atan2_LUT(); //similar to FFLD hog
}

//destructor
libHOG_kernels::~libHOG_kernels(){ }

//stuff for approximate vectorized atan2
void libHOG_kernels::init_atan2_constants(){
    double  uu_local[9] = {1.0000, 0.9397, 0.7660, 0.500, 0.1736, -0.1736, -0.5000, -0.7660, -0.9397}; //from voc-release5 features.cc
    double  vv_local[9] = {0.0000, 0.3420, 0.6428, 0.8660, 0.9848, 0.9848, 0.8660, 0.6428, 0.3420}; 

    for(int i=0; i<9; i++){
        uu[i] = uu_local[i]; //can't do array[9]={data, data, ...} initialization in a class. 
        vv[i] = vv_local[i];

        uu_fixedpt[i] = round(uu[i] * 100);
        vv_fixedpt[i] = round(vv[i] * 100);

        //vector of copies of uu and vv for SSE vectorization
        uu_fixedpt_epi16[i] = _mm_set_epi16(uu_fixedpt[i],uu_fixedpt[i],uu_fixedpt[i],uu_fixedpt[i],uu_fixedpt[i],uu_fixedpt[i],uu_fixedpt[i],uu_fixedpt[i]); 
        vv_fixedpt_epi16[i] = _mm_set_epi16(vv_fixedpt[i],vv_fixedpt[i],vv_fixedpt[i],vv_fixedpt[i],vv_fixedpt[i],vv_fixedpt[i],vv_fixedpt[i],vv_fixedpt[i]); 
    }
}

//TODO: make this much smaller than 512x512.
// ASSUMES that init_atan2_constants() has already been called.
void libHOG_kernels::init_atan2_LUT(){
    for (int dy = -255; dy <= 255; ++dy) { //pixels are 0 to 255, so gradient values are -255 to 255
        for (int dx = -255; dx <= 255; ++dx) {

            // snap to one of 18 orientations [VOC5 style]
            float best_dot = 0;
            int best_o = 0;
            for (int o = 0; o < 9; o++) {
                float dot = uu[o]*dx + vv[o]*dy;
                if (dot > best_dot) {
                    best_dot = dot;
                    best_o = o;
                }
                else if (-dot > best_dot) {
                    best_dot = -dot;
                    best_o = o+9;
                }
            }
            ATAN2_TABLE[dy + 255][dx + 255] = best_o;
        }
    }
}

//enables us to load 8-bit values, but work in 16-bit. 
void libHOG_kernels::upcast_8bit_to_16bit(__m128i in_xLo,  __m128i in_xHi, __m128i in_yLo,     __m128i in_yHi,
                          __m128i &out_xLo_0, __m128i &out_xHi_0, __m128i &out_yLo_0, __m128i &out_yHi_0, //bottom bits, in 16-bit
                          __m128i &out_xLo_1, __m128i &out_xHi_1, __m128i &out_yLo_1, __m128i &out_yHi_1) //top bits,    in 16-bit
{
    //convert inputs for gradY to 16 bits
    out_xLo_0 = _mm_unpacklo_epi8(in_xLo, _mm_setzero_si128()); //unsigned cast to 16-bit ints -- bottom bits 
    out_xHi_0 = _mm_unpacklo_epi8(in_xHi, _mm_setzero_si128()); 
    out_xLo_1 = _mm_unpackhi_epi8(in_xLo, _mm_setzero_si128()); //unsigned cast to 16-bit ints -- top bits
    out_xHi_1 = _mm_unpackhi_epi8(in_xHi, _mm_setzero_si128());

    //convert inputs for gradY to 16 bits
    out_yLo_0 = _mm_unpacklo_epi8(in_yLo, _mm_setzero_si128()); //unsigned cast to 16-bit ints -- bottom bits 
    out_yHi_0 = _mm_unpacklo_epi8(in_yHi, _mm_setzero_si128()); 
    out_yLo_1 = _mm_unpackhi_epi8(in_yLo, _mm_setzero_si128()); //unsigned cast to 16-bit ints -- top bits
    out_yHi_1 = _mm_unpackhi_epi8(in_yHi, _mm_setzero_si128());
}

//@param magChannel = current channel's magnitude
//@param old_magMax = maximum gradient *seen by previous iteration* (note: this code *doesn't* update old_magMax)
//@in-out gradX_max, gradY_max = output gradient of max channel (of the channels checked so far)
void libHOG_kernels::select_epi16(__m128i magChannel, __m128i old_magMax, 
                             __m128i gradX_channel, __m128i gradY_channel,
                             __m128i &gradX_max, __m128i &gradY_max){

    //print_epi16(gradX_max, "gradX_max prev iteration");

    __m128i isMax = _mm_cmpgt_epi16(magChannel, old_magMax); // = 1 when magChannel is max that we have seen so far

    //if magChannel is max, gradX_channel_tmp=gradX_channel; 
    //else                  gradX_channel_tmp = 0
    __m128i gradX_channel_tmp = _mm_and_si128(gradX_channel, isMax); //zero out non-maxes from this channel
    __m128i gradY_channel_tmp = _mm_and_si128(gradY_channel, isMax); 

    //if magChannel is NOT max, gradX_max_tmp = gradX_max
    //else                      gradX_max_tmp = 0
    __m128i gradX_max_tmp = _mm_andnot_si128(isMax, gradX_max); //zero out non-maxes from previous channels
    __m128i gradY_max_tmp = _mm_andnot_si128(isMax, gradY_max);
   
    gradX_max = _mm_or_si128(gradX_channel_tmp, gradX_max_tmp); //for each element, ONE of these 2 args is nonzero
    gradY_max = _mm_or_si128(gradY_channel_tmp, gradY_max_tmp); 
}

//outOri_currPtr[0:15] = atan2(gradX_max[0:15], gradY_max[0:15])
// compute orientation from {gradX_max, gradY_max}. 
//  implemented as non-vectorized atan2 table lookup. 
// @param grad{X,Y}_max{0,1} = packed 16-bit gradients of max-mag channel
// @param outOri_currPtr = &outOri[y*stride + x]. 
void libHOG_kernels::ori_atan2_LUT(__m128i gradX_max_0, __m128i gradX_max_1, 
                              __m128i gradY_max_0, __m128i gradY_max_1, uint8_t* outOri_currPtr)
{
    //compute orientation from {gradX_max, gradY_max}. 
    //  implemented as non-vectorized atan2 table lookup. 
    int16_t gradX_max_unpacked[16]; //unpacked 8-bit numbers
    int16_t gradY_max_unpacked[16];

    _mm_store_si128( (__m128i*)(&gradX_max_unpacked[0]), gradX_max_0 ); //0:7
    _mm_store_si128( (__m128i*)(&gradX_max_unpacked[8]), gradX_max_1 ); //8:15
    _mm_store_si128( (__m128i*)(&gradY_max_unpacked[0]), gradY_max_0 ); //0:7
    _mm_store_si128( (__m128i*)(&gradY_max_unpacked[8]), gradY_max_1 ); //8:15

    // non-vectorized atan2 table lookup.
    for(int i=0; i<16; i++){ 
        int16_t dx = gradX_max_unpacked[i];
        int16_t dy = gradY_max_unpacked[i];
        uint8_t ori = ATAN2_TABLE[dy+255][dx+255]; //ATAN2_TABLE is 0-18. (char)
        //uint8_t ori = ATAN2_TABLE[ (dy>>2) + 255 ][ (dx>>2) + 255 ]; //test LUT quantization -- looks fine.
        #ifdef SCALE_ORI
            ori = ori*9; //to be more visible in output images for debugging
        #endif
        outOri_currPtr[i] = ori; //outOri[y*stride + x + i] = ori;
    }
}

//@param 16-bit packed gradX and gradY
//@return 16-bit sqrt(gradX^2 + gradY^2)
static inline __m128i _L2(__m128i gradX, __m128i gradY){

    __m128i gradX_int32[2]; //[0] = lo bits, [1] = hi bits
    __m128i gradY_int32[2]; 
    __m128i result_int32[2]; 
    __m128i result_int16;

    //int16 -> int32
    gradX_int32[0] = _mm_unpacklo_epi16(gradX, _mm_set1_epi16(0));
    gradX_int32[1] = _mm_unpackhi_epi16(gradX, _mm_set1_epi16(0));
    gradY_int32[0] = _mm_unpacklo_epi16(gradY, _mm_set1_epi16(0));
    gradY_int32[1] = _mm_unpackhi_epi16(gradY, _mm_set1_epi16(0));


    for(int i=0; i<2; i++){ //lo and hi bits
        //int32 -> float
        __m128 gradX_float = _mm_cvtepi32_ps(gradX_int32[i]);
        __m128 gradY_float = _mm_cvtepi32_ps(gradY_int32[i]);

        //result = gradX^2 + gradY^2
        gradX_float = _mm_mul_ps(gradX_float, gradX_float);
        gradY_float = _mm_mul_ps(gradY_float, gradY_float);
        __m128 result_float = _mm_add_ps(gradX_float, gradY_float); 

        //result = sqrt(result)
        //result_float = _mm_rsqrt_ps(result_float); //approx reciprocal sqrt
        //result_float = _mm_rcp_ps(result_float); // sqrt = 1/rsqrt
        result_float = _mm_sqrt_ps(result_float);

        //float -> int32
        result_int32[i] = _mm_cvtps_epi32(result_float);
    }

    //int32 -> int16
    result_int16 = _mm_packs_epi16(result_int32[0], result_int32[1]);
    return result_int16;
}

//TODO: replace outOri with outGradX_max and outGradY_max. (after calling gradient_libHOG_kernels, you do a lookup table)
//  or, just do the lookup in here...
void libHOG_kernels::gradient(int height, int width, int stride, int n_channels_input, int n_channels_output,
                  uint8_t *__restrict__ img, uint8_t *__restrict__ outOri, int16_t *__restrict__ outMag){
    assert(n_channels_input == 3);
    assert(n_channels_output == 1);
    assert(sizeof(__m128i) == 16);
    int loadSize = sizeof(__m128i); // 16 bytes = 128 bits -- load 16 uint8_t at a time
    int loadSize_16bit = loadSize/2; //load 8 int16_t at a time

    //input pixels
    __m128i xLo, xHi, yLo, yHi; //packed 8-bit
    __m128i xLo_0, xHi_0, yLo_0, yHi_0; //bottom bits: upcast from 8-bit to 16-bit
    __m128i xLo_1, xHi_1, yLo_1, yHi_1; //top bits: upcast from 8-bit to 16-bit

    //gradients
    __m128i gradX_ch[3],   gradY_ch[3];   //packed 8-bit
    __m128i gradX_0_ch[3], gradY_0_ch[3]; //bottom bits: upcast from 8-bit to 16-bit
    __m128i gradX_1_ch[3], gradY_1_ch[3]; //top bits: upcast from 8-bit to 16-bit
    __m128i gradX_max_0, gradX_max_1; //gradX of the max-mag channel
    __m128i gradY_max_0, gradY_max_1; //gradY of the max-mag channel
    __m128i gradMax_0, gradMax_1; //bottom bits, top bits (after arctan)

    //magnitudes
    __m128i mag_ch[3]; //packed 8-bit
    __m128i mag_0_ch[3]; //bottom bits
    __m128i mag_1_ch[3]; //top bits
    __m128i magMax, magMax_0, magMax_1; //packed 8-bit, bottom bits, top bits

    //TODO: assert that stride = (width + width%8), so we can load 8 uchars starting at img[width-2]

    for(int y=0; y<height-2; y++){
        //for(int x=0; x < stride-2; x+=loadSize){ //(stride-2) to avoid falling off the end when doing (location+2) to get xHi
        for(int x=0; x < width-2; x+=loadSize){

            magMax = magMax_0 = magMax_1 = _mm_setzero_si128();

            for(int channel=0; channel<3; channel++){ //TODO: unroll channels

                xLo = _mm_loadu_si128( (__m128i*)(&img[(y+1)*stride + x + channel*height*stride    ]) ); //load sixteen 1-byte unsigned char pixels
                xHi = _mm_loadu_si128( (__m128i*)(&img[(y+1)*stride + x + channel*height*stride + 2]) ); //index as chars, THEN cast to __m128i*  

                yLo = _mm_loadu_si128( (__m128i*)(&img[y*stride + (x+1) + channel*height*stride           ]) );
                yHi = _mm_loadu_si128( (__m128i*)(&img[y*stride + (x+1) + channel*height*stride + 2*stride]) );

                upcast_8bit_to_16bit(xLo, xHi, yLo, yHi,
                                     xLo_0, xHi_0, yLo_0, yHi_0,
                                     xLo_1, xHi_1, yLo_1, yHi_1);

                gradX_0_ch[channel] =  _mm_sub_epi16(xHi_0, xLo_0); // xHi[0:7] - xLo[0:7]
                gradX_1_ch[channel] =  _mm_sub_epi16(xHi_1, xLo_1); // xHi[8:15] - xLo[8:15]

                gradY_0_ch[channel] =  _mm_sub_epi16(yHi_0, yLo_0);
                gradY_1_ch[channel] =  _mm_sub_epi16(yHi_1, yLo_1); 

#ifdef L2_GRAD
                //mag = sqrt(gradX^2 + gradY^2)
                mag_0_ch[channel] = _L2(gradX_0_ch[channel], gradY_0_ch[channel]);
                mag_1_ch[channel] = _L2(gradX_1_ch[channel], gradY_1_ch[channel]);
#else //L1 gradient 
                //mag = abs(gradX) + abs(gradY)
                // this is using the non-sqrt approach that has proved equally accurate to mag=sqrt(gradX^2 + gradY^2)
                mag_0_ch[channel] = _mm_add_epi16( _mm_abs_epi16(gradX_0_ch[channel]), _mm_abs_epi16(gradY_0_ch[channel]) ); // abs(gradX[0:7]) + abs(gradY[0:7])
                mag_1_ch[channel] = _mm_add_epi16( _mm_abs_epi16(gradX_1_ch[channel]), _mm_abs_epi16(gradY_1_ch[channel]) ); // abs(gradX[8:15]) + abs(gradY[8:15])
#endif

                //gradX, gradY of the argmax(magnitude) channel
                select_epi16(mag_0_ch[channel], magMax_0, gradX_0_ch[channel], gradY_0_ch[channel], gradX_max_0, gradY_max_0); //output gradX_max_0, gradY_max_0
                select_epi16(mag_1_ch[channel], magMax_1, gradX_1_ch[channel], gradY_1_ch[channel], gradX_max_1, gradY_max_1); //output gradX_max_1, gradY_max_1 

                //magMax = max(mag_ch[0,1,2])
                magMax_0 = _mm_max_epi16(magMax_0, mag_0_ch[channel]);
                magMax_1 = _mm_max_epi16(magMax_1, mag_1_ch[channel]); 

                gradX_ch[channel] = _mm_packs_epi16(gradX_0_ch[channel], gradX_1_ch[channel]); //16-bit -> 8bit (temporary ... typically, we'd pack up the results later in the pipeline)
                gradY_ch[channel] = _mm_packs_epi16(gradY_0_ch[channel], gradY_1_ch[channel]); //temporary ... typically, we'd pack up the results later in the pipeline.
            }

            //magnitude output is 16-bit to avoid overflow.
            _mm_store_si128( (__m128i*)(&outMag[y*stride + x]                 ), magMax_0 );           
            _mm_store_si128( (__m128i*)(&outMag[y*stride + x + loadSize_16bit]), magMax_1 );

            //atan2 nonvectorized LUT. (not tested for correctness)
            //outOri[y*stride + x + 0:15] = atan2(gradX_max[0:15], gradY_max[0:15])
            ori_atan2_LUT(gradX_max_0, gradX_max_1, gradY_max_0, gradY_max_1, &outOri[y*stride + x]);
        }
    }
}

//traditional approach: "for each pixel, add it to the relevant output cells"
//computeCells_gather(): "for each cell, accumulate all the input pixels"
void libHOG_kernels::computeCells_gather(int imgHeight, int imgWidth, int imgStride, int sbin,
                                    uint8_t *__restrict__ ori, int16_t *__restrict__ mag,
                                    int outHistHeight, int outHistWidth,
                                    float *__restrict__ outHist){

    assert(outHistHeight == (int)round((double)imgHeight/(double)sbin));
    assert(outHistWidth == (int)round((double)imgWidth/(double)sbin));

    assert( (sbin % 2) == 0); //v_LUT[] is only verified for even sbin values
    float v_LUT[sbin*2]; //vx, vy
    for(int i=0; i<sbin; i++){
        //#e.g. for sbin=4, v_LUT = [1/8, 3/8, 5/8, 7/8, 7/8, 5/8, 3/8, 1/8]

        v_LUT[i] = (i*2 + 1.0f)/(sbin*2.0f); //left-hand pixel weights
        v_LUT[sbin*2 - 1 - i] = v_LUT[i]; //mirror for right-hand pixel weights
    }

    const int hogDepth = 32;
    float sbin_inverse = 1.0f / (float)sbin;

    const int n_ori = 18; //TODO: take this as input and assert it's true?
    float local_hist[n_ori]; //accumulate locally, then save to outHist
    int half_sbin = (int)ceil((float)sbin/2);

    //for each HOG cell
//#pragma omp parallel for
    for(int hogY=0; hogY<outHistHeight; hogY++){
        for(int hogX=0; hogX<outHistWidth; hogX++){

            memset(&local_hist[0], 0, n_ori*sizeof(float));

            //range of pixels that influence this hog cell
            //     explained in test_scatter_gather.py: px_gather()
            //     confirmed to match voc-release5 for sbin=4, 6, 7, and 8.

            // example: hogX=2, sbin=4. center pixel = 9.5
            //          look at pixels[x = 6 7 8 9 | 10 11 12 13]
            //same as above, with cached value of half_sbin
            int minPx_x_unclamped = hogX*sbin - half_sbin;
            int minPx_x = max( 0, hogX*sbin - half_sbin ); 
            int maxPx_x = min( hogX*sbin + 2*sbin - half_sbin - 1, imgWidth-1 );

            int minPx_y_unclamped = hogY*sbin - half_sbin;
            int minPx_y = max( 0, hogY*sbin - half_sbin ); 
            int maxPx_y = min( hogY*sbin + 2*sbin - half_sbin - 1, imgHeight-1 );

            //indices into v_LUT
            int startX_local = abs(abs(minPx_x) - abs(minPx_x_unclamped)); //account for clamping 
            int startY_local = abs(abs(minPx_y) - abs(minPx_y_unclamped)); 

            int y_local = startY_local;

            //TODO: make "one cell" a separate inline function?
            for(int y=minPx_y; y<=maxPx_y; y++){

                //int x_local = startX_local; //21ms on R8
                int x_local = 0; //14ms on R8 (breaks hogX=0 ... all else is fine.)
                for(int x=minPx_x; x<=maxPx_x; x++){
                    int curr_ori = ori[y*imgStride + x];
                    float curr_mag = (float)mag[y*imgStride + x];

                    float vx = v_LUT[x_local];
                    float vy = v_LUT[y_local];

                    local_hist[curr_ori] += curr_mag * vx * vy;
                    x_local++;
                }
                y_local++;
            } 

            //write back results. (TODO: use memcpy instead?)
            int outHist_idx = hogX*hogDepth + hogY*outHistWidth*hogDepth; 
            for(int o=0; o<n_ori; o++)
            {
                outHist[outHist_idx + o] = local_hist[o];
            }

        }
    } 
}

//TODO: handle hog padding. (for now, just use padded HOG dims as input here)
//@in-out normImg = 1-channel img of size (histWidth x histHeight). we populate this with hist sums.
void libHOG_kernels::hogCell_gradientEnergy(float *__restrict__ hogHist, int histHeight, int histWidth, 
                                       float *__restrict__ normImg)
{
    const int hogDepth = 32;

    //sum up the (0 to 360 degree) hog cells
    for(int y=0; y < histHeight; y++){
        for(int x=0; x < histWidth; x++){

            int hogIdx = x*hogDepth + y*histWidth*hogDepth;
            float norm = 0.0f;
            for(int ori=0; ori<18; ori++){ //0-360 degree ori bins
                norm += hogHist[hogIdx+ori] * hogHist[hogIdx+ori]; //squared -- will do sqrt in hogBlock_normalize()
            }
            normImg[y*histHeight + x] = norm;
        }
    }
}

//calculation for n1, n2, n3, n4 in voc5 code
static inline float _sqrt_norm(float *__restrict__ in_normImg, int x, int y, int histWidth)
{
    float n1 = 1.0 / sqrt(in_normImg[(y)*histWidth + (x)] + //top-left
                          in_normImg[(y)*histWidth + (x+1)]   +
                          in_normImg[(y+1)*histWidth   + (x)] +
                          in_normImg[(y+1)*histWidth   + (x+1)]   + eps);
    return n1;
}

//hog cells -> hog blocks
//@param in_hogHist = hog cells
//@param in_normImg = result of hogCell_gradientEnergy()
//@param out_hogBlocks = normalized hog blocks
// DO NOT CALL hogCell_gradientEnergy() BEFORE THIS. (functionality is pipelined in here.)
void libHOG_kernels::normalizeCells(float *__restrict__ in_hogHist, float *__restrict__ in_normImg, 
                                    float *__restrict__ out_hogBlocks,
                                    int histHeight, int histWidth)
{
    const int hogDepth = 32;

    float* normImg_sqrt = (float*)malloc_aligned(32, histWidth * histHeight * sizeof(float)); //TODO: preallocate

    //precompute n1, n2, n3, n4 (save 4x computation by amoritizing this)
    for(int y=0; y < histHeight-1; y++){
        for(int x=0; x < histWidth-1; x++){
            normImg_sqrt[y*histWidth + x] = _sqrt_norm(in_normImg, x, y, histWidth);
        }
    }

    // compute features
    for(int y=1; y < histHeight-1; y++){
        for(int x=1; x < histWidth-1; x++){

          //Forrest's way of pipelining & sharing n1-n4 across neighborhoods.
            float n1 = normImg_sqrt[(y-1)*histWidth + (x-1)];
            float n2 = normImg_sqrt[(y-1)*histWidth + (x)];
            float n3 = normImg_sqrt[(y)*histWidth + (x-1)];
            float n4 = normImg_sqrt[(y)*histWidth + (x)];

            float t1 = 0;
            float t2 = 0;
            float t3 = 0;
            float t4 = 0;

            // contrast-sensitive features
            //src = hist + (x+1)*blocks[0] + (y+1);
            int hogIdx = (x+1)*hogDepth + (y+1)*histWidth*hogDepth;

            for (int o = 0; o < 18; o++) {
                float in_bin = in_hogHist[hogIdx + o]; 

                float h1 = min(in_bin * n1, 0.2f);
                float h2 = min(in_bin * n2, 0.2f);
                float h3 = min(in_bin * n3, 0.2f);
                float h4 = min(in_bin * n4, 0.2f);
               
                //*dst = 0.5 * (h1 + h2 + h3 + h4); 
                out_hogBlocks[hogIdx + o] = 0.5f * (h1 + h2 + h3 + h4);

                t1 += h1;
                t2 += h2;
                t3 += h3;
                t4 += h4;
            }

            // contrast-insensitive features
            //src = hist + (x+1)*blocks[0] + (y+1);
            for (int o = 0; o < 9; o++) {
                //float sum = *src + *(src + 9*blocks[0]*blocks[1]);
                float sum = in_hogHist[hogIdx + o] + in_hogHist[hogIdx+ o+9]; 
                float h1 = min(sum * n1, 0.2f);
                float h2 = min(sum * n2, 0.2f);
                float h3 = min(sum * n3, 0.2f);
                float h4 = min(sum * n4, 0.2f);

                out_hogBlocks[hogIdx + o + 18] = 0.5 * (h1 + h2 + h3 + h4);
            }

            out_hogBlocks[hogIdx + 27] = 0.2357f * t1;
            out_hogBlocks[hogIdx + 28] = 0.2357f * t2;
            out_hogBlocks[hogIdx + 29] = 0.2357f * t3;
            out_hogBlocks[hogIdx + 30] = 0.2357f * t4;
        }
    }
    free(normImg_sqrt);
}

