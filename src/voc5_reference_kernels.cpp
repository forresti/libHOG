#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cassert>
#include <xmmintrin.h>
#include <pmmintrin.h> //for _mm_hadd_pd()

#include "voc5_reference_kernels.h"
#include "helpers.h"
using namespace std;

#define eps 0.0001

//#define L2_GRAD
//#define SCALE_ORI //if defined, scale up the orientation (1 to 18) to make it more visible in output images for debugging

//constructor
voc5_reference_kernels::voc5_reference_kernels(){
    init_atan2_constants(); //easier to vectorize alternative to lookup table. (similar to VOC5 hog)
    init_atan2_LUT(); //similar to FFLD hog
}

//destructor
voc5_reference_kernels::~voc5_reference_kernels(){ }

//stuff for approximate vectorized atan2
void voc5_reference_kernels::init_atan2_constants(){
    double  uu_local[9] = {1.0000, 0.9397, 0.7660, 0.500, 0.1736, -0.1736, -0.5000, -0.7660, -0.9397}; //from voc-release5 features.cc
    double  vv_local[9] = {0.0000, 0.3420, 0.6428, 0.8660, 0.9848, 0.9848, 0.8660, 0.6428, 0.3420}; 

    for(int i=0; i<9; i++){
        uu[i] = uu_local[i]; //can't do array[9]={data, data, ...} initialization in a class. 
        vv[i] = vv_local[i];
    }
}

// ASSUMES that init_atan2_constants() has already been called.
void voc5_reference_kernels::init_atan2_LUT(){
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

//gradient code from voc-release5 DPM. (reference impl)
void voc5_reference_kernels::gradient(int height, int width, int stride, int n_channels_input, int n_channels_output,
                  uint8_t *__restrict__ img, uint8_t *__restrict__ outOri, int16_t *__restrict__ outMag){

    for(int y=1; y<height-1; y++){
        for(int x=1; x<width-1; x++){

            int channel = 0;
            double dx = (double)img[y*stride + (x+1) + channel*height*stride] - 
                            (double)img[y*stride + (x-1) + channel*height*stride];
            double dy = (double)img[(y+1)*stride + x + channel*height*stride] -
                            (double)img[(y-1)*stride + x + channel*height*stride];

            // second color channel
            channel=1;
            double dx2 = (double)img[y*stride + (x+1) + channel*height*stride] -
                            (double)img[y*stride + (x-1) + channel*height*stride];
            double dy2 = (double)img[(y+1)*stride + x + channel*height*stride] -
                            (double)img[(y-1)*stride + x + channel*height*stride];

            // third color channel
            channel=2;
            double dx3 = (double)img[y*stride + (x+1) + channel*height*stride] -
                            (double)img[y*stride + (x-1) + channel*height*stride];
            double dy3 = (double)img[(y+1)*stride + x + channel*height*stride] -
                            (double)img[(y-1)*stride + x + channel*height*stride];

#ifdef L2_GRAD
            double v = dx*dx + dy*dy; //max magnitude (gets updated later)
            double v2 = dx2*dx2 + dy2*dy2;
            double v3 = dx3*dx3 + dy3*dy3;
#else
            double v = fabs(dx) + fabs(dy);
            double v2 = fabs(dx2) + fabs(dy2);
            double v3 = fabs(dx3) + fabs(dy3);
#endif

            // pick channel with strongest gradient
            if (v2 > v) {
                v = v2; 
                dx = dx2;
                dy = dy2;
            }     
            if (v3 > v) {
                v = v3; 
                dx = dx3;
                dy = dy3;
            }     

            // snap to one of 18 orientations
            double best_dot = 0;
            int best_o = 0;
            for (int o = 0; o < 9; o++) {
                double dot = uu[o]*dx + vv[o]*dy;
                if (dot > best_dot) {
                    best_dot = dot;
                    best_o = o;
                } else if (-dot > best_dot) {
                    best_dot = -dot;
                    best_o = o+9;
                }   
            }

            #ifdef SCALE_ORI
            best_o = best_o * 9;
            #endif
#ifdef L2_GRAD
            v = sqrt(v); //Forrest -- no longer need to sqrt the magnitude
#endif

            //to line up with forrest's 0-indexed version....
            outMag[(y-1)*stride + (x-1)] = v;
            outOri[(y-1)*stride + (x-1)] = best_o; 
        }
    }
}

//standalone version of gradient code from voc-release5 DPM. (reference impl)
// output dimensions: outHist[imgHeight/sbin][imgWidth/sbin][hogDepth=32]. row major (like piggyHOG).
//  output stride = output width. (because we already have 32-dimensional features as the inner dimension)
void voc5_reference_kernels::computeCells(int imgHeight, int imgWidth, int imgStride, int sbin, 
                                            uint8_t *__restrict__ ori, int16_t *__restrict__ mag,
                                            int outHistHeight, int outHistWidth,
                                            float *__restrict__ outHist){

    assert(outHistHeight == (int)round((double)imgHeight/(double)sbin));
    assert(outHistWidth == (int)round((double)imgWidth/(double)sbin));

    const int hogDepth = 32;
    float sbin_inverse = 1.0f / (float)sbin;

#pragma omp parallel for
    for(int y=0; y<imgHeight; y++){
        for(int x=0; x<imgWidth; x++){

            int best_o = ori[y*imgStride + x]; //orientation bin -- upcast to int
            int v = mag[y*imgStride + x]; //upcast to int

            // add to 4 histograms around pixel using linear interpolation
            //float xp = ((float)x+0.5)/(float)sbin - 0.5; //this is expensive (replacing it with 'x/4' gives a 1.5x speedup in hogCell)
            //float yp = ((float)y+0.5)/(float)sbin - 0.5;
            float xp = ((float)x+0.5)*sbin_inverse - 0.5;
            float yp = ((float)y+0.5)*sbin_inverse - 0.5;
            int ixp = (int)floor(xp);
            int iyp = (int)floor(yp);
            float vx0 = xp-ixp;
            float vy0 = yp-iyp;
            float vx1 = 1.0-vx0;
            float vy1 = 1.0-vy0;

            if (ixp >= 0 && iyp >= 0) 
            { 
                //*(hist + ixp*imgHeight + iyp + best_o*imgHeight*imgWidth) += vx1*vy1*v; //from VOC5
                outHist[ixp*hogDepth + iyp*outHistWidth*hogDepth + best_o] += vx1*vy1*v; 
            } 

            if (ixp+1 < outHistWidth && iyp >= 0) { 
                //*(hist + (ixp+1)*imgHeight + iyp + best_o*imgHeight*imgWidth) += vx0*vy1*v;
                outHist[(ixp+1)*hogDepth + iyp*outHistWidth*hogDepth + best_o] += vx0*vy1*v;
            } 

            if (ixp >= 0 && iyp+1 < outHistHeight) { 
                //*(hist + ixp*imgHeight + (iyp+1) + best_o*imgHeight*imgWidth) += vx1*vy0*v;
                outHist[ixp*hogDepth + (iyp+1)*outHistWidth*hogDepth + best_o] += vx1*vy0*v;
            } 

            if (ixp+1 < outHistWidth && iyp+1 < outHistHeight) { 
                //*(hist + (ixp+1)*imgHeight + (iyp+1) + best_o*imgHeight*imgWidth) += vx0*vy0*v;
                outHist[(ixp+1)*hogDepth + (iyp+1)*outHistWidth*hogDepth + best_o] += vx0*vy0*v;
            } 
        }
    }
}

//looking for hogCell_gradientEnergy()? 
//the version in libHOG_kernels.cpp is unmodified from voc-release5.

//hog cells -> hog blocks
//@param in_hogHist = hog cells
//@param in_normImg = result of hogCell_gradientEnergy()
//@param out_hogBlocks = normalized hog blocks
void voc5_reference_kernels::normalizeCells(float *__restrict__ in_hogHist, float *__restrict__ in_normImg, 
                                    float *__restrict__ out_hogBlocks,
                                    int histHeight, int histWidth)
{
    const int hogDepth = 32;
    //TODO: test my ptr indexing vs. voc5 ptr indexing.

    // compute features
#pragma omp parallel for
    for(int y=1; y < histHeight-1; y++){
        for(int x=1; x < histWidth-1; x++){

            #if 0 //VOC5 original
            float *dst = feat + x*out[0] + y;
            float *src, *p, n1, n2, n3, n4;
            p = norm + (x+1)*blocks[0] + y+1;
            n1 = 1.0 / sqrt(*p + *(p+1) + *(p+blocks[0]) + *(p+blocks[0]+1) + eps);
            p = norm + (x+1)*blocks[0] + y;
            n2 = 1.0 / sqrt(*p + *(p+1) + *(p+blocks[0]) + *(p+blocks[0]+1) + eps);
            p = norm + x*blocks[0] + y+1;
            n3 = 1.0 / sqrt(*p + *(p+1) + *(p+blocks[0]) + *(p+blocks[0]+1) + eps);
            p = norm + x*blocks[0] + y;
            n4 = 1.0 / sqrt(*p + *(p+1) + *(p+blocks[0]) + *(p+blocks[0]+1) + eps);
            #endif

            #if 1 //VOC5, rewritten with zero-indexing, and without pointer incrementing.
            float n1 = 1.0 / sqrt(in_normImg[(y-1)*histWidth + (x-1)] + //top-left
                                  in_normImg[(y-1)*histWidth + (x)]   +
                                  in_normImg[(y)*histWidth   + (x-1)] +
                                  in_normImg[(y)*histWidth   + (x)]   + eps);

            float n2 = 1.0 / sqrt(in_normImg[(y-1)*histWidth + (x)]   + //top-right
                                  in_normImg[(y-1)*histWidth + (x+1)] +
                                  in_normImg[(y)*histWidth   + (x)]   +
                                  in_normImg[(y)*histWidth   + (x+1)] + eps);

            float n3 = 1.0 / sqrt(in_normImg[(y)*histWidth   + (x-1)] + //bottom-left
                                  in_normImg[(y)*histWidth   + (x)]   +
                                  in_normImg[(y+1)*histWidth + (x-1)] +
                                  in_normImg[(y+1)*histWidth + (x)]   + eps);

            float n4 = 1.0 / sqrt(in_normImg[(y)*histWidth   + (x)]   + //bottom-right
                                  in_normImg[(y)*histWidth   + (x+1)] +
                                  in_normImg[(y+1)*histWidth + (x)]   +
                                  in_normImg[(y+1)*histWidth + (x+1)] + eps);
            #endif

            float t1 = 0;
            float t2 = 0;
            float t3 = 0;
            float t4 = 0;

            #if 0 //VOC5 original
            // contrast-sensitive features
            //src = hist + (x+1)*blocks[0] + (y+1);
            src = hist + (x+1) + (y+1)*histWidth;
            for (int o = 0; o < 18; o++) {
                float h1 = min(*src * n1, 0.2);
                float h2 = min(*src * n2, 0.2);
                float h3 = min(*src * n3, 0.2);
                float h4 = min(*src * n4, 0.2);
                *dst = 0.5 * (h1 + h2 + h3 + h4);
                t1 += h1;
                t2 += h2;
                t3 += h3;
                t4 += h4;
                dst += out[0]*out[1];
                src += blocks[0]*blocks[1];
            }
            #endif

            #if 1 //VOC5, rewritten with explicit array indexing.
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
            #endif
        }
    }
}

