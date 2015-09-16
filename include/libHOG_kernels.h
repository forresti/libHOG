#ifndef __LIBHOG_KERNELS_H__
#define __LIBHOG_KERNELS_H__
#include <sys/time.h>
#include <stdint.h> //for uintptr_t
#include <immintrin.h> //256-bit AVX
#include <xmmintrin.h> //for other SSE-like stuff
#include <string>

using namespace std;

class libHOG_kernels{

  public:
    libHOG_kernels();
    ~libHOG_kernels();

    void init_atan2_constants();
    void init_atan2_LUT();

    //enables us to load 8-bit values, but work in 16-bit. 
    void upcast_8bit_to_16bit(__m128i in_xLo,     __m128i in_xHi,     __m128i in_yLo,     __m128i in_yHi,
                              __m128i &out_xLo_0, __m128i &out_xHi_0, __m128i &out_yLo_0, __m128i &out_yHi_0, //bottom bits, in 16-bit
                              __m128i &out_xLo_1, __m128i &out_xHi_1, __m128i &out_yLo_1, __m128i &out_yHi_1); //top bits,    in 16-bit

    void select_epi16(__m128i magChannel, __m128i old_magMax,
                      __m128i gradX_channel, __m128i gradY_channel,
                      __m128i &gradX_max, __m128i &gradY_max);

    __m128i approx_atan2_bin(__m128i gradX_max, __m128i gradY_max);

    void ori_atan2_LUT(__m128i gradX_max_0, __m128i gradX_max_1,
                       __m128i gradY_max_0, __m128i gradY_max_1, uint8_t* outOri_currPtr);

    void gradient(int height, int width, int stride, int n_channels_input, int n_channels_output,
                  uint8_t *__restrict__ img, uint8_t *__restrict__ outOri, int16_t *__restrict__ outMag);

    void computeCells_gather(int imgHeight, int imgWidth, int imgStride, int sbin,
                             uint8_t *__restrict__ ori, int16_t *__restrict__ mag,
                             int outHistHeight, int outHistWidth,
                             float *__restrict__ outHist);

    void hogCell_gradientEnergy(float *__restrict__ hogHist, int histHeight, int histWidth,     
                                float *__restrict__ normImg);

    void normalizeCells(float *__restrict__ in_hogHist, float *__restrict__ in_normImg,
                        float *__restrict__ out_hogBlocks,
                        int histHeight, int histWidth);

  private:
    char ATAN2_TABLE[512][512]; // values are 0 to 18

    // unit vectors used to compute gradient orientation (initialized in init_atan2_constants(), which is called in the constructor)
    //double  uu[9] = {1.0000, 0.9397, 0.7660, 0.500, 0.1736, -0.1736, -0.5000, -0.7660, -0.9397};
    //double  vv[9] = {0.0000, 0.3420, 0.6428, 0.8660, 0.9848, 0.9848, 0.8660, 0.6428, 0.3420};
    double  uu[9];
    double  vv[9];
    int16_t uu_fixedpt[9]; //scalar fixed-pt (scaled up by 100)
    int16_t vv_fixedpt[9];
    __m128i uu_fixedpt_epi16[9]; //each of these vectors is bunch of copies of uu_fixedpt[i]
    __m128i vv_fixedpt_epi16[9];


};
#endif

