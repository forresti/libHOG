#ifndef __VOC5_REFERENCE_KERNELS_H__
#define __VOC5_REFERENCE_KERNELS_H__
#include <sys/time.h>
#include <stdint.h> //for uintptr_t
#include <immintrin.h> //256-bit AVX
#include <xmmintrin.h> //for other SSE-like stuff
#include <string>

using namespace std;

class voc5_reference_kernels{

  public:
    voc5_reference_kernels();
    ~voc5_reference_kernels();

    void init_atan2_constants();
    void init_atan2_LUT();

    void gradient(int height, int width, int stride, int n_channels_input, int n_channels_output,
                  uint8_t *__restrict__ img, uint8_t *__restrict__ outOri, int16_t *__restrict__ outMag);

    void computeCells(int imgHeight, int imgWidth, int imgStride, int sbin,
                                     uint8_t *__restrict__ ori, int16_t *__restrict__ mag,
                                     int outHistHeight, int outHistWidth,
                                     float *__restrict__ outHist);

    void normalizeCells(float *__restrict__ in_hogHist, float *__restrict__ in_normImg,
                             float *__restrict__ out_hogBlocks,
                             int histHeight, int histWidth);

  private:
    char ATAN2_TABLE[512][512]; // values are 0 to 18

    // unit vectors used to compute gradient orientation (initialized in init_atan2_constants(), which is called in the constructor)
    double  uu[9];
    double  vv[9];
};
#endif

