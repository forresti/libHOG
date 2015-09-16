#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cassert>
//#include <xmmintrin.h>
//#include <pmmintrin.h> //for _mm_hadd_pd()

//#include "SimpleImg.h"
#include "SimpleImg.hpp"
//#include "streamHog.h"
//#include "helpers.h"
using namespace std;

//out_type is uint8_t or int16_t
template<typename out_type>
void avg_channels(SimpleImg<uint8_t> &in_img, SimpleImg<out_type> &out_img)
{
    for(int y=0; y<in_img.height; y++)
    {
        for(int x=0; x<in_img.width; x++)
        {
            out_img.data[y*(out_img.stride) + x] = (unsigned char)0;
            for(int ch=0; ch<3; ch++){
                out_img.data[y*(out_img.stride) + x] += in_img.data[y*in_img.stride + x + ch*in_img.stride*in_img.height] / 3;
            }
        }
    }
}

void SimpleImg_test(){
    SimpleImg<uint8_t> img("images_640x480/carsgraz_001.image.jpg");

    SimpleImg<uint8_t> out_8bit_img(img.height, img.width, 1);
    //avg_channels(img, out_8bit_img); //out_img gets filled in
    avg_channels<uint8_t>(img, out_8bit_img); //out_img gets filled in
    out_8bit_img.simple_imwrite("./tests_output/SimpleImg_out_8bit.jpg");

    SimpleImg<int16_t> out_16bit_img(img.height, img.width, 1);
    avg_channels<int16_t>(img, out_16bit_img); //out_img gets filled in
    out_16bit_img.simple_imwrite("./tests_output/SimpleImg_out_16bit.jpg");
}

int main (int argc, char **argv)
{
    SimpleImg_test();
    
    return 0;

}


