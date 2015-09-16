#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cassert>
#include <xmmintrin.h>
#include <pmmintrin.h> //for _mm_hadd_pd()
#include "SimpleImg.hpp"
#include "libHOG.h"
#include "helpers.h"

using namespace std;

int main (int argc, char **argv)
{
    //TODO: take image path as input

    int nLevels = 40;
    int interval = 10;
    bool use_voc5 = 0;
    libHOG lHOG(nLevels, interval, use_voc5);
    cv::Mat img_Mat = cv::imread("images_640x480/carsgraz_001.image.jpg");
    lHOG.compute_pyramid(img_Mat);  
    //TODO: save output HOG as CSV

    return 0;
}

