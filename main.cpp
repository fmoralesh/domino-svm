/*********************************************
 * FILE NAME: main.cpp                       *
 * DESCRIPTION: main function for our        *
 *              domino recognizing algorit   *
 * AUTHORS: Diego Peña, Victor García,       *
 *          Fabio Morales, Andreina Duarte   *
 *********************************************/

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

#include "squares.h"

int main() {
    // Reading the image.
    Mat img = imread("./data/data2.jpeg",CV_LOAD_IMAGE_COLOR);

    std::vector<vector<Point> > squares;
    
    findSquares(img, squares);
    drawSquares(img, squares);

    cv::waitKey(0);
    return 0;
}