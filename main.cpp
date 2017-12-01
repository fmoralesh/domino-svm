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
    Mat img = imread("./data/img_prueba_2_4.jpg",CV_LOAD_IMAGE_COLOR);

    // Resizing the image.
    Mat img_resized(img.size().height/4,img.size().width/4, CV_8UC3, Scalar(255,255,255));
    resize(img, img_resized, img_resized.size(), 0, 0, cv::INTER_CUBIC);
    
    std::vector<vector<Point> > squares;
    
    findSquares(img_resized, squares);
    drawSquares(img_resized, squares);

    cv::waitKey(0);
    return 0;
}