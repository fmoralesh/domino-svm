#pragma once
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
//#include <dirent.h>
#include <iostream>
#include <fstream>
#include <string>

using namespace cv;
using namespace std;

void drawLine(Mat histImage, int bin_w, int i, int height, Mat b_hist, Scalar color);
void drawHistogram(Mat img);
void equalizeColorImage(cv::InputArray src, cv::OutputArray dst);


