/**********************************************
 * FILE NAME: squares.h                       *
 * DESCRIPTION: detects squares in an image   *
 * AUTHORS: alyssaq/opencv                    *
 *                                            *
 *********************************************/

#ifndef SQUARES_H
#define SQUARES_H
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <dirent.h>
#include <iostream>
#include <fstream>
#include <string>

using namespace cv;
using namespace std;

void help();
double angle( Point pt1, Point pt2, Point pt0 );
void findSquares( const Mat& image, vector<vector<Point> >& squares );
void drawSquares( Mat& image, const vector<vector<Point> >& squares );
int numberSquares(vector<vector<Point> >& squares );

void getDominoID(cv::Mat domino, float dominosID[][128]);

#endif