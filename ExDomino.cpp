#include <opencv2\opencv.hpp>
#include <iostream>
#include <vector>
#include "ExDomino.h"

using namespace cv;
using namespace std;

void drawLine(Mat histImage, int bin_w, int i, int height, Mat b_hist, Scalar color) {
	line(histImage,
		Point(bin_w*(i - 1), height - cvRound(b_hist.at<float>(i - 1))),
		Point(bin_w*(i), height - cvRound(b_hist.at<float>(i))),
		color);
}

void drawHistogram( Mat img)
{/*
	Mat img_gray;
	cvtColor(img, img_gray, CV_BGR2GRAY);
	Mat img_bw;
	threshold(img_gray, img_bw, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);*/
	// Dividir los canales BRG
	/*vector<Mat> bgr;
	split(img, bgr);
	*/
	Mat img_bw = img;
	// Crear un histograma con 256 bin (numero de subdivisiones) uno por pixel [0..256]
	//imshow("binaria", img_bw);
	int numbins = 256;

	// Establecer rango para los canales (B, G, R)
	float range[] = { 0, 256 };
	const float* histRange = { range };

	Mat b_hist, g_hist, r_hist, hist;
	
	calcHist(&img_bw, 1, 0, Mat(), hist, 1, &numbins, &histRange);
	
	// Tamano del histograma
	int h_width = 512;
	int h_height = 500;

	// Crear una imagen para dibujar en ella
	Mat histImage(h_height, h_width, CV_8UC3, Scalar(20, 20, 20));

	normalize(hist, hist, 0, h_height, NORM_MINMAX);
	int bin_width = cvRound((float)h_width / (float)numbins);

	// Dibujar cada una de las lineas
	for (int i = 1; i < numbins; i++)
	{

		drawLine(histImage, bin_width, i, h_height, hist, Scalar(0, 255, 0));
	}

	// Mostrar el histograma
	//imshow("Histograma", histImage);
	//float a= hist.at<unsigned char>(1, 1);
	cout << endl << "Este domino tiene:"<</*a<<endl<<endl <<*/ hist<< endl;
	//InputArray arrayhist = hist;
	
}
