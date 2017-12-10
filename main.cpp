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
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <cmath>
#include "squares.h"

using namespace cv;
using namespace std;

static double getAngle(Point pt1, Point pt2, Point pt0);
void sortCorners(vector<Point2f>& corners, Point2f center);
bool comparator(Point2f a, Point2f b);
double euclideanDistance(Point2f pt1, Point2f pt2);

int main(int argv, char** argc) {
	string imageName;
	if(argv > 1){
		imageName = argc[1];
	}
	else 
		imageName = "./data/1.jpeg";

	int n_dominos=0;
	float dominosID[100][128];
	int training_labels[100];
	for(int i=0; i<100; i++){
		training_labels[i] = -1;
		for(int j=0; j<128; j++){
			dominosID[i][j] = 0;
		}
	}
    // Reading the image.
	Mat src = imread(imageName);
	if (src.empty())
		return -1;

    std::vector<vector<Point> > squares;
    //findSquares(src, squares);
    //drawSquares(src, squares);
	vector< vector<Point> > good_contours;
	vector<RotatedRect> minRect;
	vector< Point > corners;

	// convertir a escala de grises
	Mat gray;
	cvtColor(src, gray, CV_BGR2GRAY);

	// filtro canny de bordes
	Mat bw, blur, otsu;
	GaussianBlur(gray, blur, Size(9,9),0 ,0);
	threshold(blur, otsu, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
	Canny(otsu, bw, 0, 50, 5);

	// Encontrar contornos
	vector<vector<Point> > contours;
	findContours(bw.clone(), contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

	vector<Point2f> approx;
	Mat dst = src.clone();

	for (int i = 0; i < contours.size(); i++) {
		// Se aproximan los contornos a figuras simples
		// a partir de su perimetro
		approxPolyDP(Mat(contours[i]), approx, arcLength(Mat(contours[i]), true)*0.04, true);

		// Si el objeto es muy pequeño
		if (fabs(contourArea(contours[i])) < 100 || !isContourConvex(approx))
			continue;
		
		// Cuadrado (cuatro esquinas)
		if (approx.size() == 4) {
			// se guarda el numero de vertices
			int vtc = approx.size();

			// calculo de cosenos de los bordes
			vector<double> cos;
			for (int j = 2; j < vtc + 1; j++)
				cos.push_back(angle(approx[j%vtc], approx[j - 2], approx[j - 1]));

			// ordenar ascendentemente
			sort(cos.begin(), cos.end());

			// obtener el menor y mayor
			double mincos = cos.front();
			double maxcos = cos.back();

			// Si es un cuadrado, pueden habermas figuras con 4 esquinas que no lo sean
			if (mincos >= -0.3 && maxcos <= 0.5) {
				float dominoID[2][128];
				for(int k=0; k<2; k++){
					for(int m=0; m<128; m++){
						dominoID[k][m] = 0;
					}
				}
				Point2f center(0, 0);
				for (int k = 0; k < approx.size(); k++) {
					center += approx[k];
				}
				center *= (1. / approx.size());
				sortCorners(approx, center);
				int w1 = (int)euclideanDistance(approx[0], approx[1]);
				int h1 = (int)euclideanDistance(approx[1], approx[2]);
				Rect r = boundingRect(approx);
				if (r.area() < 5000)continue;
				
				// puntos finales para la transformacion de la imagen
				//int h = r.height, w = r.width;
				int h = h1, w = w1;
				Point2f t1, t2, t3, t4;
				if (r.width > r.height) {
					h = r.width;
					w = r.height;
					t1 = Point2f(w, 0);
					t2 = Point2f(w, h);
					t3 = Point2f(0, h);
					t4 = Point2f(0, 0);
				}
				else {
					t1 = Point2f(0, 0);
					t2 = Point2f(w, 0);
					t3 = Point2f(w, h);
					t4 = Point2f(0, h);
				}
				Mat quad = Mat::zeros(h, w, CV_8UC3);

				vector<Point2f> quad_pts;
				quad_pts.push_back(t1);
				quad_pts.push_back(t2);
				quad_pts.push_back(t3);
				quad_pts.push_back(t4);
				// matriz de transformacion
				Mat transmtx = getPerspectiveTransform(approx, quad_pts);
				// aplicar transformacion de perspectiva 
				warpPerspective(src, quad, transmtx, quad.size());
				stringstream ss;
				ss << i << ".jpg";
				//imshow(ss.str(), quad);
				//waitKey(0);
				getDominoID(quad, dominoID);
				for(int k=0; k<2; k++){
					//std::cout << "Primera Mitad -----------------" << std::endl;
					for(int m=0; m<128; m++){
						dominosID[n_dominos][m] = dominoID[k][m];
						//std::cout << dominosID[n_dominos][m] << std::endl;
						if(m == 64){
						//	std::cout << "Segunda Mitad -----------------" << std::endl;
						}
					}
					n_dominos++;
				}
				//waitKey(0);
			}
			good_contours.push_back(contours[i]);
		}
	}
	std::cout << "Numero de dominos: " << n_dominos/2 << std::endl;
	// Dibujar los contornos correctos
	for (int i = 0; i < good_contours.size(); i++) {
		drawContours(dst, good_contours, i, Scalar(255, 0, 0), 2);
	}

	loadLabelstxt(training_labels, n_dominos);
	saveSVMtxt(training_labels, dominosID, n_dominos);

	namedWindow("DOMINO TABLE", CV_WINDOW_NORMAL);
	imshow("DOMINO TABLE", dst);
	imwrite("detect_domino.jpg",dst);
	waitKey(0);
	
    return 0;
}


// Funcion para ordenar las esquinas del cuadrado, top left, top right, bottom right bottom lefh
void sortCorners(vector<Point2f>& corners, Point2f center) {
	vector<Point2f> top, bot;
	for (int i = 0; i < corners.size(); i++)
	{
		if (corners[i].y < center.y)
			top.push_back(corners[i]);
		else
			bot.push_back(corners[i]);
	}
	sort(top.begin(), top.end(), comparator);
	sort(bot.begin(), bot.end(), comparator);
	Point2f tl = top[0];
	Point2f tr = top[top.size() - 1];
	Point2f bl = bot[0];
	Point2f br = bot[bot.size() - 1];
	corners.clear();
	corners.push_back(tl);
	corners.push_back(tr);
	corners.push_back(br);
	corners.push_back(bl);
}

bool comparator(Point2f a, Point2f b) {
	return a.x<b.x;
}

static double getAngle(Point pt1, Point pt2, Point pt0) {
	double dx1 = pt1.x - pt0.x;
	double dy1 = pt1.y - pt0.y;
	double dx2 = pt2.x - pt0.x;
	double dy2 = pt2.y - pt0.y;
	return (dx1*dx2 + dy1*dy2) / sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
}

double euclideanDistance(Point2f pt1, Point2f pt2){
	return pow(pow(pt1.x - pt2.x,2) + pow(pt1.y - pt2.y,2), 0.5);
}