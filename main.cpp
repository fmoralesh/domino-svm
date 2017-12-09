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
#include <cmath>
#include "squares.h"

using namespace cv;
using namespace std;

static double getAngle(Point pt1, Point pt2, Point pt0);
void sortCorners(vector<Point2f>& corners, Point2f center);
bool comparator(Point2f a, Point2f b);


int main() {
	int i, j;
	int n_dominos=0;
	float dominosID[100][185];
	for(i=0; i<100; i++){
		for(j=0; j<185; j++){
			dominosID[i][j] = 0;
		}
	}
    // Reading the image.
	Mat src = imread("domino.jpeg");
	if (src.empty())
		return -1;

    std::vector<vector<Point> > squares;
    //findSquares(src, squares);
    //drawSquares(src, squares);
	vector< vector<Point> > good_contours;
	vector< Point > corners;

	// convertir a escala de grises
	Mat gray;
	cvtColor(src, gray, CV_BGR2GRAY);

	// filtro canny de bordes
	Mat bw;
	Canny(gray, bw, 0, 50, 5);

	// Encontrar contornos
	vector<vector<Point> > contours;
	findContours(bw.clone(), contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

	vector<Point2f> approx;
	Mat dst = src.clone();

	for (int i = 0; i < contours.size(); i++)
	{
		// Se aproximan los contornos a figuras simples
		// a partir de su perimetro
		approxPolyDP(Mat(contours[i]), approx, arcLength(Mat(contours[i]), true)*0.02, true);

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
			if (mincos >= -0.1 && maxcos <= 0.3) {
				float dominoID[2][185];
				for(int i=0; i<2; i++){
					for(int j=0; j<185; j++){
						dominoID[i][j] = 0;
					}
				}
				Point2f center(0, 0);
				for (int i = 0; i < approx.size(); i++) {
					center += approx[i];
				}
				center *= (1. / approx.size());
				sortCorners(approx, center);

				Rect r = boundingRect(approx);
				if (r.area() < 5000)continue;
				cout << r.area() << endl;
				// puntos finales para la transformacion de la imagen
				int h = r.height, w = r.width;
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

				getDominoID(quad, dominoID);
				for(int i=0; i<2; i++){
					for(int j=0; j<185; j++){
						dominosID[n_dominos][j] = dominoID[i][j];
						std::cout << dominosID[n_dominos][j] << std::endl;
					}
					n_dominos++;
				}
				
				waitKey(0);
			}
			good_contours.push_back(contours[i]);
		}
	}
	std::cout << "Numero de dominos: " << n_dominos << std::endl;
	// Dibujar los contornos correctos
	for (int i = 0; i < good_contours.size(); i++) {
		drawContours(dst, good_contours, i, Scalar(255, 0, 0), 2);
	}
	
	imshow("src", src);
	imshow("dst", dst);
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