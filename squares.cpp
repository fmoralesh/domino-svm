/**********************************************
 * FILE NAME: squares.cpp                     *
 * DESCRIPTION: detects squares in an image   *
 * AUTHORS: alyssaq/opencv                    *
 *                                            *
 *********************************************/
#include "squares.h"

void help()
{
    cout <<
    "\nA program using pyramid scaling, Canny, contours, contour simpification and\n"
    "memory storage to find squares in a list of images\n"
    "Returns sequence of squares detected on the image.\n"
    "the sequence is stored in the specified memory storage\n"
    "Call:\n"
    "./squares\n"
    "Using OpenCV version %s\n" << CV_VERSION << "\n" << endl;
}


int thresh = 50, N = 5;
const char* wndname = "Square Detection Demo";

// helper function:
// finds a cosine of angle between vectors
// from pt0->pt1 and from pt0->pt2
double angle( Point pt1, Point pt2, Point pt0 )
{
    double dx1 = pt1.x - pt0.x;
    double dy1 = pt1.y - pt0.y;
    double dx2 = pt2.x - pt0.x;
    double dy2 = pt2.y - pt0.y;
    return (dx1*dx2 + dy1*dy2)/sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
}

// returns sequence of squares detected on the image.
// the sequence is stored in the specified memory storage
void findSquares( const Mat& timg, vector<vector<Point> >& squares )
{
    squares.clear();

//s    Mat pyr, timg, gray0(image.size(), CV_8U), gray;

    // down-scale and upscale the image to filter out the noise
    //pyrDown(image, pyr, Size(image.cols/2, image.rows/2));
    //pyrUp(pyr, timg, image.size());


    // blur will enhance edge detection
    //Mat timg(image);
    //medianBlur(image, timg, 9);
    Mat gray0(timg.size(), CV_8U), gray;

    vector<vector<Point> > contours;

    // find squares in every color plane of the image
    for( int c = 0; c < 3; c++ )
    {
        int ch[] = {c, 0};
        mixChannels(&timg, 1, &gray0, 1, ch, 1);

        // try several threshold levels
        for( int l = 0; l < N; l++ )
        {
            // hack: use Canny instead of zero threshold level.
            // Canny helps to catch squares with gradient shading
            if( l == 0 )
            {
                // apply Canny. Take the upper threshold from slider
                // and set the lower to 0 (which forces edges merging)
                Canny(gray0, gray, 5, thresh, 5);
                // dilate canny output to remove potential
                // holes between edge segments
                dilate(gray, gray, Mat(), Point(-1,-1));
            }
            else
            {
                // apply threshold if l!=0:
                //     tgray(x,y) = gray(x,y) < (l+1)*255/N ? 255 : 0
                gray = gray0 >= (l+1)*255/N;
            }

            // find contours and store them all as a list
            findContours(gray, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);

            vector<Point> approx;

            // test each contour
            for( size_t i = 0; i < contours.size(); i++ )
            {
                // approximate contour with accuracy proportional
                // to the contour perimeter
                approxPolyDP(Mat(contours[i]), approx, arcLength(Mat(contours[i]), true)*0.02, true);

                // square contours should have 4 vertices after approximation
                // relatively large area (to filter out noisy contours)
                // and be convex.
                // Note: absolute value of an area is used because
                // area may be positive or negative - in accordance with the
                // contour orientation
                if( approx.size() == 4 &&
                    fabs(contourArea(Mat(approx))) > 1000 &&
                    isContourConvex(Mat(approx)) )
                {
                    double maxCosine = 0;

                    for( int j = 2; j < 5; j++ )
                    {
                        // find the maximum cosine of the angle between joint edges
                        double cosine = fabs(angle(approx[j%4], approx[j-2], approx[j-1]));
                        maxCosine = MAX(maxCosine, cosine);
                    }

                    // if cosines of all angles are small
                    // (all angles are ~90 degree) then write quandrange
                    // vertices to resultant sequence
                    if( maxCosine < 0.3 )
                        squares.push_back(approx);
                }
            }
        }
    }
}


// the function draws all the squares in the image
void drawSquares( Mat& image, const vector<vector<Point> >& squares )
{
    for( size_t i = 0; i < squares.size(); i++ )
    {
        const Point* p = &squares[i][0];

        int n = (int)squares[i].size();
        //dont detect the border
        if (p-> x > 3 && p->y > 3)
          polylines(image, &p, &n, 1, true, Scalar(0,255,0), 3, LINE_AA);
    }

    imshow(wndname, image);
}

void getDominoID(cv::Mat domino, float dominoID[][128]){
    int i, j, k, m = 0;
    int counter = 0;

    cv::resize(domino, domino, cv::Size(64, 128));

    // Mat where the Gray image output will be saved
    cv::Mat dominogray(domino.size().height, domino.size().width, CV_8U);
    cv::cvtColor(domino, dominogray, cv::COLOR_RGB2GRAY);

    // Mat where the Gaussian output will be saved
    cv::Mat dominoGauss(dominogray.size().height, dominogray.size().width, CV_8U);
    cv::GaussianBlur(dominogray, dominoGauss, Size(5,5), 0);

    // Mat where the OTSU output will be saved
    cv::Mat dominoOTSU(dominoGauss.size().height, dominoGauss.size().width, CV_8U);
    cv::threshold(dominoGauss, dominoOTSU, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);

    //imshow("OTSU", dominoOTSU);
    // Make the process for first and second half of the domino
    for(k=0; k<2; k++){
        // Make the horizontal sum of values
        for(i=dominoOTSU.size().height*k/2; i<dominoOTSU.size().height*k/2 + dominoOTSU.size().height/2; i++){
            for(j=0; j<dominoOTSU.size().width; j++){
                if(dominoOTSU.at<unsigned char>(i,j) >= 0)
                    counter += (int)dominoOTSU.at<unsigned char>(i,j);
            }
            dominoID[k][i] = counter;
            counter = 0;
        }
        // Make the vertical sum of values
        for(j=0; j<dominoOTSU.size().width; j++){
            for(i= dominoOTSU.size().height*k/2; i<dominoOTSU.size().height*k/2 + dominoOTSU.size().height/2; i++){
                if(dominoOTSU.at<unsigned char>(i,j) >= 0)
                    counter += (int)dominoOTSU.at<unsigned char>(i,j);
            }
            dominoID[k][dominoOTSU.size().height/2+j] = counter;
            counter = 0;
        }
    }

    // Normalizing the two vectors
    float max = 0;
    for(i=0; i<2; i++){
        for(j=0; j<128; j++){
            if(dominoID[i][j] > max)
                max = dominoID[i][j];
        }
        if(max!=0){
            for(j=0; j<128; j++){
                dominoID[i][j] = dominoID[i][j]/max; 
            }
        }
        max = 0;
    }

    // Freeing memory from Mat images
    dominogray.release();
    dominoGauss.release();
    dominoOTSU.release();
}

void loadLabelstxt(int training_labels[100], int n_dominos){
    int i;
    string fileName = "training_labels1.txt";
    
    //if(n_dominos < 25)
    //    fileName = "training_labels10.txt";

    std::ifstream file(fileName);

    if (file.is_open()){
        std::string str;
        for(i=0; i<n_dominos; i++){ 
            file >> str;       
            training_labels[i] = atoi(str.c_str());
        }
        file.close();
    } else{
        std::cout << "Error loading training_labels.txt" << std::endl;
    }
}

void saveSVMtxt(int training_labels[100], float dominosID[][128], int n_dominos){
    int i, k;
    std::ofstream file("domino-for-libsvm");

    // write the histogram descriptor in a file to use with libsvm library
    if (file.is_open()){
        for(i=0; i<n_dominos; i++){
            file << training_labels[i];
            file << " ";
            for(k=0; k < 128; k++){
                file << k << ":";
                file << dominosID[i][k];
                file << " ";
            }
            file << std::endl;
        }
        file.close();
    }else{
        std::cout << "Error saving domino-for-libsvm" << std::endl;
    } 
}