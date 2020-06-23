#include <stdio.h>
#include <stdint.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <dface_detect.h>
#include <mat.h>
#include <symbol.h>
#include <list>
#include <types.h>
#include <chrono>


using namespace cv;
using namespace std;
// double t = (double)getTickCount();
// t = ((double)getTickCount() - t)/getTickFrequency();
// cout << "Times passed in seconds: " << t << endl;

string DFACE_MODEL_PATH = "/home/appltini/Desktop/myOwnAlogrithm/dface_usage/model/normal_binary";
int main(int argc, char** argv )
{

    if ( argc != 2 )
    {
        cout << "usage: DisplayImage.out <Image_Path>" << endl;
        return -1;
    }
    
    Mat image;
    image = imread( argv[1], 1 );
    if ( !image.data )
    {
        printf("No image data \n");
        return -1;
    }
    dface::Mat aMat;
    dface::DfaceDetect *dfd = NULL;

    dfd = load_dface_detect(DFACE_MODEL_PATH);
    
   
    dfd->SetMinSize(30);
    dfd->SetNumThreads(2);
    //dfd->setWorkMode(1);
    vector<dface::Box> bbox;
    //undefined reference, some lib not included error
    //aMat = dface::Mat::from_pixels(image.data, dface::Mat::PIXEL_RGB, image.cols, image.rows);
    while(1)
    {
        //double t = 0;
        const auto t1 = std::chrono::system_clock::now();


        dfd->detection(image.data, image.cols, image.rows, dface::PIXELS_RGB, bbox);
        for (auto iter : bbox)
        {
            rectangle(image, Point(iter.x, iter.y), Point(iter.x+iter.w, iter.y+iter.h), Scalar(0, 255, 255));
            cout << "vector:(" << iter.x << "," << iter.y << "," << iter.w << "," << iter.h << ")" << endl;
        }
       const auto t2 = std::chrono::system_clock::now();
       const auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

       std::cout << "time consumed "<<duration * 1e-3<<std::endl;
        //t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        //cout << t << endl;
       namedWindow("Display Image", WINDOW_AUTOSIZE );
       imshow("Display Image", image);
       waitKey(0);

    }

    return 0;
}
