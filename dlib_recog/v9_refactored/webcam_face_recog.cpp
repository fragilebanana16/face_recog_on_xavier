#include <dlib/opencv.h>
#include <dlib/dnn.h>
#include <dlib/clustering.h>
#include <dlib/string.h>
#include <dlib/image_io.h>
#include <opencv2/highgui/highgui.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <cmath>
#include <stdio.h>
#include <stdlib.h>

#include <recogHrt.h>
using namespace dlib;
using namespace std;
int main()
{

    try
    {
        cv::VideoCapture cap(0);
        if (!cap.isOpened())
        {
            cerr << "[ERROR]Unable to connect to camera" << endl;
            return -1;
        }
        // load models
        load_models(detector, sp, net);
        // read locals 
        if(read_local(img_path, fileNames, count_img)<0)
        {
            cerr << "[ERROR]invalid local images" << endl;
            return -1;
        }
        // encode locals
        resnet_encoder(face_descriptors_local, img_obj);
        // Grab and process frames until the main window is closed by the user.
        while(!win.is_closed())
        {
            t = (double)cv::getTickCount();
            // Grab a frame
            cv::Mat temp;
            if (!cap.read(temp))
                break;
            cv_image<bgr_pixel> cimg(temp);
            std::vector<matrix<rgb_pixel>> faces; // facechips
            int sample_true = 1;
            std::vector<dlib::rectangle> facesDetected = detect_and_draw(cimg, faces, win, sample_true);
            face_descriptors = net(faces);
            compare_faces(face_descriptors, face_descriptors_local, facesDetected);
        }
    }
    catch(serialization_error& e)
    {
        cout << "You need dlib's default face landmarking model file to run this example." << endl;
        cout << "You can get it from the following URL: " << endl;
        cout << "   http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" << endl;
        cout << endl << e.what() << endl;
    }
    catch(exception& e)
    {
        cout << e.what() << endl;
    }
}
