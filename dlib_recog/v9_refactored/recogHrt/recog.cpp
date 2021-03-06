#include <recogHrt.h>
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
frontal_face_detector detector;
shape_predictor sp;
anet_type net;
std::vector<cv::String> img_path, fileNames; 
int count_img = 0; 
const float thresh = 0.4; 
float vec_error[30]; 
image_window win;
std::string file_name ="";
matrix<rgb_pixel> img_obj; 
std::vector<matrix<float, 0, 1>> face_descriptors_local;  
std::vector<matrix<float,0,1>> face_descriptors;
dlib::rectangle dets;
std::string text = "unknown";
double fps;
double t = 0;
std::string model_path = "../models/";
std::string localfaces_path = "./localfaces/";

int read_local(std::vector<cv::String>& img_path, std::vector<cv::String>& fileNames, int& count_img)
{
    cv::glob("./localfaces",img_path);
    fileNames = img_path;
    for (int i = 0; i < img_path.size(); i++)
    {
        if((img_path[i].find(".jpg") != img_path[i].npos)||(img_path[i].find(".png") != img_path[i].npos))
        {
	    size_t pos = img_path[i].find_last_of('/');
	    size_t len = img_path[i].find_last_of('.');	
	    fileNames[i] = img_path[i].substr(pos+1,len-pos-1);
	    cout << "file name:" << fileNames[i] << endl;
	    count_img++;
	}

    }     
    cout << "[INFO]The number of local picture is:" << count_img << endl; 
    return count_img;
}

void load_models(frontal_face_detector& detector, shape_predictor& sp, anet_type& net)
{
    try
    {
        detector = get_frontal_face_detector();
        deserialize(model_path + "dlib_face_recognition_resnet_model_v1.dat") >> net;    
        deserialize(model_path + "shape_predictor_5_face_landmarks.dat") >> sp;
    }
    catch(exception& e)
    {
        cout << "[ERROR]load models failed" << endl;
        cout << e.what() << endl;
    }
}
void resnet_encoder(std::vector<matrix<float, 0, 1>>& face_descriptors_local, matrix<rgb_pixel>& img_obj)
{
    for (int k = 0; k < count_img; k++)
    {
        string fileFullName = img_path[k];
	load_image(img_obj, fileFullName);
	matrix<float, 0, 1> vec = net(img_obj);
	face_descriptors_local.push_back(vec);
    }
    cout << "[INFO]face_descriptors_local.size():" << face_descriptors_local.size() << endl; 
}
std::vector<rectangle> detect_and_draw(cv_image<bgr_pixel>& cimg, std::vector<matrix<rgb_pixel>>& faces, image_window& win, int sample_true)
{
    int face_counter = 10;
    auto facesDetected = detector(cimg);
    for (auto face : facesDetected)
    {
        auto shape = sp(cimg, face); 
        matrix<rgb_pixel> face_chip; // a basic img for storing the face
        extract_image_chip(cimg, get_face_chip_details(shape,150,0.25), face_chip);
        if (sample_true > 0)
        {
            if (face_counter < 10)
            {
                file_name= localfaces_path +std::to_string(face_counter)+".jpg";
                save_jpeg(face_chip, file_name);
                face_counter++;
            }      
        }   
        faces.push_back(move(face_chip)); // std::move does not copy it, so if u use that latter on that object will be empty
        win.add_overlay(face); // rectangle face
    }
    win.clear_overlay();
    win.set_image(cimg);
    return facesDetected;
}
void compare_faces(const std::vector<matrix<float,0,1>>& face_descriptors, const std::vector<matrix<float,0,1>>& face_descriptors_local, const std::vector<dlib::rectangle>& facesDetected)
{
    for (size_t i = 0; i < face_descriptors.size(); ++i)                
    {
        for (size_t j = 0; j < face_descriptors_local.size(); j++)
        {
            vec_error[j] = (double)length(face_descriptors[i] - face_descriptors_local[j]);
	    if (vec_error[j] < 0.4)
            {
	        text = fileNames[j];
	    } 
            dets = facesDetected[i];
	}
        cout << "Target:" << text <<endl;
        cout << "[l, t, r, b]: "<< facesDetected[i].left() <<","<< facesDetected[i].top() << "," << facesDetected[i].right() << "," << facesDetected[i].bottom() << endl;
        win.add_overlay(dlib::image_window::overlay_rect(dets, rgb_pixel(255,0,0),text ));
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        fps = (1.0 / t);
        std::string fpsString = std::to_string(fps);
        // draw fps
        dlib::rectangle fpsPos(100, 100, 100, 100);
        win.add_overlay(dlib::image_window::overlay_rect(fpsPos, rgb_pixel(0,0,255),fpsString));
    }
}
