#ifndef RECOG_HRT
#define RECOG_HRT
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
using namespace dlib;
using namespace std;
template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = add_prev1<block<N,BN,1,tag1<SUBNET>>>;

template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = add_prev2<avg_pool<2,2,2,2,skip1<tag2<block<N,BN,2,tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET> 
using block  = BN<con<N,3,3,1,1,relu<BN<con<N,3,3,stride,stride,SUBNET>>>>>;

template <int N, typename SUBNET> using ares      = relu<residual<block,N,affine,SUBNET>>;
template <int N, typename SUBNET> using ares_down = relu<residual_down<block,N,affine,SUBNET>>;

template <typename SUBNET> using alevel0 = ares_down<256,SUBNET>;
template <typename SUBNET> using alevel1 = ares<256,ares<256,ares_down<256,SUBNET>>>;
template <typename SUBNET> using alevel2 = ares<128,ares<128,ares_down<128,SUBNET>>>;
template <typename SUBNET> using alevel3 = ares<64,ares<64,ares<64,ares_down<64,SUBNET>>>>;
template <typename SUBNET> using alevel4 = ares<32,ares<32,ares<32,SUBNET>>>;

using anet_type = loss_metric<fc_no_bias<128,avg_pool_everything<
                            alevel0<
                            alevel1<
                            alevel2<
                            alevel3<
                            alevel4<
                            max_pool<3,3,2,2,relu<affine<con<32,7,7,2,2,
                            input_rgb_image_sized<150>
                            >>>>>>>>>>>>;


extern frontal_face_detector detector;
extern shape_predictor sp;
extern anet_type net;
extern std::vector<cv::String> img_path, fileNames; 
extern int count_img; 

extern float vec_error[30]; 
extern image_window win;
extern std::string file_name;
extern matrix<rgb_pixel> img_obj; 
extern std::vector<matrix<float, 0, 1>> face_descriptors_local;  
extern std::vector<matrix<float,0,1>> face_descriptors;
extern dlib::rectangle dets;
extern std::string text;
extern double fps;
extern double t;
extern std::string model_path;
extern std::string localfaces_path;

int read_local(std::vector<cv::String>& img_path, std::vector<cv::String>& fileNames, int& count_img);
void load_models(frontal_face_detector& detector, shape_predictor& sp, anet_type& net);
void resnet_encoder(std::vector<matrix<float, 0, 1>>& face_descriptors_local, matrix<rgb_pixel>& img_obj);
std::vector<rectangle> detect_and_draw(cv_image<bgr_pixel>& cimg, std::vector<matrix<rgb_pixel>>& faces, image_window& win, int sample_true);
void compare_faces(const std::vector<matrix<float,0,1>>& face_descriptors, const std::vector<matrix<float,0,1>>& face_descriptors_local, const std::vector<dlib::rectangle>& facesDetected);
#endif
