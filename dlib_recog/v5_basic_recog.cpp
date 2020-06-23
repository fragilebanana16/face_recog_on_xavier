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

int main()
{
    try
    {
        cv::VideoCapture cap(0);
        if (!cap.isOpened())
        {
            cerr << "Unable to connect to camera" << endl;
            return 1;
        }

        image_window win;

        // Load face detection and pose estimation models.
        frontal_face_detector detector = get_frontal_face_detector();
        
	shape_predictor sp;
	deserialize("../models/shape_predictor_5_face_landmarks.dat") >> sp;
        // And finally we load the DNN responsible for face recognition.
        anet_type net;
        deserialize("../models/dlib_face_recognition_resnet_model_v1.dat") >> net;        
        std::vector<matrix<float, 0, 1>> face_descriptors_local;        //定义一个向量组，用于存放每一个人脸的编码；
        float thresh = 0.4;
	float vec_error[30];int count_img = 0;                      
        // locals 
	std::vector<cv::String> fileNames,img_path;
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
	cout << "The number of local picture is:" << count_img << endl; 
              
        matrix<rgb_pixel> img_obj; // temp       
	for (int k = 0; k < count_img; k++)  //依次加载完图片库里的文件
	{
            string fileFullName = img_path[k];//图片地址+文件名
	    load_image(img_obj, fileFullName);
	    matrix<float, 0, 1> vec = net(img_obj);//载入Resnet残差网络，返回128D人脸特征
	    face_descriptors_local.push_back(vec);//保存这一个人脸的特征向量到vec向量的对应位置
	}

        cout << "face_descriptors_local.size():" << face_descriptors_local.size() << endl; 
     	            

        int face_counter = 10;
        std::string file_name ="";
        cv::Mat save_mat; 
        const size_t localFaceCount = 1;
        // Grab and process frames until the main window is closed by the user.
        while(!win.is_closed())
        {
            // Grab a frame
            cv::Mat temp;
            if (!cap.read(temp))
            {
                break;
            }
            // Turn OpenCV's Mat into something dlib can deal with.  Note that this just
            // wraps the Mat object, it doesn't copy anything.  So cimg is only valid as
            // long as temp is valid.  Also don't do anything to temp that would cause it
            // to reallocate the memory which stores the image as that will make cimg
            // contain dangling pointers.  This basically means you shouldn't modify temp
            // while using cimg.
            cv_image<bgr_pixel> cimg(temp);

            
            
            std::vector<matrix<rgb_pixel>> faces; // facechips
            matrix<rgb_pixel> imglocal; // array2d and matrix, the same when saving
            
            auto facesDetected = detector(cimg);
            for (auto face : facesDetected)
            {
                auto shape = sp(cimg, face); // refer to the override of the '()', returns locations of landmarks, face is just a rectangle, so we need a rect and a img together to locate the face
                matrix<rgb_pixel> face_chip; // a basic img for storing the face
                extract_image_chip(cimg, get_face_chip_details(shape,150,0.25), face_chip);
// extract to face chips, also known as ROI with size 150, padding 0.25, not the face_chip is a reference and note its data type
                // save face ROIs
                if (face_counter < 10)
                {
                    file_name="./localfaces/"+std::to_string(face_counter)+".jpg";
                    save_jpeg(face_chip, file_name);
                    face_counter++;
                }     
                faces.push_back(move(face_chip)); // std::move does not copy it, so if u use that latter on that object will be empty

                win.add_overlay(face); // rectangle face



                // read local faces here  
                
                //load_image(imglocal,"./localfaces/9.jpg");  
                //faces.push_back(move(imglocal)); // local joins the cluster
 


                  
                //cout << "ok here" << endl;
                    
            }
            if (faces.size() == 0)
            {
                cout << "No faces found in image!" << endl;
                //return 1;
            }	
            //else
            //{
            //    cout << faces.size() << endl;
            //}	

            win.clear_overlay();
            win.set_image(cimg);

            
            std::vector<matrix<float,0,1>> face_descriptors = net(faces);
            cout << "face_descriptors.size(): "<<face_descriptors.size() << endl;
            dlib::rectangle dets;
            std::string text = "unknown";
            for (size_t i = 0; i < face_descriptors.size(); ++i)                 //比对，识别
	    {
		for (size_t j = 0; j < face_descriptors_local.size(); j++)
                {
		    vec_error[j] = (double)length(face_descriptors[i] - face_descriptors_local[j]);
		    //cout <<face_descriptors.size()<< ":pic_"<<j<<"->pic_"<<i<<" vec_error is:" << vec_error[j] << endl;
		    
                    
		    if (vec_error[j] < 0.4)
                    {//多个人就找出一个阀值以下的
                        //thresh = vec_error[j];
			text = fileNames[j];//得到文件名
			
		    } 
                    
		    
		    dets = facesDetected[i];
		    //ft2->putText(mimg, text, origin, 40/*size*/,Scalar(255,0,0), -1, 8, true );
	        }
                cout <<"most likely foundto face"<<i<<".jpg"<<text<<endl;
                win.add_overlay(dlib::image_window::overlay_rect(dets, rgb_pixel(255,0,0),text ));
		    //cv::rectangle(mimg, cv::Rect(dets_test[i].left(), dets_test[i].top(), dets_test[i].width(), dets_test[i].width()), cv::Scalar(0, 0, 255), 1, 1, 0);//画矩形框
	    }
		//dlib::cv_image<bgr_pixel> dimg(mimg);//转成dlib格式
	    //win.add_overlay(face);
            
	    //win.wait_until_closed();
 
	
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
