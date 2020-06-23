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
        
        

        int face_counter = 0;
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

            
            
            std::vector<matrix<rgb_pixel>> faces;
            matrix<rgb_pixel> imglocal; // array2d and matrix, the same when saving
            std::vector<std::string> locals = {"localfaces"}; // local faces dir 
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
            
                    // the following comments use opencv to save ROI, but dlib use facechips
                    //cv::Mat mat_img = dlib::toMat(cimg);
                    //cv::Mat roi(mat_img, cv::Rect(face.left(),face.top(),face.bottom()-face.top(),face.right()-face.left()));
                    //cv::imwrite(file_name, roi);
                    //cv::waitKey(0);
                    //cout << file_name << endl;
                    //cout << face<< face.left() <<face.right()<<face.bottom()<<face.top() << endl;
                    save_jpeg(face_chip, file_name);
                    face_counter++;
                }     
                faces.push_back(move(face_chip)); // std::move does not copy it, so if u use that latter on that object will be empty

                win.add_overlay(face); // rectangle face



                // read local faces here  
                
                load_image(imglocal,"./localfaces/9.jpg");  
                faces.push_back(move(imglocal)); // local joins the cluster
 


                  
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
            //cout << face_descriptors.size() << endl;

         
                     



            for(auto iter=locals.begin(); iter!=locals.end(); iter++ )
            {
               std::vector<sample_pair> edges;
               for (size_t i = 0; i < face_descriptors.size(); ++i)
               {
                   for (size_t j = i; j < face_descriptors.size(); ++j)
                   {
                       if (length(face_descriptors[i]-face_descriptors[j]) < 0.6)
                          
                           edges.push_back(sample_pair(i,j));
                   }
               }
               std::vector<unsigned long> labels;
               const auto num_clusters = chinese_whispers(edges, labels);
               cout<< "numbers: " << num_clusters << endl;
               //if (faces.size()>0)
               //{
                   size_t detected = std::abs((int)faces.size() - (int)localFaceCount);
                   
                   cout<<"faces.size()"<<(int)faces.size()<<" detected:"<<detected <<endl;
              // }
               //else
                  // cout<<"faces.size is zero"<<endl;
               
               if( detected >= num_clusters && num_clusters>0) 
               {
                   for (size_t i = 0;  i < detected; ++i)
                   {
                          if (labels[i] == labels[detected+1])
                              cout<< "Hello, " << (*iter) <<facesDetected[i]<<endl;
                   }
               } 
            }
     	    
	
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
