#include <opencv2/core/utility.hpp>
#include <opencv2/saliency.hpp> // contrib/modules/saliency
#include <opencv2/highgui.hpp>
#include <iostream>
#include <ctime>
using namespace std;
using namespace cv;
using namespace saliency;

void spectral_residual(Mat& image, Mat& binaryMap, Ptr<Saliency>& saliencyAlgorithm)
{
  Mat saliencyMap;
  saliencyAlgorithm = StaticSaliencySpectralResidual::create();
  if( saliencyAlgorithm->computeSaliency( image, saliencyMap ) )
  {
    StaticSaliencySpectralResidual spec;
    spec.computeBinaryMap( saliencyMap, binaryMap );

    //imshow( "Saliency Map", saliencyMap );
    //imshow( "Original Image", image );
    //imshow( "Binary Map", binaryMap );
    //waitKey( 0 );
  }
}
void fine_grained(Mat& image, Mat& binaryMap, Ptr<Saliency>& saliencyAlgorithm)
{
  Mat saliencyMap;
  saliencyAlgorithm = StaticSaliencyFineGrained::create();
  saliencyAlgorithm->computeSaliency( image, saliencyMap );
  
    //imshow( "Saliency Map", saliencyMap );
    //imshow( "Original Image", image );
    //waitKey( 0 );
  
}

int bing(Mat& image, Mat& binaryMap, Ptr<Saliency>& saliencyAlgorithm, String training_path)
{
  if( training_path.empty() )
  {
    cout << "Path of trained files missing! " << endl;
    return -1;
  }
  else
  {
    saliencyAlgorithm = ObjectnessBING::create();
    vector<Vec4i> saliencyMap;
    saliencyAlgorithm.dynamicCast<ObjectnessBING>()->setTrainingPath( training_path );
    saliencyAlgorithm.dynamicCast<ObjectnessBING>()->setBBResDir( "Results" );

    if( saliencyAlgorithm->computeSaliency( image, saliencyMap ) )
    {
      int ndet = int(saliencyMap.size());
      std::cout << "Objectness done " << ndet << std::endl;
      // The result are sorted by objectness. We only use the first maxd boxes here.
      int maxd = 7, step = 255 / maxd, jitter=9; // jitter to separate single rects
      Mat draw = image.clone();
      for (int i = 0; i < std::min(maxd, ndet); i++) {
        Vec4i bb = saliencyMap[i];
        Scalar col = Scalar(((i*step)%255), 50, 255-((i*step)%255));
        Point off(theRNG().uniform(-jitter,jitter), theRNG().uniform(-jitter,jitter));
        //rectangle(draw, Point(bb[0]+off.x, bb[1]+off.y), Point(bb[2]+off.x, bb[3]+off.y), col, 2);
        //rectangle(draw, Rect(20, 20+i*10, 10,10), col, -1); // mini temperature scale
      }
      //imshow("BING", draw);
      //waitKey();
    }
  }
}
int motion_saliency(Mat& binaryMap, Ptr<Saliency>& saliencyAlgorithm)
{
  VideoCapture cap;
  cap.open( "v2_18.mp4" );
  
  
  VideoWriter Writer("cut_out.avi", CV_FOURCC('M', 'J', 'P', 'G'), 25, Size(cap.get(CAP_PROP_FRAME_WIDTH), cap.get(CAP_PROP_FRAME_HEIGHT)), true);

  Mat frame;
  Mat dst;
  saliencyAlgorithm = MotionSaliencyBinWangApr2014::create();
  saliencyAlgorithm.dynamicCast<MotionSaliencyBinWangApr2014>()->setImagesize(cap.get(CAP_PROP_FRAME_WIDTH), cap.get(CAP_PROP_FRAME_HEIGHT));
  saliencyAlgorithm.dynamicCast<MotionSaliencyBinWangApr2014>()->init();

  bool paused = false;
  for ( ;; )
  {
    if( !paused )
    {
      
      cap >> frame;
      resize(frame, dst, Size(cap.get(CAP_PROP_FRAME_WIDTH), cap.get(CAP_PROP_FRAME_HEIGHT)));
      if( frame.empty() )
      {
        return 0;
      }
      cvtColor( dst, dst, COLOR_BGR2GRAY );

      Mat saliencyMap;
      clock_t time_start=clock();
      saliencyAlgorithm->computeSaliency( dst, saliencyMap );
      clock_t time_end=clock();
      cout<<"time use:"<<1000*(time_end-time_start)/(double)CLOCKS_PER_SEC<<"ms"<<endl;

      cvtColor( saliencyMap, saliencyMap, COLOR_GRAY2BGR );
     // resize(saliencyMap, saliencyMap, Size(300, 200));
      Writer.write(saliencyMap * 255);
      //imshow( "image", frame );
      
      //imshow( "saliencyMap", saliencyMap * 255 );
    }

    char c = (char) waitKey( 2 );
    if( c == 'q' )
      break;
    if( c == 'p' )
      paused = !paused;

  }
}
int main( int argc, char** argv )
{
  Mat image = imread(argv[1]);
  //instantiates the specific Saliency， Ptr :Template class for smart pointers with shared ownership. 
  Ptr<Saliency> saliencyAlgorithm;
  Mat binaryMap;
  String model_path = "/home/appltini/opencv_3.3.1/opencv_contrib-3.3.1/modules/saliency/samples/ObjectnessTrainedModel";
  switch (argv[2][0]){
    case 's':
      while(1){
        clock_t time_start=clock();
        spectral_residual(image, binaryMap, saliencyAlgorithm);
        clock_t time_end=clock();
        cout<<"time use:"<<1000*(time_end-time_start)/(double)CLOCKS_PER_SEC<<"ms"<<endl;
      }
      break;
    case 'f':
      while(1){
        clock_t time_start=clock();
        fine_grained(image, binaryMap, saliencyAlgorithm);
        clock_t time_end=clock();
        cout<<"time use:"<<1000*(time_end-time_start)/(double)CLOCKS_PER_SEC<<"ms"<<endl;
      }
      break;
    case 'b':
      while(1){
        clock_t time_start=clock();
        bing(image, binaryMap, saliencyAlgorithm, model_path);
        clock_t time_end=clock();
        cout<<"time use:"<<1000*(time_end-time_start)/(double)CLOCKS_PER_SEC<<"ms"<<endl;
      }
      break;
    case 'm':
      motion_saliency(binaryMap, saliencyAlgorithm);
      break;
  }
  return 0;
}
