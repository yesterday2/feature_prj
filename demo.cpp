#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <stdlib.h> 
#include <iostream>
#include <ctime>
#include <cstdlib>
#include <chrono>
#include <unistd.h>
class TicToc
{
  public:
    TicToc()
    {
        tic();
    }

    void tic()
    {
        start = std::chrono::system_clock::now();
    }

    double toc()
    {
        end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        return elapsed_seconds.count() * 1000;
    }

  private:
    std::chrono::time_point<std::chrono::system_clock> start, end;
};

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

int main(int argc, char* argv[])
{
    if(argc != 3)
    {
        cout << "wrong use" << endl;
    }
	chdir(".");
	//char *buffer = getcwd(NULL, 0);
    string a = argv[0];
    string DataName = argv[1];
    int method = atoi(argv[2]);

    string FilePath = "../datasets";
	//cout<<"buffer := "<< buffer<< "argv[0]:= "<< argv[0] <<"argv[1]:= "<<DataName<<endl;
    //设置特征提取
    //生成特征点算法及其匹配方法
	Ptr<Feature2D>  extractor;
	BFMatcher matcher;
	switch (method)
	{
		case 0: //"SIFT"
			extractor= SIFT::create();
			matcher = BFMatcher(NORM_L2);	
            cout << "sift" << endl;
			break;
		case 1: //"SURF"
			extractor= SURF::create();
			matcher = BFMatcher(NORM_L2);
            cout << "surf" << endl;	
			break;
		case 2: //"BRISK"
			extractor = BRISK::create();
			matcher = BFMatcher(NORM_HAMMING);
            cout << "brisk" << endl;
			break;
		case 3: //"ORB"
			extractor= ORB::create();
			matcher = BFMatcher(NORM_HAMMING);	
            cout << "orb" << endl;
			break;
		case 4: //"FREAK"
			//extractor= FREAK::create();
			matcher = BFMatcher(NORM_HAMMING);
            cout << "freak" << endl;
			break;
	}

int innersize = 0;
int index = 1;

    //读取图片并进行匹配
    for(int i = 2; i < 7; i++)
	{
		TicToc t;

		string Image1Path = FilePath + "/" + DataName + "/" + "img1" + ".png";
		string Image2Path = FilePath + "/" + DataName + "/" + "img" + to_string(i) + ".png";

		cout << Image1Path << "," << Image2Path << endl;
        
		//读取图片
		Mat img1 = imread(Image1Path, 0);
		Mat img2 = imread(Image2Path, 0);

	    Mat descriptors1;  
	    std::vector<KeyPoint> keypoints1;
	    Mat descriptors2;
	    std::vector<KeyPoint> keypoints2;

	    std::vector< DMatch > matches;
	    std::vector< DMatch > good_matches;
        
		//提取匹配
		if(method != 4)
		{
		   extractor->detectAndCompute(img1,Mat(),keypoints1,descriptors1);
		   extractor->detectAndCompute(img2,Mat(),keypoints2,descriptors2);
		}
		else
		{
			auto freak = FREAK::create();

			Ptr<Feature2D>  extractor = ORB::create();
			extractor->detect(img1, keypoints1);
			extractor->detect(img2, keypoints2);

			freak->compute(img1, keypoints1, descriptors1);
			freak->compute(img2, keypoints2, descriptors2);
		}

		matcher.match( descriptors1, descriptors2, matches );

		//阈值匹配
		double max_dist = 0; 
		double min_dist = 100;
		for( int a = 0; a < matches.size(); a++ )
		{
			double dist = matches[a].distance;
			if( dist < min_dist ) min_dist = dist;
			if( dist > max_dist ) max_dist = dist;
		}

		for( int a = 0; a < matches.size(); a++ )
		{ 
			if( matches[a].distance <= max(2*min_dist, 30.0) )
				good_matches.push_back( matches[a]); 
		}

		if (good_matches.size()<4)
		{   
			cout << i << "th image can not match" << endl;
			cout<<" 有效特征点数目小于4个，粗匹配失败 "<<endl;
			continue;
		}
        
		//单应性矩阵
		std::vector<Point2f> obj;
		std::vector<Point2f> scene;
		for( int a = 0; a < (int)good_matches.size(); a++ )
		{    
			obj.push_back( keypoints1[good_matches[a].queryIdx ].pt );
			scene.push_back( keypoints2[good_matches[a].trainIdx ].pt );
		}

		vector<uchar>inliers;
        Mat H;
		H = findHomography(obj, scene,inliers,8);
		
        std::vector<DMatch> good_matches_ransac;
		
int num = 0;
		for(int i = 0; i < inliers.size(); i++)
		{
			if(inliers[i])
			{
				num++;
				good_matches_ransac.push_back(good_matches[i]);
			}
		}

		cout << num << " points have matched" << endl;
		Mat matTmp;
		drawMatches(img1,keypoints1,img2,keypoints2,good_matches_ransac,matTmp);
		string Path = FilePath + "/" + DataName + "/result" + to_string(index++) + ".png";
        imwrite(Path,matTmp);

		cout << t.toc() << " ms" << endl;

				//计算内点数目
				Mat matObj;
				Mat matScene;
				CvMat H_temp=(CvMat)H;
				CvMat* pcvMat = &H_temp;
				const double* Hmodel = pcvMat->data.db;
				double Htmp = Hmodel[6];
				for( int isize = 0; isize < obj.size(); isize++ )
				{
					double ww = 1./(Hmodel[6]*obj[isize].x + Hmodel[7]*obj[isize].y + 1.);
					double dx = (Hmodel[0]*obj[isize].x + Hmodel[1]*obj[isize].y + Hmodel[2])*ww - scene[isize].x;
					double dy = (Hmodel[3]*obj[isize].x + Hmodel[4]*obj[isize].y + Hmodel[5])*ww - scene[isize].y;
					float err = (float)(dx*dx + dy*dy); //3个像素之内认为是同一个点
					if (err< 9)
					{
						innersize = innersize+1;
					}
				}
				//打印内点占全部特征点的比率
				float ff = (float)innersize / (float)good_matches.size();
				cout<<"inner point rate is "<<ff<<endl;
                     ff = 0;
		     innersize = 0;
		     matches.clear();
		     good_matches.clear(); 


          }


    return 0;

}
