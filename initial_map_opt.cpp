#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <cvsba/cvsba.h>
#include <string>
#include <sstream>
#include <unistd.h>
#include <ctime>
#include <math.h>
#include <pcl/common/common_headers.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/Geometry>
#include <opencv2/xfeatures2d.hpp>
#define MIN_NUM_FEAT 1000

using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;

cv::Mat loadImage(std::string _folder, int _number, cv::Mat &_intrinsics, cv::Mat &_coeffs) {
	stringstream ss; /*convertimos el entero l a string para poder establecerlo como nombre de la captura*/
	ss << _folder << "/left_" << _number << ".png";
	std::cout << "Loading image: " << ss.str() << std::endl;
	Mat image = imread(ss.str(), CV_LOAD_IMAGE_COLOR);
	cvtColor(image, image, COLOR_BGR2GRAY);
	cv::Mat image_u;
	undistort(image, image_u, _intrinsics, _coeffs);
	return image_u;
}
void displayMatches(	cv::Mat &_img1, std::vector<cv::Point2f> &_features1, std::vector<uchar> &_filtered1,
						cv::Mat &_img2, std::vector<cv::Point2f> &_features2){
	cv::Mat display;
	cv::hconcat(_img1, _img2, display);
	cv::cvtColor(display, display, CV_GRAY2BGR);

	for(unsigned i = 0; i < _filtered1.size(); i++){
        if(_filtered1[i])
        {
		    auto p1 = _features1[i];
		    auto p2 = _features2[i] + cv::Point2f(_img1.cols, 0);
		    cv::circle(display, p1, 2, cv::Scalar(0,255,0),2);
		    cv::circle(display, p2, 2, cv::Scalar(0,255,0),2);
		    cv::line(display,p1, p2, cv::Scalar(0,255,0),1);
        }
	}

	cv::imshow("display", display);
	cv::waitKey(3);
}
void featureTracking(Mat img_1, Mat img_2, vector<Point2f>& points1, vector<Point2f>& points2, vector<uchar>& status)	{ 
    //this function automatically gets rid of points for which tracking fails
    vector<uchar> mask;
    vector<float> err;					
    Size winSize=Size(21,21);																								
    TermCriteria termcrit=TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01);
    calcOpticalFlowPyrLK(img_1, img_2, points1, points2, status, err, winSize, 3, termcrit, 0, 0.001);
    //getting rid of points for which the KLT tracking failed or those who have gone outside the frame
    vector<int> i_keypoint;
    vector<Point2f> source, destination;
    for (int i=0;i<status.size();i++)
    {   
        Point2f pt = points2[i];
        if(status[i]!=0 && pt.x>0 && pt.y>0)
        {
            i_keypoint.push_back(i);
            source.push_back(points1[i]);
            destination.push_back(pt);
        }
    }
    //additional RANSAC filter
    findFundamentalMat(source, destination, FM_RANSAC, 1.0, 0.99, mask);
	for (int m = 0; m < mask.size(); m++)
	{
		if (!mask[m])
		{
            status[i_keypoint[m]]=0;
		}
	}
}
void featureDetection(Mat img_1, vector<Point2f>& points1)	{   //uses FAST as of now, modify parameters as necessary
  vector<KeyPoint> keypoints_1;
  auto pt=ORB::create(1000);
  pt->detect(img_1,keypoints_1);
  KeyPoint::convert(keypoints_1, points1, vector<int>());
}

void add_new_point_to_map(unordered_map<int,vector<Point2f>> &points_2d,unordered_map<int,vector<int>> &imgs_index,unordered_map<int,vector<int>> &match_idx,Point2f pta,Point2f ptb,int curr_img,int &ident,int status_index)
{
    points_2d[ident]=vector<Point2f>();
    points_2d[ident].push_back(pta);
    imgs_index[ident]=vector<int>();
    imgs_index[ident].push_back(curr_img-1);
    match_idx[ident]=vector<int>();
    match_idx[ident].push_back(status_index);
    points_2d[ident].push_back(ptb);
    imgs_index[ident].push_back(curr_img);
    match_idx[ident].push_back(status_index);
    ident++;
}
void update_point_from_map(unordered_map<int,vector<Point2f>> &points_2d,unordered_map<int,vector<int>> &imgs_index,unordered_map<int,vector<int>> &match_idx,Point2f pt,int curr_img,int ident,int status_index)
{
    points_2d[ident].push_back(pt);
    imgs_index[ident].push_back(curr_img);
    match_idx[ident].push_back(status_index);
}

int main(int argc,char ** argv)
{
    //init calibration matrices
    Mat distcoef = (Mat_<float>(1, 5) << 0.2624, -0.9531, -0.0054, 0.0026, 1.1633);
	Mat distor = (Mat_<float>(5, 1) << 0.2624, -0.9531, -0.0054, 0.0026, 1.1633);
	distor.convertTo(distor, CV_64F);
	Mat intrinseca = (Mat_<float>(3, 3) << 517.3, 0., 318.6, 0., 516.5, 255.3, 0., 0., 1.);
	intrinseca.convertTo(intrinseca, CV_64F);
	distcoef.convertTo(distcoef, CV_64F);
    //number of images to compose the initial map
    int nImages =  atoi(argv[2]);
    //map for 3d points projections
    unordered_map<int,vector<Point2f>> pt_2d;
    //map for image index of 3d points projections;
    unordered_map<int,vector<int>> img_index;
    //map for match index
    unordered_map<int,vector<int>> match_index;
    int l=0;
    //identifier for 3d_point
    int ident=0;
    //we need variables to store the last image and the last features & descriptors
    Mat foto1_u;
    vector<Point2f> features1;
    //to store the used features of the previous frame
    vector<uchar> last_status;
    //load first image
    foto1_u = loadImage(argv[1], l, intrinseca, distcoef);
    featureDetection(foto1_u,features1);
    l++;
    int start=1;
    while(l<nImages)
    {   
        //load new image
        Mat foto2_u = loadImage(argv[1], l, intrinseca, distcoef);
        //feature tracking by optical flow
        vector<uchar> status;
        vector<Point2f> features2;
        featureTracking(foto1_u,foto2_u,features1,features2,status);
        displayMatches(foto1_u,features1,status,foto2_u,features2);
        //at first iteration we add all the features to map
        if(start)
        {
            for(int i=0;i<status.size();i++)
            {
                if(status[i])
                {
                    add_new_point_to_map(pt_2d,img_index,match_index,features1[i],features2[i],l,ident,i);
                }
            }
            foto1_u=foto2_u;
            last_status=status;
            features1=features2;
            start=0;
            l++;
            continue;
        }
        //are we tracking the same point?¿
        for(int i=0;i<status.size();i++)
        {
            //do we use the same feature between frames l-2|l-1 and l-1|l?
            if(status[i]==last_status[i])
            {
                int flag=0;
                for(int j=0;j<ident && !flag;j++)
                {
                    auto search_match=match_index.find(j);
                    auto search_img=img_index.find(j);
                    if(search_match!=match_index.end() && search_img!=img_index.end())
                    {
                        auto it_match=search_match->second.end();
                        it_match--;
                        auto it_img=search_img->second.end();
                        it_img--;
                        int last_match=*it_match;
                        int last_img=*it_img;
                        if(last_match==i && last_img==l-1)
                        {
                            update_point_from_map(pt_2d,img_index,match_index,features2[i],l,j,i);
                            flag=1;
                        }
                    }
                }
            }
        }
        //Do we have enough points?
        int cnt=0;
        int status_size=0;
        for(int i=0;i<status.size();i++)
        {
            status_size++;
            if(status[i])
            {
                cnt++;
            }
        }
        //if we dont have enough features we have to add more and then update status vector with the new points
        if(cnt<MIN_NUM_FEAT)
        {
            vector<uchar> status_aux;
            vector<Point2f> features1_aux,features2_aux;
            featureDetection(foto1_u,features1_aux);
            featureTracking(foto1_u,foto2_u,features1_aux,features2_aux,status_aux);
            for(int i=0;i<status_aux.size();i++)
            {
                if(status_aux[i])
                {
                    int flag=0;
                    Point2f point_aux=features2_aux[i];
                    for(int j=0;j<ident && !flag;j++)
                    {
                        auto search_pt_project=pt_2d.find(j);
                        auto search_img=img_index.find(j);
                        if(search_pt_project!=pt_2d.end() && search_img!=img_index.end())
                        {
                            auto it_img=search_img->second.end();
                            it_img--;
                            auto it_pt=search_pt_project->second.end();
                            it_pt--;
                            Point2f last_pt=*it_pt;
                            int last_img=*it_img;
                            int norm=sqrt(powf((last_pt.x-point_aux.x),(float)2)+powf((last_pt.y-point_aux.y),(float)2));
                            if(norm<0.01 && last_img==l)
                            {
                                flag=1; //we found a point that already exists
                            }
                        }   
                    }
                    if(!flag)
                    {
                        //we found a new point
                        add_new_point_to_map(pt_2d,img_index,match_index,features1_aux[i],features2_aux[i],l,ident,status_size);
                        status.push_back(1);
                        features2.push_back(features2_aux[i]);
                        status_size++;
                    }
                }
            }
            features1.clear();
            features1=features2;
            last_status.clear();
            last_status=status;
            foto1_u=foto2_u;
            l++;
        }
        else
        {
            features1=features2;
            foto1_u=foto2_u;
            l++;
        }  
	}
    //prepare varibles for cvsba's run function
    //filter to reject points that are not visible in more than 3 images
    int valid_points[ident];
    //debug cnt
    int cnt=0;
    for(int i=0;i<ident;i++)
    {
        auto search_point=pt_2d.find(i);
        if(search_point!=pt_2d.end())
        {
            int dimension= search_point->second.size();
            if(dimension>=3)
            {
                valid_points[i]=1;
                cnt++;
            }
            else
            {
                valid_points[i]=0;
            }
        }
    }
    //matrices for cvsba's run function
    //The method Sba::run() executes the bundle adjustment optimization for a scenerio with M cameras and N 3d points
    vector<vector<Point2d>> imagePoints;
	vector<Point3d> points;
	vector<vector<int>> visibility;
	vector<Mat> cameraMatrix;
	vector<Mat> R;
	vector<Mat> T;
	vector<Mat> distortion;
    /*imagePoints:(input/[output]) vector of vectors of estimated image projections of 3d points (size MxN).
    Element imagePoints[i][j] refers to j 3d point projection over camera i.*/
    for(int i=0;i<nImages;i++)
    {
        vector<Point2d> points_row;
	    vector<int> vis_row;
        for(int j=0;j<ident;j++)
        {
            if(valid_points[j]==1)
            {
                vector<Point2f> aux_pt;
                vector<int> aux_im;
                aux_pt=pt_2d[j];
                aux_im=img_index[j];
                int stop_flag=0;
                //we search point j on image i
                for(int p=0;p<aux_im.size() && !stop_flag;p++)
                {
                    if(aux_im[p]==i)
                    {  
                        vis_row.push_back(1);
                        points_row.push_back(Point2f(aux_pt[p].x,aux_pt[p].y));
                        stop_flag=1;
                    }
                }
                if(!stop_flag)
                {
                    vis_row.push_back(0);
                    points_row.push_back(Point2f(NAN,NAN));
                }
            }
        }
        imagePoints.push_back(points_row);
        visibility.push_back(vis_row);
    }
    for (int views = 0; views < l; views++) 
    {
		cameraMatrix.push_back(intrinseca);
		distortion.push_back(distor);
		Mat rotmatrix = Mat::eye(3, 3, CV_64F);
		Mat rotvector;
		Rodrigues(rotmatrix, rotvector);
		R.push_back(rotvector);
		T.push_back(Mat::zeros(3, 1, CV_64F));
	}
	for (int npts = 0; npts < ident; npts++)
    {
		if (valid_points[npts] == 1)
        {
			points.push_back(cv::Point3d(0, 0, 0.5));
		}
	}
    /*all ready to use cvsba's run function*/
	cvsba::Sba sba;
	cvsba::Sba::Params param;
	param.type = cvsba::Sba::MOTIONSTRUCTURE;
	param.fixedIntrinsics = 5;
	param.fixedDistortion = 5;
	param.verbose = true;
	param.iterations = 50;
	sba.setParams(param);
	double error = sba.run(points, imagePoints, visibility, cameraMatrix, R, T, distortion);
	/* Graphical representation of camera's position and 3d points*/
	pcl::visualization::PCLVisualizer viewer("Viewer");
	viewer.setBackgroundColor(0.35, 0.35, 0.35);
	vector<Mat> r_end;
	vector<Mat> t_end;
	Eigen::Matrix4f initT;
	for (int i = 0; i < l; i++)
    {
		stringstream sss;
		string name;
		sss << i;
		name = sss.str();
		Mat r_aux(3, 3, CV_64F);
		Mat t_aux(3, 1, CV_64F);
		Eigen::Affine3f cam_pos;
		Eigen::Matrix4f eig_cam_pos;
		Rodrigues(R[i], r_aux);
		t_aux = T[i];
		r_aux = r_aux.t();
		t_aux = -r_aux * T[i];
        r_end.push_back(r_aux);
		t_end.push_back(t_aux);
		eig_cam_pos(0, 0) = r_aux.at<double>(0, 0);
		eig_cam_pos(0, 1) = r_aux.at<double>(0, 1);
		eig_cam_pos(0, 2) = r_aux.at<double>(0, 2);
		eig_cam_pos(0, 3) = t_aux.at<double>(0);
		eig_cam_pos(1, 0) = r_aux.at<double>(1, 0);
		eig_cam_pos(1, 1) = r_aux.at<double>(1, 1);
		eig_cam_pos(1, 2) = r_aux.at<double>(1, 2);
		eig_cam_pos(1, 3) = t_aux.at<double>(1);
		eig_cam_pos(2, 0) = r_aux.at<double>(2, 0);
		eig_cam_pos(2, 1) = r_aux.at<double>(2, 1);
		eig_cam_pos(2, 2) = r_aux.at<double>(2, 2);
		eig_cam_pos(2, 3) = t_aux.at<double>(2);
		eig_cam_pos(3, 0) = 0;
		eig_cam_pos(3, 1) = 0;
		eig_cam_pos(3, 2) = 0;
		eig_cam_pos(3, 3) = 1;

		if(i==0)
        {
            initT = eig_cam_pos;
        }
        cam_pos = initT.inverse()*eig_cam_pos;
        viewer.addCoordinateSystem(0.1, cam_pos, name);
		pcl::PointXYZ textPoint(cam_pos(0,3), cam_pos(1,3), cam_pos(2,3));
		viewer.addText3D(std::to_string(i), textPoint, 0.02, 1, 1, 1, "text_"+std::to_string(i));
	}
 FILE* fcam = fopen("/home/angel/lectura_datos/odometry.txt", "wt");
    if (fcam == NULL) return -1;
    for (int i = 0; i < t_end.size(); i++)
    {
        fprintf(fcam, "%f %f %f\n", t_end[i].at<double>(0), t_end[i].at<double>(1), t_end[i].at<double>(2));
    }
    fclose(fcam);

	pcl::PointCloud<pcl::PointXYZ> cloud;
	for(auto &pt: points)
    {
		pcl::PointXYZ p(pt.x, pt.y, pt.z);
		cloud.push_back(p);
	}
	viewer.addPointCloud<pcl::PointXYZ>(cloud.makeShared(), "map");

	while (!viewer.wasStopped()) {
		viewer.spin();
	}
	return 0;
}