#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <cvsba/cvsba.h>
#include <string>
#include <sstream>
#include<stdlib.h>
#include <unistd.h>
#include <ctime>
#include <pcl/common/common_headers.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/Geometry>
#include <opencv2/xfeatures2d.hpp>
#include <sys/stat.h>
#include <fstream>

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

bool matchFeatures(	vector<KeyPoint> &_features1, cv::Mat &_desc1, 
					vector<KeyPoint> &_features2, cv::Mat &_desc2,
					vector<int> &_ifKeypoints, vector<int> &_jfKeypoints,double dst_ratio,double confidence,double reproject_err){

	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce");
	vector<vector<DMatch>> matches;
	vector<Point2d> source, destination;
	vector<uchar> mask;
	vector<int> i_keypoint, j_keypoint;
	matcher->knnMatch(_desc1, _desc2, matches, 2);
	for (int k = 0; k < matches.size(); k++)
	{
		if (matches[k][0].distance < dst_ratio * matches[k][1].distance)
		{
			source.push_back(_features1[matches[k][0].queryIdx].pt);
			destination.push_back(_features2[matches[k][0].trainIdx].pt);
			i_keypoint.push_back(matches[k][0].queryIdx);
			j_keypoint.push_back(matches[k][0].trainIdx);
		}
	}

	//aplicamos filtro ransac
	findFundamentalMat(source, destination, FM_RANSAC,reproject_err, confidence, mask);
	for (int m = 0; m < mask.size(); m++)
	{
		if (mask[m])
		{
			_ifKeypoints.push_back(i_keypoint[m]);
			_jfKeypoints.push_back(j_keypoint[m]);
		}
	}
}

void displayMatches(	cv::Mat &_img1, std::vector<cv::KeyPoint> &_features1, std::vector<int> &_filtered1,
						cv::Mat &_img2, std::vector<cv::KeyPoint> &_features2, std::vector<int> &_filtered2){
	cv::Mat display;
	cv::hconcat(_img1, _img2, display);
	cv::cvtColor(display, display, CV_GRAY2BGR);

	for(unsigned i = 0; i < _filtered1.size(); i++){
		auto p1 = _features1[_filtered1[i]].pt;
		auto p2 = _features2[_filtered2[i]].pt + cv::Point2f(_img1.cols, 0);
		cv::circle(display, p1, 2, cv::Scalar(0,255,0),2);
		cv::circle(display, p2, 2, cv::Scalar(0,255,0),2);
		cv::line(display,p1, p2, cv::Scalar(0,255,0),1);
	}

	cv::imshow("display", display);
	cv::waitKey(3);
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
    //threshold for points tha are seen in more than i images
    int img_threshold;
    sscanf(argv[3],"%d",&img_threshold);
    //distance ratio to reject features
    double dst_ratio;
    sscanf(argv[4],"%lf",&dst_ratio);
    //confidence on ransac
    double confidence;
    sscanf(argv[5],"%lf",&confidence);
    //threshold for error_reprojection on ransac
    double reproject_err;
    sscanf(argv[6],"%lf",&reproject_err);
    //cvsba iterations
    int niter;
    sscanf(argv[7],"%d",&niter);
    //map for 3d points projections
    unordered_map<int,vector<Point2f>> pt_2d;
    //map for image index of 3d points projections;
    unordered_map<int,vector<int>> img_index;
    //map for match index
    unordered_map<int,vector<int>> match_index;
    //iterator to iterate through images
    int l=0;
    //identifier for 3d_point
    int ident=0;
    //we need variables to store the last image and the last features & descriptors
    auto pt =SURF::create(500);
    Mat foto1_u;
    vector<KeyPoint> features1;
    Mat descriptors1;
    //load first image
    foto1_u = loadImage(argv[1], l, intrinseca, distcoef);
    pt->detectAndCompute(foto1_u, Mat(), features1, descriptors1);
    l++;
    while(l<nImages)
    {   
        //load new image
        Mat foto2_u = loadImage(argv[1], l, intrinseca, distcoef);
        //create pair of features
        vector<KeyPoint> features2;
	    Mat descriptors2;
	    pt->detectAndCompute(foto2_u, Mat(), features2, descriptors2);
        //match features
        vector<int> if_keypoint, jf_keypoint;
        matchFeatures(features1, descriptors1, features2, descriptors2, if_keypoint, jf_keypoint,dst_ratio,confidence,reproject_err);
        displayMatches(foto1_u, features1, if_keypoint,foto2_u, features2, jf_keypoint);
        Mat used_features=Mat::zeros(1,int(if_keypoint.size()),CV_64F);//to differentiate the features that correspond to new points from those that do not
        if(ident>0)
        {
            for(int j=0;j<ident;j++)
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
                    int flag=0;
                    for(int k=0;k<if_keypoint.size() && !flag;k++)
                    {
                        if(if_keypoint[k]==last_match && last_img==l-1)
                        {
                            //we add the new projection for the same 3d point
                            pt_2d[j].push_back(features2[jf_keypoint[k]].pt);
                            img_index[j].push_back(l);
                            match_index[j].push_back(jf_keypoint[k]);
                            used_features.at<double>(k)=1;
                            flag=1;
                        }
                    }
                }
            }
        }
        //we add the projections of the new 3d points for two consecutive frames
       
        for (int i=0;i<if_keypoint.size();i++)
        {
            if(used_features.at<double>(i)==0)
            {
                pt_2d[ident]=vector<Point2f>();
                pt_2d[ident].push_back(features1[if_keypoint[i]].pt);
                img_index[ident]=vector<int>();
                img_index[ident].push_back(l-1);
                match_index[ident]=vector<int>();
                match_index[ident].push_back(if_keypoint[i]);
                pt_2d[ident].push_back(features2[jf_keypoint[i]].pt);
                img_index[ident].push_back(l);
                match_index[ident].push_back(jf_keypoint[i]);
                ident++;
            }
        }
        foto1_u=foto2_u;
        features1=features2;
        descriptors1=descriptors2;
        used_features.release();
        l++;
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
            if(dimension>=img_threshold)
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
    
    //we know the first position of the camera
    //orientation
    Eigen::Quaterniond q;
    q.x()=0.0;
    q.y()=-0.0;
    q.z()=0.0;
    q.w()=-0.1;
    Eigen::Matrix3d f_rot=q.normalized().toRotationMatrix();
    //xyz position in world coordinates
    Mat first_pos=Mat::zeros(3,1,CV_64F);
    first_pos.at<double>(0)=0.0;
    first_pos.at<double>(1)=0.0;
    first_pos.at<double>(2)=0.0;
    //camera extrinsics for cvsba
    Mat first_rot=Mat::zeros(3,3,CV_64F);
    first_rot.at<double>(0,0)=f_rot(0,0);
    first_rot.at<double>(0,1)=f_rot(0,1);
    first_rot.at<double>(0,2)=f_rot(0,2);
    first_rot.at<double>(1,0)=f_rot(1,0);
    first_rot.at<double>(1,1)=f_rot(1,1);
    first_rot.at<double>(1,2)=f_rot(1,2);
    first_rot.at<double>(2,0)=f_rot(2,0);
    first_rot.at<double>(2,1)=f_rot(2,1);
    first_rot.at<double>(2,2)=f_rot(2,2);
    Mat first_tras;
    first_tras=-(first_rot.t())*first_pos;
    first_rot=first_rot.t();
    for (int views = 0; views < l; views++) 
    {
		cameraMatrix.push_back(intrinseca);
		distortion.push_back(distor);
		//Mat rotmatrix = Mat::eye(3, 3, CV_64F);
		Mat rotvector;
		//Rodrigues(rotmatrix, rotvector);
        Rodrigues(first_rot,rotvector);
		R.push_back(rotvector);
		T.push_back(first_tras);
	}
	for (int npts = 0; npts < ident; npts++)
    {
		if (valid_points[npts] == 1)
        {
            //double rnd_x=((double)rand() / RAND_MAX);
            //double rnd_y=((double)rand() / RAND_MAX);

			points.push_back(cv::Point3d(0,0,0.5));
		}
	}
    /*all ready to use cvsba's run function*/
	cvsba::Sba sba;
	cvsba::Sba::Params param;
	param.type = cvsba::Sba::MOTIONSTRUCTURE;
	param.fixedIntrinsics = 5;
	param.fixedDistortion = 5;
	param.verbose = true;
	param.iterations = niter;
	sba.setParams(param);
	double error = sba.run(points, imagePoints, visibility, cameraMatrix, R, T, distortion);
	/* Graphical representation of camera's position and 3d points*/
	pcl::visualization::PCLVisualizer viewer("Viewer");
	viewer.setBackgroundColor(0.35, 0.35, 0.35);
	vector<Mat> r_end;
	vector<Mat> t_end;
	Eigen::Matrix4f initT;

    mkdir("./lectura_datos", 0777);
    std::ofstream file("./lectura_datos/odometry.txt");
    if (!file.is_open()) return -1;

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
		// t_aux = -r_aux.t() * T[i];
        // r_end.push_back(r_aux.t());
		// t_end.push_back(t_aux);
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

        // eig_cam_pos = eig_cam_pos.inverse().eval();

		if(i==0) {
            initT = eig_cam_pos;
        }
        cam_pos = initT.inverse()*eig_cam_pos;

        viewer.addCoordinateSystem(0.05, cam_pos, name);
		pcl::PointXYZ textPoint(cam_pos(0,3), cam_pos(1,3), cam_pos(2,3));
		viewer.addText3D(std::to_string(i), textPoint, 0.01, 1, 1, 1, "text_"+std::to_string(i));

        Eigen::Quaternionf q(cam_pos.matrix().block<3,3>(0,0));

        file << 0.0 << " " << cam_pos(0,3) << " " <<  cam_pos(1,3) << " " << cam_pos(2,3) << " " << 
                q.x() << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;
    
    }
    
    file.close();

	pcl::PointCloud<pcl::PointXYZ> cloud;
	for(auto &pt: points)
    {
        /*Eigen::Vector4f aux_point;
        Eigen::Vector4f end_point;
        aux_point(0,0)=pt.x;
        aux_point(1,0)=pt.y;
        aux_point(2,0)=pt.z;
        aux_point(3,0)=1.0 ;
        end_point=initT.inverse()*aux_point;
		pcl::PointXYZ p(end_point(0,0)/end_point(3,0),end_point(1,0)/end_point(3,0),end_point(2,0)/end_point(3,0));
        */

        Eigen::Vector4f p_aux = {pt.x,pt.y,pt.z,1};
        p_aux = initT.inverse()*p_aux;
        pcl::PointXYZ p(p_aux[0], p_aux[1], p_aux[2]);
		cloud.push_back(p);
	}
	viewer.addPointCloud<pcl::PointXYZ>(cloud.makeShared(), "map");

	while (!viewer.wasStopped()) {
		viewer.spin();
	}
	return 0;
}
