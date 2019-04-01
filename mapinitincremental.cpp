#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <cvsba/cvsba.h>
#include <string>
#include <sstream>
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

using namespace cv;
using namespace std;

struct pt_2d
{
	Point2f pt;
	int img_idx;
	int match_idx;
	int p3d_ident;
};

cv::Mat loadImage(std::string _folder, int _number, cv::Mat &_intrinsics, cv::Mat &_coeffs)
{
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
					vector<int> &_ifKeypoints, vector<int> &_jfKeypoints){

	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce");
	vector<vector<DMatch>> matches;
	vector<Point2d> source, destination;
	vector<uchar> mask;
	vector<int> i_keypoint, j_keypoint;
	matcher->knnMatch(_desc1, _desc2, matches, 2);
	for (int k = 0; k < matches.size(); k++)
	{
		if (matches[k][0].distance < 0.8 * matches[k][1].distance)
		{
			source.push_back(_features1[matches[k][0].queryIdx].pt);
			destination.push_back(_features2[matches[k][0].trainIdx].pt);
			i_keypoint.push_back(matches[k][0].queryIdx);
			j_keypoint.push_back(matches[k][0].trainIdx);
		}
	}

	//aplicamos filtro ransac
	findFundamentalMat(source, destination, FM_RANSAC, 1.0, 0.99, mask);
	for (int m = 0; m < mask.size(); m++)
	{
		if (mask[m])
		{
			_ifKeypoints.push_back(i_keypoint[m]);
			_jfKeypoints.push_back(j_keypoint[m]);
		}
	}

}

int main(int argc, char **argv)
{
	// Init calibration matrices
	Mat distcoef = (Mat_<float>(1, 5) << 0.2624, -0.9531, -0.0054, 0.0026, 1.1633);
	Mat distor = (Mat_<float>(5, 1) << 0.2624, -0.9531, -0.0054, 0.0026, 1.1633);
	distor.convertTo(distor, CV_64F);
	Mat intrinseca = (Mat_<float>(3, 3) << 517.3, 0., 318.6, 0., 516.5, 255.3, 0., 0., 1.);
	intrinseca.convertTo(intrinseca, CV_64F);
	distcoef.convertTo(distcoef, CV_64F);

	vector<pt_2d> mapa;

	// Load first image.
	int l = 1;
	int ident = 0;
	Mat foto1_u = loadImage(argv[1], l, intrinseca, distor);

	// Load second image
	l++;
	Mat foto2_u = loadImage(argv[1], l, intrinseca, distor);
	;

	// Create first pair of features
	vector<KeyPoint> features1, features2;
	Mat descriptors1, descriptors2;
	auto pt = cv::ORB::create();
	pt->detectAndCompute(foto1_u, cv::Mat(), features1, descriptors1);
	pt->detectAndCompute(foto2_u, cv::Mat(), features2, descriptors2);

	// Match features
	vector<int> if_keypoint, jf_keypoint;
	matchFeatures(features1, descriptors1, features2, descriptors2, if_keypoint, jf_keypoint);

	//almacenamos los primeros puntos en el mapa bidimensional correspondientes a las dos primeras imágenes
	for (int i = 0; i < if_keypoint.size(); i++)
	{
		struct pt_2d mi_pt;
		mi_pt.img_idx = l - 1;
		mi_pt.match_idx = if_keypoint[i];
		mi_pt.p3d_ident = ident;
		mi_pt.pt = features1[if_keypoint[i]].pt;
		mapa.push_back(mi_pt);
		mi_pt.img_idx = l;
		mi_pt.match_idx = jf_keypoint[i];
		mi_pt.p3d_ident = ident;
		mi_pt.pt = features2[jf_keypoint[i]].pt;
		mapa.push_back(mi_pt);
		ident += 1;
	}
	if_keypoint.clear();
	jf_keypoint.clear();
	l++;

	while (l < 50)
	{
		Mat foto3_u = loadImage(argv[1], l, intrinseca, distor);
		
		vector<KeyPoint> features3;
		Mat descriptors3;
		pt->detectAndCompute(foto3_u, cv::Mat(), features3, descriptors3);
		matchFeatures(features2, descriptors2, features3, descriptors3, if_keypoint, jf_keypoint);

		vector<pt_2d> new_pts;
		//recorremos los puntos ya existentes vemos si son nuevos y si no lo son los añadimos al conjunto
		for (int n = 0; n < if_keypoint.size(); n++) {
			struct pt_2d punto;
			int flag = 0;
			for (int p = 0; p < mapa.size() && !flag; p++) {
				if (mapa[p].match_idx == if_keypoint[n] && mapa[p].img_idx == l - 1) {
					/*el punto ya existe y además hemos encontrado una nueva imagen en la que se ve por lo que añadimos el nuevo punto 2d al conjunto*/
					punto.img_idx = l;
					punto.match_idx = jf_keypoint[n];
					punto.p3d_ident = mapa[p].p3d_ident;
					punto.pt = features3[jf_keypoint[n]].pt;
					new_pts.push_back(punto);
					flag = 1;
				}
			}
			if (!flag) { //si no se encuentran coincidencias en el mapa el punto es nuevo por lo que se añaden sus proyecciones para dos imagenes con nuevo identificador
				punto.img_idx = l - 1;
				punto.match_idx = if_keypoint[n];
				punto.p3d_ident = ident;
				punto.pt = features2[if_keypoint[n]].pt;
				new_pts.push_back(punto);
				punto.img_idx = l;
				punto.match_idx = jf_keypoint[n];
				punto.p3d_ident = ident;
				punto.pt = features3[jf_keypoint[n]].pt;
				ident += 1;
			}
		}
		//añadimos los nuevos puntos 2d encontrados
		for (int t = 0; t < new_pts.size(); t++) {
			//struct pt_2d puntillo;
			/*puntillo.img_idx=new_pts[t].img_idx;
	    puntillo.match_idx=new_pts[t].match_idx;
	    puntillo.p3d_ident=new_pts[t].p3d_ident;
	    puntillo.pt=new_pts[t].pt;
	    mapa.push_back(puntillo);*/
			mapa.push_back(new_pts[t]);
		}
		//actualizamos la imagen de referencia
		features2.clear();
		features2 = features3;
		foto2_u = foto3_u;
		descriptors2 = descriptors3;
		new_pts.clear();
		if_keypoint.clear();
		jf_keypoint.clear();
		
		l++;
	}
	//aqui ya tenemos todos los puntos 2d con identificador del punto 3d al que corresponden
	//para el total de puntos 3d calculamos en cuantas imágenes se ven si son 3 o más será un punto válido
	int validos[ident];
	for (int i = 0; i < ident; i++) {
		for (int j = 0; j < mapa.size(); j++) {
			if (mapa[j].p3d_ident == i) {
				validos[i] += 1;
			}
		}
	}
	int usados[ident];
	for (int i = 0; i < ident; i++) {
		if (validos[i] >= 3) {
			usados[i] = 1;
		}
		else {
			usados[i] = 0;
		}
	}
	/*llegados a este punto ya sabemos que puntos 3d son válidos para el mapa inicial, procedemos a calcular las matrices
	para cvsba*/
	vector<vector<Point2d>> imagePoints;
	vector<Point3d> points;
	vector<vector<int>> visibility;
	vector<Mat> cameraMatrix;
	vector<Mat> R;
	vector<Mat> T;
	vector<Mat> distortion;
	vector<Point2d> fila_pts;
	vector<int> fila_vis;
	int flag2;
	for (int idf = 0; idf < l; idf++) {
		for (int k = 0; k < ident; k++) {
			flag2 = 0;
			for (int j = 0; j < mapa.size() && !flag2; j++) {
				/*buscamos el punto 3d k en la imagen idf si lo encuentro 1 a visibilidad y sino 0*/
				if (mapa[j].img_idx == idf && mapa[j].p3d_ident == k && usados[k] == 1) {
					fila_pts.push_back(mapa[j].pt);
					fila_vis.push_back(1);
					flag2 = 1;
				}
			}
			/*si despues de buscar en todo el mapa no encuentro el punto es que no existe en esa imagen y pongo visibilidad 0*/
			if (!flag2 && usados[k] == 1) {
				Point2d puntillo;
				puntillo.x = NAN;
				puntillo.y = NAN;
				fila_pts.push_back(puntillo);
				fila_vis.push_back(0);
			}
		}
		imagePoints.push_back(fila_pts); //*añado la fila correspondiente a la camara idf
		visibility.push_back(fila_vis);  //añados la visibilidad correspondiente a la cámara idf
		fila_pts.clear();
		fila_vis.clear();
	}
	/*
	for(int views=0;views<l;views++)
	{
	cameraMatrix.push_back(intrinseca);
	distortion.push_back(distor);
	R.push_back(Mat::eye(3, 3, CV_64F));
	T.push_back(Mat::zeros(3, 1, CV_64F));
	}
	*/
	for (int views = 0; views < l; views++) {
		cameraMatrix.push_back(intrinseca);
		distortion.push_back(distor);
		Mat rotmatrix = Mat::eye(3, 3, CV_64F);
		Mat rotvector;
		Rodrigues(rotmatrix, rotvector);
		R.push_back(rotvector);
		T.push_back(Mat::zeros(3, 1, CV_64F));
	}
	for (int npts = 0; npts < ident; npts++) {
		if (usados[npts] == 1) {
			points.push_back(cv::Point3d(0, 0, 0.5));
		}
	}

	/*ya esta todo listo para usar cvsba*/
	cvsba::Sba sba;
	cvsba::Sba::Params param;
	param.type = cvsba::Sba::MOTIONSTRUCTURE;
	param.fixedIntrinsics = 5;
	param.fixedDistortion = 5;
	param.verbose = true;
	sba.setParams(param);
	double error = sba.run(points, imagePoints, visibility, cameraMatrix, R, T, distortion);
	/* representación gráfica de las poses de cámara*/
	pcl::visualization::PCLVisualizer viewer("Viewer");
	viewer.setBackgroundColor(255, 255, 255);
	//pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
	/*for(int i=0;i<l;i++)
	{
	stringstream sss;
	string name;
	sss<<i;
	name=sss.str();
	Eigen::Affine3f cam_pos;
	Eigen::Matrix4f eig_cam_pos;
	eig_cam_pos(0,0) = R[i].at<double>(0,0);eig_cam_pos(0,1) = R[i].at<double>(0,1);eig_cam_pos(0,2) = R[i].at<double>(0,2);eig_cam_pos(0,3) = T[i].at<double>(0,0);
	    eig_cam_pos(1,0) = R[i].at<double>(1,0);eig_cam_pos(1,1) = R[i].at<double>(1,1);eig_cam_pos(1,2) = R[i].at<double>(1,2);eig_cam_pos(1,3) = T[i].at<double>(1,0);
	    eig_cam_pos(2,0) = R[i].at<double>(2,0);eig_cam_pos(2,1) = R[i].at<double>(2,1);eig_cam_pos(2,2) = R[i].at<double>(2,2);eig_cam_pos(2,3) = T[i].at<double>(2,0);
	    eig_cam_pos(3,0) = 0 ;eig_cam_pos(3,1) = 0;eig_cam_pos(3,2) = 0;eig_cam_pos(3,3) = 1;
	cam_pos = eig_cam_pos;
	    viewer.addCoordinateSystem(1.0, cam_pos,name);
	viewer.spinOnce();

	}
	*/
	/*vector<Mat> t_fin;
	vector<Mat> r_fin;
	for(int i=0;i<l;i++)
	{
	stringstream sss;
	string name;
	sss<<i;
	Mat t_aux(3,1,CV_64F);
	Mat r_aux(3,3,CV_64F);
	t_aux=-1*(R[i].t())*T[i];
	r_aux=R[i].t();
	t_fin.push_back(t_aux);
	r_fin.push_back(r_aux);
	name=sss.str();
	Eigen::Affine3f cam_pos;
	Eigen::Matrix4f eig_cam_pos;
	eig_cam_pos(0,0) = r_aux.at<double>(0,0);eig_cam_pos(0,1) = r_aux.at<double>(0,1);eig_cam_pos(0,2) = r_aux.at<double>(0,2);eig_cam_pos(0,3) = t_aux.at<double>(0,0);
	    eig_cam_pos(1,0) = r_aux.at<double>(1,0);eig_cam_pos(1,1) = r_aux.at<double>(1,1);eig_cam_pos(1,2) = r_aux.at<double>(1,2);eig_cam_pos(1,3) = t_aux.at<double>(1,0);
	    eig_cam_pos(2,0) = r_aux.at<double>(2,0);eig_cam_pos(2,1) = r_aux.at<double>(2,1);eig_cam_pos(2,2) = r_aux.at<double>(2,2);eig_cam_pos(2,3) = t_aux.at<double>(2,0);
	    eig_cam_pos(3,0) = 0;eig_cam_pos(3,1) = 0;eig_cam_pos(3,2) = 0;eig_cam_pos(3,3) = 1;
	cam_pos = eig_cam_pos;
	    viewer.addCoordinateSystem(1.0, cam_pos,name);
	viewer.spinOnce();

	}
	*/
	vector<Mat> r_fin;
	vector<Mat> t_fin;
	for (int i = 0; i < l; i++) {
		stringstream sss;
		string name;
		sss << i;
		name = sss.str();
		Mat r_aux(3, 3, CV_64F);
		Mat t_aux(3, 1, CV_64F);
		Eigen::Affine3f cam_pos;
		Eigen::Matrix4f eig_cam_pos;
		Rodrigues(R[i], r_aux);
		r_aux = r_aux.t();
		r_fin.push_back(r_aux);
		t_aux = -r_aux * T[i];
		t_fin.push_back(t_aux);
		eig_cam_pos(0, 0) = r_aux.at<double>(0, 0);
		eig_cam_pos(0, 1) = r_aux.at<double>(0, 1);
		eig_cam_pos(0, 2) = r_aux.at<double>(0, 2);
		eig_cam_pos(0, 3) = t_aux.at<double>(0, 0);
		eig_cam_pos(1, 0) = r_aux.at<double>(1, 0);
		eig_cam_pos(1, 1) = r_aux.at<double>(1, 1);
		eig_cam_pos(1, 2) = r_aux.at<double>(1, 2);
		eig_cam_pos(1, 3) = t_aux.at<double>(1, 0);
		eig_cam_pos(2, 0) = r_aux.at<double>(2, 0);
		eig_cam_pos(2, 1) = r_aux.at<double>(2, 1);
		eig_cam_pos(2, 2) = r_aux.at<double>(2, 2);
		eig_cam_pos(2, 3) = t_aux.at<double>(2, 0);
		eig_cam_pos(3, 0) = 0;
		eig_cam_pos(3, 1) = 0;
		eig_cam_pos(3, 2) = 0;
		eig_cam_pos(3, 3) = 1;
		cam_pos = eig_cam_pos;
		viewer.addCoordinateSystem(1.0, cam_pos, name);
		viewer.spinOnce();
	}
	while (!viewer.wasStopped()) {
		viewer.spin();
	}
	return 0;
}
