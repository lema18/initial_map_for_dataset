#include <stdio.h>
#include <iostream>
#include "opencv2/core.hpp"
#include <opencv2/opencv.hpp>
#include "opencv2/features2d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/xfeatures2d.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
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
#include "vo_features.h"
#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/Geometry>

using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;

struct pt_2d
{
    Point2f pt;
    int img_idx;
    int match_idx;
    int p3d_ident;
};

int main(int argc,char **argv)
{
    Mat distcoef =(Mat_<float>(1,5)<<0.2624 ,-0.9531, -0.0054 ,0.0026, 1.1633);
    Mat distor=(Mat_<float>(5,1)<<0.2624 ,-0.9531, -0.0054 ,0.0026, 1.1633);
    distor.convertTo(distor,CV_64F);
    Mat intrinseca=(Mat_<float>(3,3)<<517.3, 0., 318.6, 0., 516.5, 255.3, 0., 0., 1.);
    intrinseca.convertTo(intrinseca,CV_64F);
    distcoef.convertTo(distcoef,CV_64F);
    vector<pt_2d> mapa;
    int l=0;
    int ident=0;
    stringstream ss; /*convertimos el entero l a string para poder establecerlo como nombre de la captura*/
	ss<<l;
	string nombre="left_";
	nombre+=ss.str();
	nombre+=".png";
	Mat foto1=imread(nombre,CV_LOAD_IMAGE_COLOR);
    cvtColor(foto1,foto1,COLOR_BGR2GRAY);
    Mat foto1_u;
    undistort(foto1,foto1_u,intrinseca,distcoef);
    l+=1;
    stringstream ss1;
    ss1<<l;
    string nombre1="left_";
    nombre1+=ss1.str();
    nombre1+=".png";
    Mat foto2_u;
    Mat foto2=imread(nombre1,CV_LOAD_IMAGE_COLOR);
    cvtColor(foto2,foto2,COLOR_BGR2GRAY);
    undistort(foto2,foto2_u,intrinseca,distcoef);
    vector<KeyPoint> features1,features2;
    Mat descriptors1,descriptors2;
    Ptr<SURF> pt=SURF::create(500);
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce");
    pt->detect(foto1_u,features1);
    pt->detect(foto2_u,features2);
    pt->compute(foto1_u,features1,descriptors1);
    pt->compute(foto2_u,features2,descriptors2);
    vector<vector<DMatch>> matches;
    vector<Point2d> source,destination;
    vector<uchar> mask;
    vector<int> i_keypoint, j_keypoint,if_keypoint,jf_keypoint;
    matcher->knnMatch(descriptors1,descriptors2,matches,2);
    for(int k=0;k<matches.size();k++)
    {
        if(matches[k][0].distance < 0.8*matches[k][1].distance)
        {
            source.push_back(features1[matches[k][0].queryIdx].pt);
            destination.push_back(features2[matches[k][0].trainIdx].pt);
            i_keypoint.push_back(matches[k][0].queryIdx);
            j_keypoint.push_back(matches[k][0].trainIdx);
        }
    }
    //aplicamos filtro ransac
    findFundamentalMat(source,destination,FM_RANSAC,1.0,0.99,mask);
    for(int m=0;m<mask.size();m++)
    {
        if(mask[m])
        {
        if_keypoint.push_back(i_keypoint[m]);
        jf_keypoint.push_back(j_keypoint[m]);
        }
    }
    //almacenamos los primeros puntos en el mapa bidimensional correspondientes a las dos primeras imágenes
    for(int i=0;i<if_keypoint.size();i++)
    {
        struct pt_2d mi_pt;
        mi_pt.img_idx=l-1;
        mi_pt.match_idx=if_keypoint[i];
        mi_pt.p3d_ident=ident;
        mi_pt.pt=features1[if_keypoint[i]].pt;
        mapa.push_back(mi_pt);
        mi_pt.img_idx=l;
        mi_pt.match_idx=jf_keypoint[i];
        mi_pt.p3d_ident=ident;
        mi_pt.pt=features2[jf_keypoint[i]].pt;
        mapa.push_back(mi_pt);
        ident+=1;
    }
    matches.clear();
    i_keypoint.clear();
    if_keypoint.clear();
    source.clear();
    destination.clear();
    j_keypoint.clear();
    jf_keypoint.clear();
    mask.clear();
    l+=1;
    while(l<50)
    {
        stringstream ss2;
        ss2<<l;
        string nombre2="left_";
	    nombre2+=ss2.str();
	    nombre2+=".png";
        Mat foto3_u;
	    Mat foto3=imread(nombre2,CV_LOAD_IMAGE_COLOR);
        cvtColor(foto3,foto3,COLOR_BGR2GRAY);
        undistort(foto3,foto3_u,intrinseca,distcoef);
        vector<KeyPoint> features3;
        Mat descriptors3;
        pt->detect(foto3_u,features3);
        pt->compute(foto3_u,features3,descriptors3);
        matcher->knnMatch(descriptors2,descriptors3,matches,2);
        for(int k=0;k<matches.size();k++)
        {
            if(matches[k][0].distance < 0.8*matches[k][1].distance)
            {   
            source.push_back(features2[matches[k][0].queryIdx].pt);
            destination.push_back(features3[matches[k][0].trainIdx].pt);
            i_keypoint.push_back(matches[k][0].queryIdx);
            j_keypoint.push_back(matches[k][0].trainIdx);
            }
        }
        //aplicamos filtro ransac
        findFundamentalMat(source,destination,FM_RANSAC,1.0,0.99,mask);
        for(int m=0;m<mask.size();m++)
        {
            if(mask[m])
            {
                if_keypoint.push_back(i_keypoint[m]);
                jf_keypoint.push_back(j_keypoint[m]);
            }
        }
        vector<pt_2d> new_pts;
        //recorremos los puntos ya existentes vemos si son nuevos y si no lo son los añadimos al conjunto
        for(int n=0;n<if_keypoint.size();n++)
        {
            struct pt_2d punto;
            int flag=0;
            for(int p=0;p<mapa.size() && !flag;p++)
            {
               
                if(mapa[p].match_idx==if_keypoint[n] && mapa[p].img_idx==l-1)
                {
                    /*el punto ya existe y además hemos encontrado una nueva imagen en la que se ve por lo que añadimos el nuevo
                    punto 2d al conjunto*/
                    punto.img_idx=l;
                    punto.match_idx=jf_keypoint[n];
                    punto.p3d_ident=mapa[p].p3d_ident;
                    punto.pt=features3[jf_keypoint[n]].pt;
                    new_pts.push_back(punto);
                    flag=1;

                }
            }
            if(!flag)//si no se encuentran coincidencias en el mapa el punto es nuevo por lo que se añaden sus proyecciones para dos imagenes con nuevo identificador
            {
                punto.img_idx=l-1;
                punto.match_idx=if_keypoint[n];
                punto.p3d_ident=ident;
                punto.pt=features2[if_keypoint[n]].pt;
                new_pts.push_back(punto);
                punto.img_idx=l;
                punto.match_idx=jf_keypoint[n];
                punto.p3d_ident=ident;
                punto.pt=features3[jf_keypoint[n]].pt;
                ident+=1;
            }
        }
        //añadimos los nuevos puntos 2d encontrados
        for(int t=0;t<new_pts.size();t++)
        {
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
        features2=features3;
        foto2_u=foto3_u;
        descriptors2=descriptors3;
        new_pts.clear();
        matches.clear();
        i_keypoint.clear();
        if_keypoint.clear();
        j_keypoint.clear();
        jf_keypoint.clear();
        source.clear();
        destination.clear();
        mask.clear();
        l+=1;
    }
    //aqui ya tenemos todos los puntos 2d con identificador del punto 3d al que corresponden
    //para el total de puntos 3d calculamos en cuantas imágenes se ven si son 3 o más será un punto válido
    int validos[ident];
    for(int i=0;i<ident;i++)
    {
        for(int j=0;j<mapa.size();j++)
        {
            if(mapa[j].p3d_ident==i)
            {
                validos[i]+=1;
            }
        }
    }
    int usados[ident];
    for(int i=0;i<ident;i++)
    {
        if(validos[i]>=3)
        {
            usados[i]=1;
        }
        else
        {
            usados[i]=0;
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
    for(int idf=0;idf<l;idf++)
    {
        for (int k=0;k<ident;k++)
        {
            flag2=0;
            for(int j=0;j<mapa.size() && !flag2;j++)
            {   
                /*buscamos el punto 3d k en la imagen idf si lo encuentro 1 a visibilidad y sino 0*/
                if(mapa[j].img_idx==idf && mapa[j].p3d_ident==k && usados[k]==1)
                {
                    fila_pts.push_back(mapa[j].pt);
                    fila_vis.push_back(1);
                    flag2=1;
                }
            }
            /*si despues de buscar en todo el mapa no encuentro el punto es que no existe en esa imagen y pongo visibilidad 0*/
            if(!flag2 && usados[k]==1)
            {
                Point2d puntillo;
                puntillo.x=NAN;
                puntillo.y=NAN;
                fila_pts.push_back(puntillo);
                fila_vis.push_back(0);
            }
        }
        imagePoints.push_back(fila_pts);//*añado la fila correspondiente a la camara idf
        visibility.push_back(fila_vis);//añados la visibilidad correspondiente a la cámara idf
        fila_pts.clear();
        fila_vis.clear();
    }
    for(int views=0;views<l;views++)
    {
        cameraMatrix.push_back(intrinseca);
        distortion.push_back(distor);
        R.push_back(Mat::eye(3, 3, CV_64F));
        T.push_back(Mat::zeros(3, 1, CV_64F));
    }
    for(int npts=0;npts<ident;npts++)
    {
        if(usados[npts]==1)
        {
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
    double error = sba.run(points,imagePoints,visibility,cameraMatrix,R,T,distortion);
    /* representación gráfica de las poses de cámara*/
    pcl::visualization::PCLVisualizer viewer("Viewer");
  	viewer.setBackgroundColor (255, 255, 255);
    //pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
    for(int i=0;i<l;i++)
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
	while (!viewer.wasStopped ())
	{
		viewer.spin();
	}
    return 0;
}