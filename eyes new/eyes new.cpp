/**
* @file objectDetection2.cpp
* @author A. Huaman ( based in the classic facedetect.cpp in samples/c )
* @brief A simplified version of facedetect.cpp, show how to load a cascade classifier and how to find objects (Face + eyes) in a video stream - Using LBP here
*/
#include "stdafx.h" 
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>

#define NOR_H    80  //正規化後圖形的高
#define NOR_W    600   //正規化後圖形的寬

#define L_P_X    150    //正規化後圖形左眼角X 
#define L_P_Y    40    //正規化後圖形左眼角Y 

#define R_P_X    450    //正規化後圖形右眼角X 
#define R_P_Y    40    //正規化後圖形右眼角Y 

using namespace std;
using namespace cv;

/** Global variables，包含面部检测xml和眼睛检测xml */
//string face_cascade_name = "lbpcascade_frontalface.xml"; //导入正确的xml绝对路径
string eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";
//CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
//string face_name = "Capture - Face detection";
string eyes_name = "Capture - Eyes detection";

RNG rng(12345);

CvCapture* capture;//攝影機參數
Mat frame;//main RGB frame
Mat LeyeROI;//左眼ROI
Mat ReyeROI;//右眼ROI
Mat LeyeCornerTpl;//左眼眼角TPL
Mat ReyeCornerTpl;//右眼眼角TPL

Mat norImgDraw;
Mat norLeyeROI;
Mat norReyeROI;
Mat norImg;
Mat norImgTpl;

cv::Rect LeyeRect;//左眼RECT
cv::Rect ReyeRect;//右眼RECT
cv::Rect LeyeCornerRect;//左眼眼角RECT
cv::Rect ReyeCornerRect;//右眼眼角RECT  
//RECT就是整張影像中ROI的位址跟大小 但不存在影像資訊 用以控制ROI
cv::Rect LeyeBallRect;
cv::Rect ReyeBallRect; 
cv::Rect LeyePupilRect;
cv::Rect ReyePupilRect;
cv::Rect MainRect;
cv::Rect norImgRect;
Point LeyeCorner;//左眼眼角
Point ReyeCorner;//右眼眼角
Point LeyePupil;//左眼瞳孔
Point ReyePupil;//右眼瞳孔
bool Pupilcheck;

int norImgTplcounter=0;
double* imgArr;
double* oimgArr;

vector<Vec3f> circles;
Mat norImgc;

double GetPositionGrayValue(double *imagechannel,int ImageW,int ImageH,int i,int j)
{
	if(i<ImageW && j<ImageH)
	return (imagechannel[i+j*ImageW]);
	else
	return 0;
}

//Imagechanel: Input&OutputImg (1D: row-major) with size Width& Height

void HistogramEqualiz(double *Imagechanel,int Width,int Height)
{
	int i;
	long *count;
	long *EquaTable; //save the Equalization lookup table
	int *tempMatrix;
	count=new long [256];//count[i]表示gray val=i的個數
	EquaTable=new long [256];//EquaTable[i] 表示gray val=i時要對應到的val
	tempMatrix=new int [Width*Height];//暫存image2

	for(i=0;i<256;i++)//初始化
	{
		count[i]=0;
		EquaTable[i]=0;
	}

	for(i=0;i<Width*Height;i++)//計算每一gray val的個數
	{
		count[(int)Imagechanel[i]]++;
	}
	for(i=1;i<256;i++)        //累計
	{
		count[i]=count[i-1]+count[i];
	}
    //以上計算pdf(累計)
	for(i=0;i<256;i++)
	{
		int temp;
		temp = count[i];
		EquaTable[i]=int (255*temp/count[255]);
	}
	//result=pdf(累計)*255
	for(i=0;i<Width*Height;i++)
	{
		tempMatrix[i]=EquaTable[(int)(Imagechanel[i])];
	}

	for(i=0;i<Width*Height;i++)
	{
		Imagechanel[i]=(int)tempMatrix[i];
	}
	delete []tempMatrix;
	delete [] count;
	delete []EquaTable;
}

//InImage: input image (1D: row-major)
//InH: H of input image
//InW: W of input image
//lx: leye.x@InImage
//OutImage (pre-def: size NOR_H*NOR_W )

void Normal_2Points(double *InImage,int InH,int InW,double lx,double ly,double rx,double ry,double *OutImage)
{
	// TODO: Add your control notification handler code here
		
	    //inwh = 輸入影像的寬*高
		int Inwh=InW*InH;
	    double OriginalDistance;
		double Dx,Dy;
		//NormalCenterX = 經過轉換後兩眼平均 (X)
		double NormalCenterX=(L_P_X+R_P_X)/2;
		//NormalCenterY = 經過轉換後兩眼平均 (Y)
		double NormalCenterY=(L_P_Y+R_P_Y)/2;
		//OriginalCenterX = 原圖兩眼中心座標 (X)
		double OrigninalCenterX=(lx+rx)/2;
		//OriginalCenterY = 原圖兩眼中心座標 (Y)
		double OrigninalCenterY=(ly+ry)/2;
		//OriginalDistance = 原圖兩眼距離
		OriginalDistance=sqrt(pow((lx-rx),2)+pow((ly-ry),2));
		//S = 轉換後的兩眼(X)差/原圖兩點距離
		double S=fabs((double)(L_P_X-R_P_X))/OriginalDistance;
		//Dx = 原圖兩眼(X)差
		Dx=(lx-rx);
		//Dy = 原圖兩眼(Y)差
		Dy=(ly-ry);
		//theda = 兩眼連線與水平的夾角
	   	double theda=atan2(fabs(Dy),fabs(Dx));

		if(Dx<0 && Dy<0)
			theda=theda;
		else if(Dx<0 && Dy>0)
			theda=-theda;
		else if(Dx>0 && Dy<0)
			theda=3.1415926-theda;
		else if(Dx>0 && Dy>0)
			theda=theda-3.1415926;
		else if(Dx<0 && Dy==0)
			theda=0;
		else if(Dx>0 && Dy==0)
			theda=3.1415926;
		else if(Dx==0 && Dy<0)
			theda=3.1415926/2;
		else if(Dx==0 && Dy>0)
			theda=-3.1415926/2;

		for (int j=0;j<NOR_H;j++)
		{
			for (int i=0;i<NOR_W;i++)
			{
				double Original_u,Original_v;
				double a,b,c,d;
				double TempshiftX,TempshiftY;

				TempshiftX=double(i-NormalCenterX);
				TempshiftY=double(j-NormalCenterY);

				Original_u=double(TempshiftX*cos(theda)-TempshiftY*sin(theda))/S+OrigninalCenterX;
				Original_v=double(TempshiftX*sin(theda)+TempshiftY*cos(theda))/S+OrigninalCenterY;

				int p=int(Original_u);
				int q=int(Original_v);

				if (p>InW-2 || p<0 || q>InH-2 || q<0)
				{
					a=b=c=d=0;
				}
				else
				{
					a=GetPositionGrayValue(InImage,InW,InH,p,q);
					b=GetPositionGrayValue(InImage,InW,InH,p+1,q);
					c=GetPositionGrayValue(InImage,InW,InH,p,q+1);
					d=GetPositionGrayValue(InImage,InW,InH,p+1,q+1);
				}
				double color=(q+1-Original_v)*(p+1-Original_u)*a+
							 (q+1-Original_v)*(Original_u-p)*b+
							 (Original_v-q)*(p+1-Original_u)*c+
							 (Original_v-q)*(Original_u-p)*d;

			OutImage[i+j*NOR_W]=fabs(color+0.5);


			}
		}
        HistogramEqualiz(OutImage,NOR_W,NOR_H); 
}


void MattoArr(Mat img, double* imgArr){
	for(int i=0;i<img.cols*img.rows;i++)
        imgArr[i]=(double)img.data[i];
}

void ArrtoMat(Mat img, double* imgArr){
	for(int i=0;i<NOR_H*NOR_W;i++)
		img.data[i]=(int)imgArr[i];	
}
void detectStage(Mat frame_gray,
				 Mat &LeyeROI,
				 Mat &ReyeROI,
				 Mat &LeyeCornerTpl,
				 Mat &ReyeCornerTpl,
				 cv::Rect &LeyeRect,
				 cv::Rect &ReyeRect,
				 cv::Rect &LeyeCornerRect,
				 cv::Rect &ReyeCornerRect,
				 Point &LeyeCorner,
				 Point &ReyeCorner
				 )
{
	Mat faceROI;
	Mat LeyeCornerROI;
	Mat ReyeCornerROI;
	std::vector<Rect> faces;//for faces detect function
	std::vector<Rect> eyes;//for eyes detect function
	std::vector<Rect> eyesTemp;
	cv::Rect faceRect;
	
	face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0, Size(80, 80) );//偵測臉
	if(faces.size()==1&&faces[0].width<frame.cols&&faces[0].height<frame.rows)
	{
		for( size_t i = 0; i < (faces.size()==1); i++ )
		{		
			faceROI = frame_gray( faces[i] );
			eyes_cascade.detectMultiScale( faceROI, eyes, 1.1, 2, 0 |CV_HAAR_SCALE_IMAGE, Size(30, 30) );//偵測眼睛
			if(eyes.size()==2)
			{
				for( size_t j = 0; j < (eyes.size()==2); j++ )
				{
					eyesTemp=eyes;
					if(eyes[j].x>eyes[j+1].x) //設定眼睛位置eyes[j]=左 eyes[j+1]=右
					{
						eyesTemp[j]=eyes[j];
						eyes[j]=eyes[j+1];
						eyes[j+1]=eyesTemp[j];
					}
					LeyeRect = Rect(eyes[j].x + faces[i].x, eyes[j].y + faces[i].y + eyes[j].height*0.3, eyes[j].width, eyes[j].height*0.6);
					LeyeROI = frame_gray( LeyeRect );
					//設定眼角位置為眼睛ROI的1/2大小並設在右邊中間
					LeyeCornerRect = cv::Rect(eyes[j].x + eyes[j].width/2, eyes[j].y + eyes[j].height/4, eyes[j].width/2, eyes[j].height/2) + cv::Point(faces[i].x, faces[i].y);
					LeyeCornerROI = frame_gray( LeyeCornerRect );

					ReyeRect = Rect(eyes[j+1].x + faces[i].x, eyes[j+1].y + faces[i].y + eyes[j+1].height*0.3, eyes[j+1].width, eyes[j+1].height*0.6);
					ReyeROI = frame_gray( ReyeRect );
					//設定眼角位置為眼睛ROI的1/2大小並設在右邊中間	
					ReyeCornerRect = cv::Rect(eyes[j+1].x, eyes[j+1].y + eyes[j+1].height/4, eyes[j+1].width/2, eyes[j+1].height/2) + cv::Point(faces[i].x, faces[i].y);
					ReyeCornerROI = frame_gray( ReyeCornerRect );
					
					cv::resize(LeyeCornerROI, LeyeCornerTpl, cv::Size(LeyeCornerRect.width, LeyeCornerRect.height), 0, 0);//設定Tpl根CornerROI依樣大以比較
					cv::resize(ReyeCornerROI, ReyeCornerTpl, cv::Size(ReyeCornerRect.width, ReyeCornerRect.height), 0, 0);
					if(abs(LeyeROI.cols-ReyeROI.cols)>=(LeyeROI.cols + ReyeROI.cols)/12||abs(LeyeROI.rows-ReyeROI.rows)>=(LeyeROI.rows + ReyeROI.rows)/12)
					{
						LeyeRect.width=0;
						LeyeRect.height=0;
						ReyeRect.width=0;
						ReyeRect.height=0;
					}
				}
			}
		}
	}
}

void trackStage(cv::Mat im, cv::Mat tpl, cv::Rect& rect)
{	
	cv::Size size(rect.width * 2, rect.height * 2);
	cv::Rect window(rect + size - cv::Point(size.width/2, size.height/2 ));	
	window &= cv::Rect(0, 0, im.cols, im.rows);

	cv::Mat dst;

	cv::matchTemplate(im(window), tpl, dst, CV_TM_SQDIFF_NORMED);
	double minval, maxval;
	cv::Point minloc, maxloc;	
	cv::minMaxLoc(dst, &minval, &maxval, &minloc, &maxloc);	
	if (minval <= 0.2)	
	{		
		rect.x = window.x + minloc.x;		
		rect.y = window.y + minloc.y;
	}	
	else
	{	
		rect.x = rect.y = rect.width = rect.height = 0;
	}
}
//畫出內眼角和眼睛ROI
void drawStage(Mat frame_gray, cv::Rect LeyeRect, cv::Rect ReyeRect, Point LeyeCorner, Point ReyeCorner)
{
	rectangle( frame, LeyeRect, CV_RGB(0, 255, 0), 1, 8, 0 );
	rectangle( frame, ReyeRect, CV_RGB(0, 255, 0), 1, 8, 0 );
	circle(frame, LeyeCorner, 1, CV_RGB(255, 255, 0), 2, 8, 0);
	circle(frame, ReyeCorner, 1, CV_RGB(255, 255, 0), 2, 8, 0);
	rectangle(frame, LeyeBallRect, CV_RGB(255,0,0), 1);
	rectangle(frame, ReyeBallRect, CV_RGB(255,0,0), 1);
	rectangle(norImgDraw, LeyePupilRect, CV_RGB(255, 0, 0), 2, 8, 0);
	rectangle(norImgDraw, ReyePupilRect, CV_RGB(255, 0, 0), 2, 8, 0);
	circle(norImgDraw, LeyePupil, 1, CV_RGB(255, 255, 255), 2, 8, 0);
	circle(norImgDraw, ReyePupil, 1, CV_RGB(255, 255, 255), 2, 8, 0);
}

void detectEyeball(Mat src, Rect &eyeBallRect)
{
	cv::Mat gray;
	cv::cvtColor(~src, gray, CV_BGR2GRAY);
	double maxarea=0;
	// Invert the source image and convert to grayscale
	cv::threshold(gray, gray, 240, 255, cv::THRESH_BINARY);
		
	cv::dilate(gray,gray,NULL,Point(0,0),40);
	cv::erode(gray,gray,NULL,Point(0,0),100);
	imshow("gray3",gray);
	// Find all contours
	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(gray.clone(), contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	// Fill holes in each contour
	cv::drawContours(gray, contours, -1, CV_RGB(255,255,255), -1);

	for (int i = 0; i < contours.size(); i++)
	{
		double area = cv::contourArea(contours[i]);
		cv::Rect rect = cv::boundingRect(contours[i]);

		// If contour is big enough and has round shape
		// Then it is the pupil
		if (area>maxarea&&area>=70)	
		{
			maxarea=area;
			eyeBallRect=rect;
		}
		rectangle(norImgDraw, rect, CV_RGB(255, 255, 255), 2, 8, 0);
	}
}
void detectPupil(Mat src, Rect &LeyePupilRect, Rect &ReyePupilRect)
	{
	Pupilcheck=false;
	cv::Mat gray;
	threshold(src, src, 20, 255,CV_THRESH_BINARY);
	cv::cvtColor(~src, gray, CV_BGR2GRAY);

	equalizeHist( gray, gray );

	double Larea=0;
	double Rarea=0;
	// Invert the source image and convert to grayscale

	cv::dilate(src,src,NULL,Point(0,0),20);
	cv::erode(src,src,NULL,Point(0,0),40);
	cv::dilate(src,src,NULL,Point(0,0),20);
	cv::erode(src,src,NULL,Point(0,0),200);
	cv::Canny(src,src,5,70,3);
	// Find all contours
	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(gray.clone(), contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	// Fill holes in each contour
	cv::drawContours(gray, contours, -1, CV_RGB(255,255,255), -1);

	LeyePupilRect=Rect(0,0,0,0);
	ReyePupilRect=Rect(0,0,0,0);

	for (int i = 0; i < contours.size(); i++)
		{
		double area = cv::contourArea(contours[i]);
		cv::Rect rect = cv::boundingRect(contours[i]);
		// If contour is big enough and has round shape
		// Then it is the pupil
		if (contours.size()>=2&&area>Larea&&area>600)
			{
			LeyePupilRect=rect;
			}
		else if(contours.size()>=2&&area>Rarea&&area<Larea&&area>600)
			{
			ReyePupilRect=rect;
			}
		
		if(contours.size()>=2&&LeyePupilRect.x>ReyePupilRect.x)
			{
			cv::Rect tmpRect;
			tmpRect=LeyePupilRect;
			LeyePupilRect=ReyePupilRect;
			ReyePupilRect=tmpRect;
			}
		}
	if(LeyePupilRect.width/2>0&&LeyePupilRect.height/2>0&&ReyePupilRect.width/2>0&&ReyePupilRect.height/2>0){
		LeyePupil=cv::Point(LeyePupilRect.width/2,LeyePupilRect.height/2);
		ReyePupil=cv::Point(ReyePupilRect.width/2,ReyePupilRect.height/2);
		Pupilcheck=true;
		}
	else
		Pupilcheck=false;
	rectangle(gray,LeyePupilRect, CV_RGB(255,255,255), 1);
	rectangle(gray,ReyePupilRect, CV_RGB(255,255,255), 1);

	imshow("src1",gray);
	

		/*
	cv::GaussianBlur(src,src,cv::Size(3,3),2,2);
	
	//vector<Vec3f> circles;
	//HoughCircles(src, circles, CV_HOUGH_GRADIENT, 3, 32, 50, 100 );
	
	for( size_t i = 0; i < circles.size(); i++ )
		{
		Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
		int radius = cvRound(circles[i][2]);
		// draw the circle center
		circle( src, center, 3, Scalar(255,255,255), -1, 8, 0 );
		// draw the circle outline
		circle( src, center, radius, Scalar(255,255,255), 3, 8, 0 );
		}
		*/
	}
void ecExecution()
		{
		//change color to gray and do equalize Histogram
		Mat frame_gray;
		cvtColor( frame, frame_gray, CV_BGR2GRAY );
		equalizeHist( frame_gray, frame_gray );
		//Detect faces
		//若左右眼的Rect不存在則detection
		if(LeyeRect.width==0||LeyeRect.height==0||ReyeRect.width==0||ReyeRect.height==0||LeyeCornerRect.width==0||LeyeCornerRect.height==0||ReyeCornerRect.width==0||ReyeCornerRect.height==0)
			{
			detectStage(frame_gray,
				LeyeROI,
				ReyeROI,
				LeyeCornerTpl,
				ReyeCornerTpl,
				LeyeRect,
				ReyeRect,
				LeyeCornerRect,
				ReyeCornerRect,
				LeyeCorner,
				ReyeCorner
				);
			norImgTplcounter=0;
		}
		//若存在則tracking
		else if(LeyeRect.width!=0&&LeyeRect.height!=0&&ReyeRect.width!=0&&ReyeRect.height!=0&&LeyeCornerRect.width!=0&&LeyeCornerRect.height!=0&&ReyeCornerRect.width!=0&&ReyeCornerRect.height!=0)
		{
			trackStage(frame_gray, LeyeROI, LeyeRect);//在影像中追蹤眼睛ROI
			trackStage(frame_gray, ReyeROI, ReyeRect);

			if(LeyeRect.width!=0&&LeyeRect.height!=0&&ReyeRect.width!=0&&ReyeRect.height!=0)
			{
				detectEyeball(frame(LeyeRect), LeyeBallRect);
				detectEyeball(frame(ReyeRect), ReyeBallRect);

				LeyeBallRect = LeyeBallRect + cv::Point(LeyeRect.x,LeyeRect.y);
				ReyeBallRect = ReyeBallRect + cv::Point(ReyeRect.x,ReyeRect.y);
		
				LeyeCornerRect.x = LeyeCornerRect.y = ReyeCornerRect.x = ReyeCornerRect.y = 0;//眼角ROI起始先設為零 預防TRACCKING出錯

				trackStage(frame_gray(Rect(LeyeRect.x + LeyeRect.width/2, LeyeRect.y, LeyeRect.width/2, LeyeRect.height)), LeyeCornerTpl, LeyeCornerRect);//在眼睛ROI中追蹤眼角ROI
				trackStage(frame_gray(Rect(ReyeRect.x, ReyeRect.y, ReyeRect.width/2, ReyeRect.height)), ReyeCornerTpl, ReyeCornerRect);

				LeyeCornerRect = LeyeCornerRect + cv::Point(LeyeRect.x, LeyeRect.y);//補回眼角ROI起始位址
				ReyeCornerRect = ReyeCornerRect + cv::Point(ReyeRect.x, ReyeRect.y);

				LeyeCorner = cv::Point(LeyeCornerRect.x + LeyeCornerRect.width/2 + LeyeRect.width/2, LeyeCornerRect.y + LeyeCornerRect.height/2);//設定眼角ROI的正中央為眼角點
				ReyeCorner = cv::Point(ReyeCornerRect.x + ReyeCornerRect.width/2, ReyeCornerRect.y + ReyeCornerRect.height/2);

				if((LeyeCornerRect.width!=0||LeyeCornerRect.height!=0)&&(ReyeCornerRect.width!=0||ReyeCornerRect.height!=0))
				{
					MattoArr(frame_gray, imgArr);
					Normal_2Points(imgArr,frame.rows, frame.cols, LeyeCorner.x,LeyeCorner.y, ReyeCorner.x ,ReyeCorner.y, oimgArr);
					ArrtoMat(norImg, oimgArr);
					/*
					norImgDraw = norImg;
					if(norImgTplcounter==0)
						{
						norImgRect = Rect(0, 0, norImg.cols, norImg.rows);
						norImgTpl = norImg;
						norImgTplcounter++;
						}
					else
						{
						trackStage(norImg, norImgTpl, norImgRect);
						norImgTpl = norImg;
						}
					*/
					imshow("norImg5",norImg);
					GaussianBlur( norImg, norImg, Size(9, 9), 2, 2 );
                    
				    cvtColor( norImg, norImgc, CV_GRAY2RGB );
					//cv::adaptiveThreshold(norImg,norImg,255,CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY,55,7);

					//threshold(norImg, norImg, 19, 255,CV_THRESH_BINARY);
					detectPupil(norImgc, LeyePupilRect, ReyePupilRect);

					//畫出內眼角和眼睛ROI
					drawStage(frame, 
						LeyeRect, 
						ReyeRect,
						LeyeCorner,
						ReyeCorner
						);
					//若兩眼ROI間距過小(即ROI重疊)則重設RECT為零 令下一張圖重新DETECT
					if(abs((LeyeRect.x+LeyeRect.width/2)-(ReyeRect.x+ReyeRect.width/2)) < max(LeyeRect.width, ReyeRect.width))
					{
						LeyeRect.x = LeyeRect.y = LeyeRect.width = LeyeRect.height = 0;
						ReyeRect.x = ReyeRect.y = ReyeRect.width = ReyeRect.height = 0;
					}
				}
			}
		}
}

int main( void )
{	
	oimgArr=new double[NOR_H*NOR_W];
	norImg.create(NOR_H, NOR_W, 0);
	norImgc.create(NOR_H, NOR_W, 1);

	//-- 1. Load the cascade
	if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
	if( !eyes_cascade.load( eyes_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

	//-- 2. Read the video stream
	//攝影機讀取
	capture = cvCaptureFromCAM( -1 );
	while (cv::waitKey(15) != 'q')
	{
		//輸入攝影機影像
		frame = cvQueryFrame( capture );
		imgArr=new double[frame.rows*frame.cols];
		//判斷是否讀入影像
		if(frame.empty())
		{ 
			printf(" --(!) No captured frame -- Break!"); 
			break;  
		}
		ecExecution();
		
		MainRect=cv::Rect(frame.cols/3, frame.rows/6, frame.cols/3, frame.rows*0.65);
		rectangle(frame, MainRect, CV_RGB(255,0,0), 1);

		imshow( face_name, frame );
		imshow("norImg", norImg);
		delete imgArr;
	}
	return 0;
}
