/* INCLUDES FOR THIS PROJECT */
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "dataStructures.h"
#include "matching2D.hpp"

using namespace std;
/* MAIN PROGRAM */
int main(int argc, const char *argv[])
{

    /* INIT VARIABLES AND DATA STRUCTURES */

    // data location
    std::string dataPath = "../";

    // camera
    std::string imgBasePath = dataPath + "images/";
    std::string imgPrefix = "KITTI/2011_09_26/image_00/data/000000"; // left camera, color
    std::string imgFileType = ".png";
    int imgStartIndex = 0; // first file index to load (assumes Lidar and camera names have identical naming convention)
    int imgEndIndex = 9;   // last file index to load
    int imgFillWidth = 4;  // no. of digits which make up the file index (e.g. img-0001.png)

    // misc
    int dataBufferSize = 2;       // no. of images which are held in memory (ring buffer) at the same time
    vector<DataFrame> dataBuffer; // list of data frames which are held in memory at the same time
    bool bVis = false;            // visualize results


    /* set detector types and descriptor types for comparision*/
    vector<std::string> detectortype = {"HARRIS", "SHITOMASI", "FAST", "ORB", "AKAZE", "BRISK", "SIFT"};
    vector<std::string> descriptortype = {"BRISK", "SIFT", "ORB", "FREAK", "BRIEF", "AKAZE"};

    ofstream kpsaveFileCSV;
    kpsaveFileCSV.open ("../AlgorithmComparision/Keypoints.csv");

    ofstream matchedKepointsCSV;
    matchedKepointsCSV.open ("../AlgorithmComparision/Matched_Keypoints.csv");

    ofstream timeLogCSV;
    timeLogCSV.open ("../Log_Time.csv");


    //Note : Selecting each and every combination of Descriptors and mathers is cumbersome for logging values
    // hence I have looped through all detector types and its all possible descriptors and saved the result in csv file.

    for (auto i_detector:detectortype) //loop for every detector types
    {
        bool write = false;

        for(auto i_descriptor:descriptortype)// start loop descriptor_types
        { 
            //class implementing AKAZE keypoint detecot and descriptor can only be used with KAZE/AKAZE keypoints
            if(i_detector.compare("AKAZE")!=0 && i_descriptor.compare("AKAZE")==0)
                continue;

            if(i_detector.compare("AKAZE")==0 && i_descriptor.compare("AKAZE")==0)
                continue; 


            dataBuffer.clear();

            std::cout << "*************************************************************************" << std::endl;
            std::cout << "Detector Type: "<< i_detector <<" Descriptor Type:" << i_descriptor <<std::endl;
            std::cout << "*************************************************************************" << std::endl;

            //save performance evaluation 1 MP.7, keypoints of each detector to the csv file
            if (!write){
                kpsaveFileCSV<<i_detector;
            }

            // save MP.8 Performance Evaluaion 2
            matchedKepointsCSV<<i_detector<<"+"<<i_descriptor;

            //save MP.9 Performance Evaluaion 3, Log the time taken by detectors and descriptors
            timeLogCSV<<i_detector<<"+"<<i_descriptor;

            /* MAIN LOOP OVER ALL IMAGES */

            for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex++)
            {

                //MP.9 Evaluation 3 for logging time taken by each comibnation of detectors and descriptors
                double time_taken = (double)cv::getTickCount();


                /* LOAD IMAGE INTO BUFFER */

                // assemble filenames for current index
                ostringstream imgNumber;
                imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + imgIndex;
                std::string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

                // load image from file and convert to grayscale
                cv::Mat img, imgGray;
                img = cv::imread(imgFullFilename);
                cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);

                //// STUDENT ASSIGNMENT
                //// TASK MP.1 -> replace the following code with ring buffer of size dataBufferSize

                // push image into data frame buffer
                DataFrame frame;
                frame.cameraImg = imgGray;

                if( dataBuffer.size()+1>dataBufferSize){
                    dataBuffer.erase(dataBuffer.begin());
                    std::cout<<"Erasing and replacing ring databuffer of size "<<dataBufferSize<<"Done!"<<std::endl;
                }
                dataBuffer.push_back(frame);

                //// EOF STUDENT ASSIGNMENT
                std::cout << "#1 : LOAD IMAGE INTO BUFFER done" << std::endl;

                /* DETECT IMAGE KEYPOINTS */

                // extract 2D keypoints from current image
                vector<cv::KeyPoint> keypoints; // create empty feature list for current image
                std::string detectorType = i_detector;

                //// STUDENT ASSIGNMENT
                //// TASK MP.2 -> add the following keypoint detectors in file matching2D.cpp and enable std::string-based selection based on detectorType
                //// -> HARRIS, FAST, BRISK, ORB, AKAZE, SIFT

                if (detectorType.compare("SHITOMASI") == 0)
                {
                    detKeypointsShiTomasi(keypoints, imgGray, false);
                }
            
                else if (detectorType.compare("HARRIS") == 0) 
                {
                    detKeypointsHarris(keypoints, imgGray, false);
                }

                //modern detectors 
                else if (detectorType.compare("FAST")  == 0 || detectorType.compare("BRISK") == 0 ||detectorType.compare("ORB")   == 0 || detectorType.compare("AKAZE") == 0 || detectorType.compare("SIFT")  == 0){
                    detKeypointsModern(keypoints,imgGray,detectorType,false);
                }
                else{
                    throw std::invalid_argument("pls. enter correct detector type");
                }
                //// EOF STUDENT ASSIGNMENT

                //// STUDENT ASSIGNMENT
                //// TASK MP.3 -> only keep keypoints on the preceding vehicle

                // only keep keypoints on the preceding vehicle
                bool bFocusOnVehicle = true;
                cv::Rect vehicleRect(535, 180, 180, 150);
                //keypoint removal and roi
                vector<cv::KeyPoint>::iterator keyptsptr;
                vector<cv::KeyPoint> keypts_roi;
                if (bFocusOnVehicle)
                {
                    //remove keypoints for roi
                    for(keyptsptr = keypoints.begin(); keyptsptr != keypoints.end(); ++keyptsptr)
                    {
                        if (vehicleRect.contains(keyptsptr->pt))
                        {  
                            cv::KeyPoint newKeyPoint;
                            newKeyPoint.pt = cv::Point2f(keyptsptr->pt);
                            newKeyPoint.size = 1;
                            keypts_roi.push_back(newKeyPoint);
                        }
                    }
                    keypoints = keypts_roi;
                    std::cout << "ROI size= " << keypoints.size()<<" keypoints"<<std::endl;
                }

                //// EOF STUDENT ASSIGNMENT

                if(!write){
                    kpsaveFileCSV<<", "<<keypoints.size();
                }

                // optional : limit number of keypoints (helpful for debugging and learning)
                bool bLimitKpts = false;
                if (bLimitKpts)
                {
                    int maxKeypoints = 50;

                    if (detectorType.compare("SHITOMASI") == 0)
                    { // there is no response info, so keep the first 50 as they are sorted in descending quality order
                        keypoints.erase(keypoints.begin() + maxKeypoints, keypoints.end());
                    }
                    cv::KeyPointsFilter::retainBest(keypoints, maxKeypoints);
                    std::cout << " NOTE: Keypoints have been limited!" << std::endl;
                }

                // push keypoints and descriptor for current frame to end of data buffer
                (dataBuffer.end() - 1)->keypoints = keypoints;
                std::cout << "#2 : DETECT KEYPOINTS done" << std::endl;

                /* EXTRACT KEYPOINT DESCRIPTORS */

                //// STUDENT ASSIGNMENT
                //// TASK MP.4 -> add the following descriptors in file matching2D.cpp and enable std::string-based selection based on descriptorType
                //// -> BRIEF, ORB, FREAK, AKAZE, SIFT

                cv::Mat descriptors;
                std::string descriptorType = i_descriptor; // BRIEF, ORB, FREAK, AKAZE, SIFT
                descKeypoints((dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->cameraImg, descriptors, descriptorType);
                //// EOF STUDENT ASSIGNMENT

                // push descriptors for current frame to end of data buffer
                (dataBuffer.end() - 1)->descriptors = descriptors;

                std::cout << "#3 : EXTRACT DESCRIPTORS done\n" << std::endl;

                if (dataBuffer.size() > 1) // wait until at least two images have been processed
                {

                    /* MATCH KEYPOINT DESCRIPTORS */

                    vector<cv::DMatch> matches;
                    std::string matcherType = "MAT_BF";        // MAT_BF, MAT_FLANN
                    
                    //std::string descriptorType = "DES_BINARY"; // DES_BINARY, DES_HOG
                    //tring selectorType = "SEL_NN";       // SEL_NN, SEL_KNN

                    //// STUDENT ASSIGNMENT
                    //// TASK MP.5 -> add FLANN matching in file matching2D.cpp
                    //// TASK MP.6 -> add KNN match selection and perform descriptor distance ratio filtering with t=0.8 in file matching2D.cpp

                    //std::string descriptorType;
                    
                    if (descriptorType.compare("SIFT") == 0) 
                    {
                        descriptorType == "DES_HOG";
                    }
                    else
                    {
                        descriptorType == "DES_BINARY";
                    }                    
                    
                    std::string selectorType = "SEL_KNN";       // SEL_NN, SEL_KNN
                    
                    
                    
                    
                    matchDescriptors((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints,
                                    (dataBuffer.end() - 2)->descriptors, (dataBuffer.end() - 1)->descriptors,
                                    matches, descriptorType, matcherType, selectorType);

                    //// EOF STUDENT ASSIGNMENT

                    // store matches in current data frame
                    (dataBuffer.end() - 1)->kptMatches = matches;


                    // MP.8 Performance Evaluation 2, saving mathes  for each detector and discriptor types in csv file defined above
                    matchedKepointsCSV << ", " << matches.size();
                    
                    // MP.9 Performance Evaluation 3, logging time taken  for each detector and discriptor types in csv file defined above
                    time_taken = ((double)cv::getTickCount() - time_taken) / cv::getTickFrequency();
                    timeLogCSV << ", " << 1000*time_taken;
                    
                    std::cout << "#4 : MATCH KEYPOINT DESCRIPTORS done" << std::endl;

                    // visualize matches between current and previous image
                    bVis = true;
                    if (bVis)
                    {
                        cv::Mat matchImg = ((dataBuffer.end() - 1)->cameraImg).clone();
                        cv::drawMatches((dataBuffer.end() - 2)->cameraImg, (dataBuffer.end() - 2)->keypoints,
                                        (dataBuffer.end() - 1)->cameraImg, (dataBuffer.end() - 1)->keypoints,
                                        matches, matchImg,
                                        cv::Scalar::all(-1), cv::Scalar::all(-1),
                                        vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

                        std::string windowName = "Matching keypoints between two camera images";
                        cv::namedWindow(windowName, 7);
                        cv::imshow(windowName, matchImg);
                        std::cout << "Press key to continue to next image" << std::endl;
                        cv::waitKey(0); // wait for key to be pressed
                    }
                    bVis = false;
                }

            } // eof loop over all images

            // writing all the data to csv files and get ready for next iteration.
            if(!write)
            {
                kpsaveFileCSV << std::endl;   
            }

            write = true;
            matchedKepointsCSV << std::endl;
            timeLogCSV << std::endl;



            
        }// eof loop over descriptor types
    }//eof loop over detecrtor types

    //close the csv files
    kpsaveFileCSV.close();
    matchedKepointsCSV.close();
    timeLogCSV.close();


    
    return 0;
}
