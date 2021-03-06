## [Rubric](https://review.udacity.com/#!/rubrics/2549/view) Points
---
### 1. Data Buffer

#### MP.1 Data Buffer Optimization
* Implement a vector for dataBuffer objects whose size does not exceed a limit (e.g. 2 elements). 
* This can be achieved by pushing in new elements on one end and removing elements on the other end.
* Solution: Lines 119 ~ 123 at `MidTermProject_Camera_Student.cpp`
```c++
DataFrame frame;
frame.cameraImg = imgGray;
 if( dataBuffer.size()+1>dataBufferSize)
{
	dataBuffer.erase(dataBuffer.begin());
        std::cout<<"Erasing and replacing ring databuffer of size"<<dataBufferSize<<"Done!"<<std::endl;
}
dataBuffer.push_back(frame);
```

### 2. Keypoints

#### MP.2 Keypoint Detection
* Implement detectors HARRIS, FAST, BRISK, ORB, AKAZE, and SIFT and make them selectable by setting a string accordingly.
* Solution : Lines 138 ~ 154 at `MidTermProject_Camera_Student.cpp`
```c++

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
                   detKeypointsModern(keypoints,imgGray,detectorType,false);}
                
else
{
	 throw std::invalid_argument("pls. enter correct detector type");
}
```
* Solution : Lines 162 ~ 285 at `matching2D_Student.cpp`
```c++

// detectorType = HARRIS
void detKeypointsHarris(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // Detector parameters
    int blockSize = 2;     // for every pixel, a blockSize × blockSize neighborhood is considered
    int apertureSize = 3;  // aperture parameter for Sobel operator (must be odd)
    int minResponse = 100; // minimum value for a corner in the 8bit scaled response matrix
    double k = 0.04;       // Harris parameter (see equation for details)

    double t = (double)cv::getTickCount();

    // Detect Harris corners and normalize output
    cv::Mat dst, dst_norm, dst_norm_scaled;
    dst = cv::Mat::zeros(img.size(), CV_32FC1);
    cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    cv::convertScaleAbs(dst_norm, dst_norm_scaled);

    // Look for prominent corners and instantiate keypoints

    double maxOverlap = 0.0; // max. permissible overlap between two features in %, used during non-maxima suppression
    for (size_t j = 0; j < dst_norm.rows; j++)
    {
        for (size_t i = 0; i < dst_norm.cols; i++)
        {
            int response = (int)dst_norm.at<float>(j, i);
            if (response > minResponse)
            { // only store points above a threshold

                cv::KeyPoint newKeyPoint;
                newKeyPoint.pt = cv::Point2f(i, j);
                newKeyPoint.size = 2 * apertureSize;
                newKeyPoint.response = response;

                // perform non-maximum suppression (NMS) in local neighbourhood around new key point
                bool bOverlap = false;
                for (auto it = keypoints.begin(); it != keypoints.end(); ++it)
                {
                    double kptOverlap = cv::KeyPoint::overlap(newKeyPoint, *it);
                    if (kptOverlap > maxOverlap)
                    {
                        bOverlap = true;
                        if (newKeyPoint.response > (*it).response)
                        {                      // if overlap is >t AND response is higher for new kpt
                            *it = newKeyPoint; // replace old key point with new one
                            break;             // quit loop over keypoints
                        }
                    }
                }
                if (!bOverlap)
                {                                     // only add new key point if no overlap has been found in previous NMS
                    keypoints.push_back(newKeyPoint); // store new keypoint in dynamic list
                }
            }
        } // eof loop over cols
    }     // eof loop over rows

    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "Harris corner detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Harris Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

// detectorType = FAST, BRISK, ORB, AKAZE, SIFT
void detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, std::string detectorType, bool bVis)
{
    double t = (double)cv::getTickCount();
    cv::Ptr<cv::Feature2D> detector;

    if(detectorType.compare("FAST") == 0)
    {
        detector = cv::FastFeatureDetector::create();
        detector->detect(img, keypoints);
    }
    else if(detectorType.compare("BRISK") == 0)
    {
        detector = cv::BRISK::create();
        detector->detect(img, keypoints);
    }
    else if(detectorType.compare("ORB") == 0)
    {
        detector = cv::ORB::create();
        detector->detect(img, keypoints);
    }
    else if(detectorType.compare("AKAZE") == 0)
    {
        detector = cv::AKAZE::create();
        detector->detect(img, keypoints);   
    }
    else if(detectorType.compare("SIFT") == 0)
    {
        detector = cv::xfeatures2d::SIFT::create();
        detector->detect(img, keypoints);        
    }
    else
    {
        throw invalid_argument(detectorType + " is not a valid detectorType. Try FAST, BRISK, ORB, AKAZE, SIFT.");
    }

    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << detectorType << " detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = detectorType + " Keypoint Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}
```

#### MP.3 Keypoint Removal
* Remove all keypoints outside of a pre-defined rectangle and only use the keypoints within the rectangle.
* Solution code: Lines 162 ~ 181 at `MidTermProject_Camera_Student.cpp`
```c++
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
}
```
### 3. Descriptors

#### MP.4 Keypoint Descriptors
* Implement descriptors BRIEF, ORB, FREAK, AKAZE and SIFT and make them selectable by setting a string accordingly.
* Solution code: Lines 240 ~ 249
 at `MidTermProject_Camera_Student.cpp`
```c++

string descriptorType;
if (descriptorType.compare("SIFT") == 0) 
{
    descriptorType == "DES_HOG";
}
else
{
    descriptorType == "DES_BINARY";
}                      
```
* Solution code: In function `descKeypoints`, Lines 63 ~ 95 at `matching2D_Student.cpp`
```c++
// ...add start: MP.4 Keypoint Descriptors
else if(descriptorType.compare("BRIEF") == 0)
{
    extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
}
else if(descriptorType.compare("ORB") == 0)
{
    extractor = cv::ORB::create();
}
else if(descriptorType.compare("AKAZE") == 0)
{
    extractor = cv::AKAZE::create();
}
else if(descriptorType.compare("FREAK") == 0)
{
    extractor = cv::xfeatures2d::FREAK::create();
}
else if(descriptorType.compare("SIFT") == 0)
{
    extractor = cv::xfeatures2d::SIFT::create();
}
else
{
    throw invalid_argument( "The input method is not supported. Try BRIEF, BRISK, ORB, AKAZE, FREAK, SIFT." );
}
```

#### MP.5 Descriptor Matching
* Implement FLANN matching as well as k-nearest neighbor selection. 
* Both methods must be selectable using the respective strings in the main function.
* Solution code: In function `matchDescriptors`, Lines 14 ~ 30 at `matching2D_Student.cpp`
```c++
if (matcherType.compare("MAT_BF") == 0)
{
    int normType = cv::NORM_L2;

    if(descriptorType.compare("DES_BINARY") == 0)
    {
        normType = cv::NORM_HAMMING;        
    }        
    matcher = cv::BFMatcher::create(normType, crossCheck);
    cout << "BF matching cross-check=" << crossCheck;
}    
else if (matcherType.compare("MAT_FLANN") == 0)
{
    // OpenCV bug workaround : convert binary descriptors to floating point due to a bug in current OpenCV implementation
    if (descSource.type() != CV_32F)
    { 
        descSource.convertTo(descSource, CV_32F);
        descRef.convertTo(descRef, CV_32F);
    }

    matcher = cv::FlannBasedMatcher::create();              
}
```
#### MP.6 Descriptor Distance Ratio
* Use the K-Nearest-Neighbor matching to implement the descriptor distance ratio test, which looks at the ratio of best vs. second-best match to decide whether to keep an associated pair of keypoints.
* Solution code: Lines 249 at `MidTermProject_Camera_Student.cpp`
```c++

string selectorType = "SEL_KNN";       // SEL_NN, SEL_KNN
```
* Solution code: In function `matchDescriptors`, Lines 39 ~ 54 at `matching2D_Student.cpp`
```
else if (selectorType.compare("SEL_KNN") == 0)
{ // k nearest neighbors (k=2)


    vector<vector<cv::DMatch>> knn_matches;
    matcher->knnMatch(descSource, descRef, knn_matches, 2);

    double minDescDistRatio = 0.8;
    for(auto it = knn_matches.begin(); it != knn_matches.end(); ++it)
    {
        if( ((*it)[0].distance) < ((*it)[1].distance * minDescDistRatio) )
        {
            matches.push_back((*it)[0]);
        }                
    }
    cout << "# keypoints removed = " << knn_matches.size() - matches.size() << endl;

}
```

### 4. Performance
---
#### MP.7 Performance Evaluation 1
* Count the number of keypoints on the preceding vehicle for all 10 images and take note of the distribution of their neighborhood size. 
* Do this for all the detectors you have implemented.
* Solution Result: please check `AlgorithmComparision/Keypoints.csv` file [CSV file](https://github.com/aditya-167/2D-feature-tracking-CV/blob/master/AlgorithmComparision/Keypoints.csv).

DETECTOR  | Number of keypoints
--------  | -------------------
SHITOMASI | 111 ~ 125
HARRIS    |  14 ~  43
FAST      | 386 ~ 427
BRISK     | 254 ~ 297
ORB       |  92 ~ 130
AKAZE     | 155 ~ 179
SIFT      | 124 ~ 159

HARRIS detector has the smallest amount of keypoints.
FAST detector has the bigest amount of keypoints.

#### MP.8 Performance Evaluation 2
* Count the number of matched keypoints for all 10 images using all possible combinations of detectors and descriptors. 
* In the matching step, the BF approach is used with the descriptor distance ratio set to 0.8.
* Solution Result: please check `AlgorithmComparision/Matched_Keypoints.csv` file [CSV file](https://github.com/aditya-167/2D-feature-tracking-CV/blob/master/AlgorithmComparision/Matched_Keypoints.csv).

#### MP.9 Performance Evaluation 3
* Log the time it takes for keypoint detection and descriptor extraction. 
* The results must be entered into a spreadsheet and based on this data, the TOP3 detector / descriptor combinations must be recommended as the best choice for our purpose of detecting keypoints on vehicles.
* Solution Result: please check `AlgorithmComparision/Log_time.csv` file [CSV file](https://github.com/studian/SFND_P3_2D_Feature_Tracking/MP_9_Log_Time.csv).

Considering `AlgorithmComparision/Matched_Keypoints.csv.csv` and `AlgorithmComparision/Log_Time.csv` The TOP3 detector / descriptor combinations as the best choice for our purpose of detecting keypoints on vehicles are:

Rank  |  Detector/Descriptor  | The Average Number of Keypoints | Average Time
------|---------------------- | --------------------------------| --------
1st   |FAST/BRIEF             | 242 keypoints                   |  7.22 ms
2nd   |FAST/ORB               | 229 keypoints                   |  7.45 ms 
3rd   |FAST/SIFT              | 247 keypoints                   |  15.38 ms

---
* Solution code of `MP.7`, `MP.8`, and `MP.9`: Lines 44 ~ 91 at `MidTermProject_Camera_Student.cpp`
```c++
 vector<std::string> detectortype = {"HARRIS", "SHITOMASI", "FAST", "ORB", "AKAZE", "BRISK", "SIFT"};
    vector<std::string> descriptortype = {"BRISK", "SIFT", "ORB", "FREAK", "BRIEF", "AKAZE"};

    ofstream kpsaveFileCSV;
    kpsaveFileCSV.open ("../AlgorithmComparision/Keypoints.csv");

    ofstream matchedKepointsCSV;
    matchedKepointsCSV.open ("../AlgorithmComparision/Matched_Keypoints.csv");

    ofstream timeLogCSV;
    timeLogCSV.open ("../AlgorithmComparision/Log_Time.csv");


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
```
* Solution code of `MP.7`, `MP.8`, and `MP.9`: Lines 132 at 

`MidTermProject_Camera_Student.cpp`
```c++
string detectorType = i_detector; //"SHITOMASI", "HARRIS", "FAST", "BRISK", "ORB", "AKAZE", "SIFT"
```

* Solution code of `MP.7`: Lines 186 at `MidTermProject_Camera_Student.cpp`
```c++

 if(!write){
                    kpsaveFileCSV<<", "<<keypoints.size();
                }             
```

* Solution code of `MP.7`, `MP.8`, and `MP.9`: Lines 214 at `MidTermProject_Camera_Student.cpp`

```c++
string descriptorType = i_descriptor; // BRIEF, ORB, FREAK, AKAZE, SIFT
```

* Solution code of `MP.8` and `MP.9`: Lines 265 ~ 269 at `MidTermProject_Camera_Student.cpp`

```c++
matchedKepointsCSV << ", " << matches.size();

time_taken = ((double)cv::getTickCount() - time_taken) / cv::getTickFrequency();
matchedKepointsCSV << ", " << 1000*time_taken;

```
* Solution code of `MP.7`, `MP.8`, and `MP.9`: Lines 296 ~ 314 at `MidTermProject_Camera_Student.cpp`
```c++
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
```

