//--------------------------------------------------
// Startup code module
//
// @author: Wild Boar
//
// @date: 2023-01-11
//--------------------------------------------------

#include <iostream>
using namespace std;

#include <NVLib/Logger.h>
#include <NVLib/FileUtils.h>
#include <NVLib/DisplayUtils.h>
#include <NVLib/RandomUtils.h>
#include <NVLib/Model/StereoFrame.h>
#include <NVLib/ImageUtils.h>
#include <NVLib/Parameters/Parameters.h>

#include <opencv2/opencv.hpp>
using namespace cv;

#include "ArgReader.h"

//--------------------------------------------------
// Solution Class
//--------------------------------------------------

class Solution 
{
private:
    Mat _disparity;
    Mat _score;
public:
    Solution(const Size& size, int min = 0, int max = 180)
    {
        // Create the random disparity
        _disparity = Mat_<float>(size); auto ddata = (float *) _disparity.data;
        auto pixelCount = _disparity.rows * _disparity.cols;
        for (auto i = 0; i < pixelCount; i++) ddata[i] = NVLib::RandomUtils::GetInteger(min, max);

        // Create a score map
        _score = Mat_<float>(size); _score.setTo(FLT_MAX);
    }

    /**
     * @brief Update the score (if the score is better)
     * @param point The location that we want to update the score 
     * @param score The score we are updating
     * @param disparity The disparity value
     * @return Return true if score was an improvement, otherwise reject the change and return false 
     */
    inline bool UpdateScore(const Point& point, float score, float disparity) 
    {
        auto index = point.x + point.y * _disparity.cols;
        auto scoreData = (float *) _score.data; auto disparityData = (float *) _disparity.data;
        auto currentScore = scoreData[index]; if (score > currentScore) return false; // This score is worse so dont update!
        scoreData[index] = score; disparityData[index] = disparity;
        return true;
    }

    /**
     * @brief Retrieve the disparity at a given location
     * @param row The row that we are getting the value for 
     * @param column The column that we are getting the value for
     * @return float The resultant disparity value
     */
    inline float GetDisparity(int row, int column) 
    {
        auto index = column + row * _disparity.cols;
        auto data = (float *) _disparity.data;
        return data[index];
    }

    inline Mat& GetDisparity() { return _disparity; }
    inline Mat& GetScore() { return _score;}
};

//--------------------------------------------------
// Function Prototypes
//--------------------------------------------------
void Run(NVLib::Parameters * parameters);
NVLib::StereoFrame ReadFrame(const string& folder, int index);
NVLib::StereoFrame GetGradientFrame(NVLib::StereoFrame& frame);
float GetScore(NVLib::StereoFrame& frame, NVLib::StereoFrame& gradientFrame, int blockSize, const Point& point, float disparity);
Mat GetBlock(Mat& image, const Point2f& point, int blockSize);
Mat Float2Color(Mat& image);

//--------------------------------------------------
// Execution Logic
//--------------------------------------------------

/**
 * Main entry point into the application
 * @param parameters The input parameters
 */
void Run(NVLib::Parameters * parameters) 
{
    // Verify that we have some input parameters
    if (parameters == nullptr) return; auto logger = NVLib::Logger(1); cv::setUseOptimized(true);

    logger.Log(1, "Initializing random variables");
    NVLib::RandomUtils::TimeSeedRandomNumbers();

    logger.Log(1, "Load up input values");
    auto inputFolder = NVL_Utils::ArgReader::ReadString(parameters, "input");
    auto outputFolder = NVL_Utils::ArgReader::ReadString(parameters, "output");
    auto index = NVL_Utils::ArgReader::ReadInteger(parameters, "index");
    auto blockSize = NVL_Utils::ArgReader::ReadInteger(parameters, "block_size");
    auto maxDisparity = NVL_Utils::ArgReader::ReadInteger(parameters, "disparity");

    logger.Log(1, "Reading the image frame from disk");
    auto frame = ReadFrame(inputFolder, index);

    logger.Log(1, "Creating a gradient frame for the image pair");
    auto gradientFrame = GetGradientFrame(frame);

    logger.Log(1, "Creating a container for holding the solution");
    auto solution = Solution(frame.GetSize(), 0, maxDisparity);

    logger.Log(1, "Initializing Scores");
    #pragma omp parallel for
    for (auto row = blockSize; row < frame.GetSize().height - blockSize; row++) 
    {
        for (auto column = blockSize; column < frame.GetSize().width - blockSize; column++) 
        {
            auto disparity = solution.GetDisparity(row, column);
            auto score = GetScore(frame, gradientFrame, blockSize, Point(column, row), disparity);
            solution.UpdateScore(Point(column, row), score, disparity);
        }
    }

    for (auto i = 0; i < 30; i++) 
    {
        #pragma omp parallel for
        for (auto row = blockSize; row < frame.GetSize().height - blockSize; row++) 
        {
            for (auto column = blockSize; column < frame.GetSize().width - blockSize; column++) 
            {
                auto x = column; auto y = row;

                auto choice = NVLib::RandomUtils::GetInteger(0, 10);
                auto disparity = NVLib::RandomUtils::GetInteger(0, maxDisparity * 100) / 100.0f;

                if (choice < 5) 
                {
                    auto offsetX = NVLib::RandomUtils::GetInteger(-2, 2);
                    auto offsetY = NVLib::RandomUtils::GetInteger(-2, 2);
                    auto nX = x + offsetX; auto nY = y + offsetY;
                    if (nX < 0 || nX >= frame.GetSize().width || nY < 0 || nY >= frame.GetSize().height) continue;
                    disparity = solution.GetDisparity(nY, nX);
                }

                auto score = GetScore(frame, gradientFrame, blockSize, Point(x, y), disparity);
                auto accepted = solution.UpdateScore(Point(x, y), score, disparity);

                //if (accepted) cout << "Value Accepted!" << endl;
            }
        }   

        //NVLib::DisplayUtils::ShowFloatMap("Disparity", solution.GetDisparity(), 1000);
        logger.Log(1, "Writing DepthMap: %i", i);
        auto outPath = NVLib::FileUtils::PathCombine(outputFolder, "disparity.jpg");
        Mat depthImage = Float2Color(solution.GetDisparity());
        imwrite(outPath, depthImage);  
    }
}

//--------------------------------------------------
// Get Gradient Frame
//--------------------------------------------------

/**
 * @brief Add the logic to get the given gradient frame
 * @param frame The frame that we are getting the value for
 * @return NVLib::StereoFrame The resultant gradient frame
 */
NVLib::StereoFrame GetGradientFrame(NVLib::StereoFrame& frame) 
{
    Mat frame1 = NVLib::ImageUtils::GetGradientMap(frame.GetLeft());
    Mat frame2 = NVLib::ImageUtils::GetGradientMap(frame.GetRight());

    auto p1 = vector<Mat>(); split(frame1, p1); auto p2 = vector<Mat>(); split(frame2, p2);

    return NVLib::StereoFrame(p1[0], p2[0]);
}

//--------------------------------------------------
// Score Calculation Logic
//--------------------------------------------------

/**
 * @brief Retrieve the score for the associate variable
 * @param frame The frame that we are getting score info from
 * @param gradientFrame The precalculated gradient maps
 * @param blockSize The size of the block that we are using
 * @param point The point that we are getting the score for
 * @param disparity The disparity that value that we are testing against
 * @return float The resultant score
 */
float GetScore(NVLib::StereoFrame& frame, NVLib::StereoFrame& gradientFrame, int blockSize, const Point& point, float disparity) 
{
    // Determine a block
    auto halfBlock = blockSize / 2;
    auto rect = Rect(point.x - halfBlock, point.y - halfBlock, blockSize, blockSize);

    // Get a gradient difference
    Mat gradient1 = gradientFrame.GetLeft()(rect);
    Mat gradient2 = GetBlock(gradientFrame.GetRight(), Point2f(point.x - disparity, point.y), blockSize);
    Mat gDiff; absdiff(gradient1, gradient2, gDiff);

    // Calculate a score
    auto score = sum(gDiff);

    // Return the result
    return score[0];
}

/**
 * @brief Extract a block from an image
 * @param image The image that we are getting the block for
 * @param point The point location
 * @param blockSize The size of the block
 * @return Mat The resultant matrix
 */
Mat GetBlock(Mat& image, const Point2f& point, int blockSize) 
{
    Mat mapx = Mat_<float>::zeros(blockSize, blockSize); auto dataX = (float *) mapx.data;
    Mat mapy = Mat_<float>::zeros(blockSize, blockSize); auto dataY = (float *) mapy.data;

    auto uStart = point.x - (blockSize * 0.5f); auto vStart = point.y - (blockSize * 0.5f);

    for (auto row = 0; row < blockSize; row++) 
    {
        for (auto column = 0; column < blockSize; column++) 
        {
            auto index = column + row * blockSize;
            dataX[index] = uStart + column;
            dataY[index] = vStart + row;
        }
    }

    Mat result; remap(image, result, mapx, mapy, INTER_LINEAR);
    return result;
}

//--------------------------------------------------
// Loader Helpers
//--------------------------------------------------

/**
 * @brief Load the associated stereo frame
 * @param folder The folder that we are loading from 
 * @param index The index of the frame that we are loading
 * @return NVLib::StereoFrame The resultant frame
 */
NVLib::StereoFrame ReadFrame(const string& folder, int index) 
{
    // Create the file names
    auto leftFile = stringstream(); leftFile << "left_" << setw(4) << setfill('0') << index << ".png";
    auto rightFile = stringstream(); rightFile << "right_" << setw(4) << setfill('0') << index << ".png";

    // Create the paths
    auto leftPath = NVLib::FileUtils::PathCombine(folder, leftFile.str());
    auto rightPath = NVLib::FileUtils::PathCombine(folder, rightFile.str());

    // Load the files
    Mat left = imread(leftPath, IMREAD_GRAYSCALE); if (left.empty()) throw runtime_error("Unable to open image: " + leftPath);
    Mat right = imread(rightPath, IMREAD_GRAYSCALE); if (right.empty()) throw runtime_error("Unable to open image: " + rightPath);
    medianBlur(left, left, 3); medianBlur(right, right, 3);

    // Create the result
    return NVLib::StereoFrame(left, right);
}

//--------------------------------------------------
// Utilities
//--------------------------------------------------

/**
 * @brief Converts a float image into a color image
 * @param image The image that we are converting
 * @return Mat The resultant image
 */
Mat Float2Color(Mat& image) 
{
	double min=0, max=1000; minMaxIdx(image, &min, &max);
	Mat d(image.size(), CV_8UC1); image.convertTo(d, CV_8UC1, 255.0 / (max - min), -min);
	Mat colorDepth; applyColorMap(d, colorDepth, COLORMAP_JET); 
    return colorDepth;
}

//--------------------------------------------------
// Entry Point
//--------------------------------------------------

/**
 * Main Method
 * @param argc The count of the incoming arguments
 * @param argv The number of incoming arguments
 * @return SUCCESS and FAILURE
 */
int main(int argc, char ** argv) 
{
    NVLib::Parameters * parameters = nullptr;

    try
    {
        parameters = NVL_Utils::ArgReader::GetParameters(argc, argv);
        Run(parameters);
    }
    catch (runtime_error exception)
    {
        cerr << "Error: " << exception.what() << endl;
        exit(EXIT_FAILURE);
    }
    catch (string exception)
    {
        cerr << "Error: " << exception << endl;
        exit(EXIT_FAILURE);
    }

    if (parameters != nullptr) delete parameters;

    return EXIT_SUCCESS;
}
