#include "SIFT.hpp"

SIFT::SIFT(int nFeatures, int nOctaveLayers, double contrastThreshold, double edgeThreshold, double sigma)
	: nFeatures(nFeatures), nOctaveLayers(nOctaveLayers), contrastThreshold(contrastThreshold),
	  edgeThreshold(edgeThreshold), sigma(sigma)
{
	if (nFeatures < 0)
	{
		throw std::invalid_argument("Number of features must be non-negative.");
	}
	if (nOctaveLayers < 1)
	{
		throw std::invalid_argument("Number of octave layers must be at least 1.");
	}
	if (contrastThreshold <= 0)
	{
		throw std::invalid_argument("Contrast threshold must be greater than 0.");
	}
	if (edgeThreshold <= 0)
	{
		throw std::invalid_argument("Edge threshold must be greater than 0.");
	}
	if (sigma <= 0)
	{
		throw std::invalid_argument("Sigma must be greater than 0.");
	}
}
{
}

void SIFT::detectAndCompute(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors)
{
	cv::Mat grayImage;
	if (image.channels() == 3)
	{
		cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
	}
	else
	{
		grayImage = image.clone();
	}
	std::vector<cv::Mat> gaussianPyramid;
	pyramid.push_back(grayImage.clone());  // Octave 0: original
	for (int i = 1; i < numOctaves; ++i) {
		cv::Mat down;
		cv::pyrDown(pyramid[i - 1], down);  // reduce by factor of 2
		pyramid.push_back(down);
	}
	std::vector<cv::Mat> dogPyramids;
	for (size_t octave = 1; octave < gaussianPyramid.size(); ++octave) {
		std::vector<cv::Mat> dogPyramid;
		cv::Mat dog;
		
		dogPyramid.push_back(dog);
	}
}
