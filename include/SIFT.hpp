import std;
#include <opencv2/opencv.hpp>

class SIFT {
private:
	int nFeatures;
	int nOctaveLayers;
	double contrastThreshold;
	double edgeThreshold;
	double sigma;
public:
	SIFT(int nFeatures = 0, int nOctaveLayers = 3, double contrastThreshold = 0.04, double edgeThreshold = 10, double sigma = 1.6);
	~SIFT() = default;
	void detectAndCompute(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors);
};