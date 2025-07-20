#include <opencv2/opencv.hpp>

class HarrisCornerDetector {
private:
	const int windowSize_;
	const double k_, threshold_;
	const bool debugMode_;
	void displayNormalized(const cv::Mat& img, const std::string& winName) const;

public:
	HarrisCornerDetector(int windowSize, double k, double threshold, bool debugMode = false);
	~HarrisCornerDetector() = default;
	void detectCorners(const cv::Mat& image, std::vector<cv::Point>& corners) const;
};