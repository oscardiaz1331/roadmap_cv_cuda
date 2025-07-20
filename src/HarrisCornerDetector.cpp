 #include "HarrisCornerDetector.hpp"

HarrisCornerDetector::HarrisCornerDetector(int windowSize, double k, double threshold, bool debugMode)
	: windowSize_(windowSize), k_(k), threshold_(threshold), debugMode_(debugMode)
{
	if (windowSize % 2 == 0 || windowSize < 3)
	{
		throw std::invalid_argument("Window size must be an odd number greater than or equal to 3.");
	}
	if (k <= 0 || k >= 1)
	{
		throw std::invalid_argument("k must be in the range (0, 1).");
	}
	if (threshold <= 0)
	{
		throw std::invalid_argument("Threshold must be greater than 0.");
	}
}
void HarrisCornerDetector::detectCorners(const cv::Mat& image, std::vector<cv::Point>& corners) const
{
	cv::Mat grayImage;
	if (image.channels() == 3)
		cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
	else
		grayImage = image;

	displayNormalized(grayImage, "1. Grayscale");

	cv::Mat gradX, gradY, Ix2, Iy2, Ixy, A, B, C;
	cv::Sobel(grayImage, gradX, CV_64F, 1, 0, windowSize_);
	cv::Sobel(grayImage, gradY, CV_64F, 0, 1, windowSize_);

	displayNormalized(gradX, "2a. Gradient X");
	displayNormalized(gradY, "2b. Gradient Y");

	Ix2 = gradX.mul(gradX);
	Iy2 = gradY.mul(gradY);
	Ixy = gradX.mul(gradY);

	cv::GaussianBlur(Ix2, A, cv::Size(windowSize_, windowSize_), 0);
	cv::GaussianBlur(Iy2, B, cv::Size(windowSize_, windowSize_), 0);
	cv::GaussianBlur(Ixy, C, cv::Size(windowSize_, windowSize_), 0);

	displayNormalized(A, "3a. Sum(Ix^2)");
	displayNormalized(B, "3b. Sum(Iy^2)");
	displayNormalized(C, "3c. Sum(Ixy)");

	cv::Mat detM = A.mul(B) - C.mul(C);
	cv::Mat traceM = A + B;
	cv::Mat R = detM - k_ * traceM.mul(traceM);

	cv::normalize(R, R, 0, 1, cv::NORM_MINMAX);
	displayNormalized(R, "4. Harris Response (R)");

	corners.clear();
	int radius = windowSize_ / 2;

	for (int y = radius; y < R.rows - radius; ++y) {
		for (int x = radius; x < R.cols - radius; ++x) {
			double val = R.at<double>(y, x);
			if (val < threshold_) [[likely]] continue;

			bool isMax = true;
			for (int dy = -radius; dy <= radius && isMax; ++dy) {
				for (int dx = -radius; dx <= radius; ++dx) {
					if (dx == 0 && dy == 0) continue;
					if (R.at<double>(y + dy, x + dx) >= val) {
						isMax = false;
						break;
					}
				}
			}

			if (isMax)
				corners.emplace_back(x, y);
		}
	}
}

// Helper function to normalize and display a floating-point image
void HarrisCornerDetector::displayNormalized(const cv::Mat& img, const std::string& winName) const
{
	if (!debugMode_) return;
	cv::Mat displayImg;
	double minVal, maxVal;
	cv::minMaxLoc(img, &minVal, &maxVal);
	// Normalize to 0-255 range and convert to 8-bit unsigned
	img.convertTo(displayImg, CV_8U, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));
	cv::imshow(winName, displayImg);
	cv::waitKey(0); // Wait indefinitely until a key is pressed
}

int main() {
	constexpr int kWindowSize = 21;
	constexpr double kHarrisK = 0.06;
	constexpr double kThreshold = 0.01; // Normalized threshold
	constexpr bool kDebugMode= false;
	const HarrisCornerDetector cornerDetector(kWindowSize, kHarrisK, kThreshold, kDebugMode);

	cv::Mat image = cv::imread("data/chessboard.png", cv::IMREAD_COLOR);
	if (image.empty()) {
		std::cerr << "Could not open image.\n";
		return EXIT_FAILURE;
	}

	std::vector<cv::Point> corners;

	const auto start = std::chrono::high_resolution_clock::now();
	cornerDetector.detectCorners(image, corners);
	const auto end = std::chrono::high_resolution_clock::now();
	const auto duration_ms = std::chrono::duration<double, std::milli>(end - start).count();

	cv::Mat result = image.clone();
	std::ranges::for_each(corners, [&](const cv::Point& pt) {
		cv::circle(result, pt, 3, cv::Scalar(0, 0, 255), cv::FILLED);
		});

	cv::imshow("Detected Corners", result);
	std::cout << "Corners found: " << corners.size()
		<< " | Time elapsed: " << duration_ms << " ms\n";

	cv::waitKey(0);
	return EXIT_SUCCESS;
}