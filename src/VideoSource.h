/*
Copyright 2016-2017 Krzysztof Lis

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#pragma once

#include <opencv2/videoio.hpp>

class VideoSource
{
public:
	cv::Mat_<double> IntrinsicMatrix;
	cv::Mat_<double> DistortionCoefficients;
	cv::Size2i Resolution;
	cv::VideoCapture Camera;

	VideoSource(cv::Size2i const resolution = cv::Size2i(1920, 1080));

	bool LoadCalibration(std::string input_file_path);

	bool OpenVideoFile(std::string input_file_path);

	bool GetFrame(cv::Mat_<cv::Vec3b> & out_frame);
};
