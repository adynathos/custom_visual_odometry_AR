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

#include "VideoSource.h"
#include <iostream>

const char* KEY_RESOLUTION = "Resolution";
const char* KEY_CAMERA_MATRIX = "CameraMatrix";
const char* KEY_DISTORTION = "DistortionCoefficients";

VideoSource::VideoSource(cv::Size2i const resolution)
	: IntrinsicMatrix(3, 3)
	, DistortionCoefficients(5, 1)
	, Resolution(resolution)
{
	IntrinsicMatrix.setTo(0.0);
	const double f = 900.0;
	IntrinsicMatrix(0, 0) = f;
	IntrinsicMatrix[1][1] = f;
	IntrinsicMatrix[0][2] = 0.5 * double(Resolution.width);
	IntrinsicMatrix[1][2] = 0.5 * double(Resolution.height);

	DistortionCoefficients.setTo(0);
}

bool VideoSource::LoadCalibration(std::string input_file_path)
{
	try
	{
		cv::FileStorage cam_param_file(input_file_path, cv::FileStorage::READ);

		if (!cam_param_file.isOpened())
		{
			std::cout << "	Failed to open file " << input_file_path << std::endl;
			return false;
		}

		cam_param_file[KEY_CAMERA_MATRIX] >> IntrinsicMatrix;
		cam_param_file[KEY_DISTORTION] >> DistortionCoefficients;

		std::cout << "	Calibration loaded, f = " << IntrinsicMatrix(0, 0) << std::endl;
	}
	catch (std::exception& exc)
	{
		std::cout << "	Exception while reading file " << input_file_path << ": " << exc.what() << std::endl;
		return false;
	}

	return true;
}

bool VideoSource::OpenVideoFile(std::string input_file_path)
{
	Camera.open(input_file_path);

	if(Camera.isOpened())
	{
		Resolution.width = cv::saturate_cast<int32_t>(Camera.get(cv::CAP_PROP_FRAME_WIDTH));
		Resolution.height = cv::saturate_cast<int32_t>(Camera.get(cv::CAP_PROP_FRAME_HEIGHT));

		std::cout << "	Opened video file " << input_file_path << ", resolution = " << Resolution << std::endl;

		return true;
	}
	else
	{
		std::cout << "	Failed to open video file " << input_file_path << std::endl;
		return false;
	}
}

bool VideoSource::GetFrame(cv::Mat_<cv::Vec3b>& out_frame)
{
	return Camera.read(out_frame);
}
