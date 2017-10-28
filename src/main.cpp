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

#include <iostream>

#include <opencv2/aruco.hpp>
#include <opencv2/highgui.hpp>
#include <string>

#include "VideoSource.h"
#include "AURTracker.h"

int main(int arg_count, char** arg_values)
{
	// Determine config file
	std::string config_file_path = "default.xml";
	if(arg_count >= 2)
	{
		config_file_path = arg_values[1];
	}
	std::cout << "Config file: " << config_file_path << std::endl;
	
	
	// Read config file
	std::string input_video, output_video, calibration_file;
	
	try
	{
		cv::FileStorage param_file(config_file_path, cv::FileStorage::READ);

		if (!param_file.isOpened())
		{
			std::cout << "Failed to open file " << config_file_path << std::endl;
			return 1;
		}

		param_file["input_video"] >> input_video;
		param_file["output_video"] >> output_video;
		param_file["calibration_file"] >> calibration_file;
	}
	catch (std::exception& exc)
	{
		std::cout << "Exception while reading file " << config_file_path << ":\n" << exc.what() << std::endl;
		return 1;
	}	
	
	// Load video and calibration
	VideoSource vs;

	std::cout << "Loading calibration: " << calibration_file << std::endl;
	vs.LoadCalibration(calibration_file);
	std::cout << "Loading video: " << input_video << std::endl;
	vs.OpenVideoFile(input_video);
	
	// Start writing video if needed
	bool b_write_video = false;
	cv::VideoWriter video_writer;
	
	if(output_video.size() > 0)
	{
		std::cout << "Saving video to: " << output_video << std::endl;
		
		b_write_video = true;
		video_writer.open(output_video,
			CV_FOURCC('H', '2', '6', '4'),
			30.0, vs.Resolution,
			true
		);
	}
	
	// Init tracking algorithm
	AURTracker tracker;
	tracker.SetCameraIntrinsic(vs.IntrinsicMatrix);
	tracker.SetCameraDistortion(vs.DistortionCoefficients);
	
	// Process frames of the input video
	cv::Mat_<cv::Vec3b> video_frame;
	while(vs.GetFrame(video_frame))
	{
		// Run algorithm
		tracker.ProcessFrame(video_frame);

		// Display in window
		cv::imshow("AUR odometry tracker", video_frame);

		// Write to file
		if(b_write_video)
		{
			video_writer.write(video_frame);
		}
		
		// Wait, quit when key Q is pressed
		if(cv::waitKey(20) == 113)
		{
			break;
		} 
	}

	// Save written video
	if(b_write_video)
	{
		video_writer.release();
	}
}
