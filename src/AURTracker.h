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

#include <opencv2/aruco.hpp>
#include <opencv2/video.hpp>
#include <memory>

#include "CubeMesh.h"

class AURTracker
{
public:
	struct KeyPointGeneration
	{
		cv::Mat_<double> FirstCameraRT;
		cv::Mat_<double> FirstProjection;
		
		KeyPointGeneration();
		
		float GetCameraDistanceSq(cv::Mat_<double> const& cam_transform) const;
	};
	
	struct KeyPoint
	{
		enum KeyPointStatus {Candidate, Landmark};

		cv::Point2f FramePosition;
		KeyPointStatus Status;

		// Candidate
		cv::Point2f FirstFramePosition;
		
		// Landmark
		cv::Point3f WorldLocation;

		// Recent tracking
		bool IsInlier;
		
		// Reduced if point is an outlier
		int32_t Reputation;
		
		// Initial camera pose when point was detected
		std::shared_ptr<KeyPointGeneration> Generation;
		
		KeyPoint();
		
		float GetDisparity() const;
		float GetCameraDistanceSq(cv::Mat_<double> const& cam_transform) const;
		void DrawDiagnostic(cv::Mat_<cv::Vec3b>& out_frame, AURTracker const& tracker) const;
	};
	
	cv::Mat_<double> Transform; // current [R|T] matrix
	cv::Mat_<double> Projection; // current CAM_INTR * [R|T] matrix
	
	AURTracker();
	void SetCameraIntrinsic(cv::Mat_<double>& cam_intr);
	void SetCameraDistortion(cv::Mat_<double>& cam_dist);
	
	void ProcessFrame(cv::Mat_<cv::Vec3b>& input_image);
	
protected:
	cv::Mat_<double> CameraIntrinsic;
	cv::Mat_<double> CameraIntrinsicInv;
	cv::Mat_<double> CameraDistortion;

	std::vector<KeyPoint> KeyPoints;
	
	int32_t FrameCount;
	bool TransformInThisFrame;

	int LandmarkCount;
	bool UseAruco;

	CubeMesh Cube;

	cv::Rect FrameRect; // frame size
	cv::Mat_<uint8_t> FrameGray;
	cv::Mat_<uint8_t> FrameGrayPrev;

	cv::Mat_<cv::Vec3b> FrameOut;

	// Aruco
	cv::Ptr<cv::aruco::Dictionary> MarkerDictionary;
	cv::Ptr<cv::aruco::Board> MarkerBoard;
	cv::Ptr<cv::aruco::DetectorParameters> ArucoParameters;

	void InitArUco();
	
	void ProcessFrameArUco();
	void ProcessFrameKeypoints();

	// Track keypoints from frame to frame using Lucas-Kanade algorithm
	void StepTracking(); 
	void StepPoseEstimation();
	void StepTriangulationAttempt();
	void StepKeypointGeneration();
	bool ShouldCreateNewPoints();
	bool IsGoodNewPoint(const cv::Point2f ptf);

	void AddCandidate(const cv::Point2f ptf, std::shared_ptr<KeyPointGeneration> generation);
	void RemoveKeyPoint(const int32_t index);

	void TriangulatePoints(std::vector<size_t> const& kpts_to_triang);
};


