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

#include "AURTracker.h"
#include <iostream>
#include <vector>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include "RunTime.h"
#include "util.h"

// Number of keypoints needed to deactivate ArUco
const int32_t KeypointInitializationCount = 24; // kpt_init

// Distance between initialized corners
const float KeypointDistanceThreshold = 64;
const float KeypointDistanceThresholdSq = KeypointDistanceThreshold*KeypointDistanceThreshold;

// A candidate point is triangulated if camera moved at least that many cm since it was first detected
const float CameraMovementRequired = 30;
const float CameraMovementRequiredSq = CameraMovementRequired*CameraMovementRequired;

const int RansacIterations = 750;
const double RansacTolerance = 8;
const double RansacConfidence = 0.99;

const int32_t MAX_REPUTATION = 50;

const cv::Scalar ColorGreen(100, 255, 0);
const cv::Scalar ColorBad(0, 100, 255);
const cv::Scalar ColorYellow(0, 255, 255);
const cv::Scalar ColorBlue(255, 100, 100);
const cv::Scalar ColorCandidate(0, 255, 255);
const cv::Scalar ColorLandmarkInlier(100, 255, 0);
const cv::Scalar ColorLandmarkOutlier(50, 50, 255);
const cv::Scalar ColorReprojection(0, 100, 255);


AURTracker::KeyPointGeneration::KeyPointGeneration()
{
}

AURTracker::AURTracker()
	: Transform(3, 4)
	, FrameCount(0)
	, LandmarkCount(0)
	, UseAruco(true)
	, Cube(20.0)
{
	InitArUco();
}

void AURTracker::InitArUco()
{
	MarkerDictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_100);

	ArucoParameters = cv::aruco::DetectorParameters::create();	
	//ArucoParameters->doCornerRefinement = true;
	
	// Initialize ArUco marker pattern
	const std::vector<int32_t> marker_ids = {10, 11, 12, 13};
	const std::vector<cv::Point3f> marker_corners{
		cv::Point3f(8.5, -8.5, 0.0),
		cv::Point3f(8.5, -1.5, 0.0),
		cv::Point3f(1.5, -1.5, 0.0),
		cv::Point3f(1.5, -8.5, 0.0),
	};

	std::vector< std::vector<cv::Point3f> > object_points;
	
	for(float dx : {0.0, -10.0}) for(float dy : {0, 10})
	{
		std::vector<cv::Point3f> result(4);
		std::transform(marker_corners.begin(), marker_corners.end(), result.begin(), [&](cv::Point3f p) {
			return cv::Point3f(p.x+dx, p.y+dy, p.z);
		});

		object_points.push_back(result);
	}

	MarkerBoard = cv::aruco::Board::create(object_points, MarkerDictionary, marker_ids);
}

void AURTracker::SetCameraIntrinsic(cv::Mat_<double>& cam_intr)
{
	CameraIntrinsic = cam_intr;
	CameraIntrinsicInv = cam_intr.inv();
}

void AURTracker::SetCameraDistortion(cv::Mat_<double>& cam_dist)
{
	CameraDistortion = cam_dist;
}

void AURTracker::ProcessFrame(cv::Mat_<cv::Vec3b>& input_image)
{
	FrameCount += 1;
	TransformInThisFrame = false;

	cv::swap(FrameGray, FrameGrayPrev);

	FrameGray.create(input_image.size());
	cv::cvtColor(input_image, FrameGray, cv::COLOR_BGR2GRAY);

	FrameRect.x = 0;
	FrameRect.y = 0;
	FrameRect.width = FrameGray.cols;
	FrameRect.height = FrameGray.rows;

	input_image.copyTo(FrameOut);

	cv::putText(FrameOut, std::to_string(FrameCount), cv::Point2i(10, 30),
		cv::FONT_HERSHEY_COMPLEX, 1.0, cv::Scalar(0, 0, 255), 2);

	if(UseAruco)
	{
		ProcessFrameArUco();		
	}
	
	if(FrameCount > 5)
	{
		ProcessFrameKeypoints();
	}
	
	// Draw final transform
	if(TransformInThisFrame)
	{
		Cube.Draw(FrameOut, Projection);
	}
	
	input_image = FrameOut;
}

void AURTracker::ProcessFrameArUco()
{
	std::vector< std::vector< cv::Point2f > > aruco_found_corners;
	std::vector< int > aruco_found_ids;
	
	// Finds marker corners in image
	cv::aruco::detectMarkers(FrameGray, MarkerDictionary, aruco_found_corners, aruco_found_ids, ArucoParameters);

	// Recover pose from detected markers
	cv::Vec3d out_rot_angle_axis;
	cv::Mat_<double> out_translation;
	cv::Mat_<double> transform_aruco(3, 4);

	int found = cv::aruco::estimatePoseBoard(
		aruco_found_corners, aruco_found_ids, MarkerBoard,
		CameraIntrinsic, CameraDistortion,
		out_rot_angle_axis, out_translation
	);

	if(found > 0)
	{
		const cv::Scalar col_aruco(255, 20, 20);
		
		cv::Rodrigues(out_rot_angle_axis, transform_aruco.colRange(0, 3));
		out_translation.copyTo(transform_aruco.col(3));

		// Draw ArUco transform
		Cube.Draw(FrameOut, CameraIntrinsic * transform_aruco, col_aruco);
		cv::aruco::drawDetectedMarkers(FrameOut, aruco_found_corners, aruco_found_ids, cv::Scalar(255, 100, 100));

		Transform = transform_aruco;
		Projection = CameraIntrinsic * Transform;
		TransformInThisFrame = true;
		
		cv::putText(FrameOut, "ArUco", cv::Point2i(100, 30), cv::FONT_HERSHEY_COMPLEX, 0.75, col_aruco, 2);
	}
}


void AURTracker::AddCandidate(const cv::Point2f ptf, std::shared_ptr<KeyPointGeneration> generation)
{
	KeyPoint kpt;
	kpt.Status = KeyPoint::Candidate;
	kpt.FirstFramePosition = ptf;
	kpt.FramePosition = ptf;
	kpt.Generation = generation;
	
	KeyPoints.push_back(kpt);
}

void AURTracker::RemoveKeyPoint(const int32_t index)
{
	if(KeyPoints[index].Status == KeyPoint::Landmark)
	{
		LandmarkCount -= 1;
	}
	
	std::swap(KeyPoints[index], KeyPoints[KeyPoints.size()-1]);
	KeyPoints.pop_back();
}


void AURTracker::TriangulatePoints(std::vector<size_t> const& kpts_to_triang)
{
	cv::Mat_< cv::Point2f > pts1(1, 1);
	cv::Mat_< cv::Point2f > pts2(1, 1);

	cv::Mat_< float > triang_landmarks;

	for(auto ckpt : kpts_to_triang)
	{
		KeyPoint &kpt = KeyPoints[ckpt];

		pts1(0, 0) = kpt.FirstFramePosition;
		pts2(0, 0) = kpt.FramePosition;
		//cv::triangulatePoints(kpt.FirstTransform, Transform, pts1, pts2, triang_landmarks);
		cv::triangulatePoints(Projection, kpt.Generation->FirstProjection, pts2, pts1, triang_landmarks);

		kpt.WorldLocation.x = triang_landmarks(0, 0);
		kpt.WorldLocation.y = triang_landmarks(1, 0);
		kpt.WorldLocation.z = triang_landmarks(2, 0);
		kpt.WorldLocation /= triang_landmarks(3, 0);

		std::cout << "Trianguated: " << kpt.WorldLocation << std::endl;

		kpt.Status = KeyPoint::Landmark;
		LandmarkCount += 1;
	}

}

void AURTracker::StepTracking()
{
	// LK
	if(KeyPoints.size() > 0)
	{
		cv::Mat_<cv::Point2f> points_lk_prev(KeyPoints.size(), 1);
		for(size_t ckpt = 0; ckpt < KeyPoints.size(); ckpt++)
		{
			points_lk_prev(ckpt, 0) = KeyPoints[ckpt].FramePosition;
		}
		cv::Mat points_lk_new;
		cv::Mat status_lk;
		cv::Mat errors_lk;
		
		cv::calcOpticalFlowPyrLK(FrameGrayPrev, FrameGray, points_lk_prev, points_lk_new, status_lk, errors_lk);

		// iterate from end to front so that removal (swap with last, reduce size) is safe
		for(int32_t ckpt = KeyPoints.size() - 1; ckpt >= 0; ckpt--)
		{
			if(!status_lk.at<uint8_t>(ckpt, 0))
			{
				RemoveKeyPoint(ckpt);
			}
			else
			{
				KeyPoint &kpt = KeyPoints[ckpt];

				kpt.FramePosition = points_lk_new.at<cv::Point2f>(ckpt, 0);
			}
		}
	}
}

void AURTracker::StepPoseEstimation()
{
	std::vector< cv::Point2f > img_points;
	std::vector< cv::Point3f > world_points;
	std::vector< size_t > kpt_indices;
	
	for(int32_t ckpt = 0; ckpt < KeyPoints.size(); ckpt++)
	{
		KeyPoint &kpt = KeyPoints[ckpt];
		if(kpt.Status == KeyPoint::Landmark)
		{
			img_points.push_back(kpt.FramePosition);
			world_points.push_back(kpt.WorldLocation);
			kpt_indices.push_back(ckpt);
		}
		
		// we will set it to true for inliers
		kpt.IsInlier = false;
	}

	if(img_points.size() > 8)
	{
		cv::Mat_<double> pnp_transform(3, 4);

		cv::Vec3d out_rot_angle_axis;
		cv::Vec3d out_trans;
		std::vector<int32_t> out_inliers;
		
		bool pnp_result = cv::solvePnPRansac(
			world_points, img_points,
			CameraIntrinsic, CameraDistortion,
			out_rot_angle_axis, out_trans,
			false, 
			RansacIterations,
			RansacTolerance,
			RansacConfidence, 
			out_inliers,
			cv::SOLVEPNP_P3P
		);

		if(pnp_result)
		{
			const int32_t inlier_count = out_inliers.size();
			
			for(size_t inlier_idx : out_inliers)
			{
				KeyPoint &kpt = KeyPoints[kpt_indices[inlier_idx]];
				kpt.IsInlier = true;
			}
			
			// iterate from end to front so that removal (swap with last, reduce size) is safe
			for(int32_t ckpt = KeyPoints.size() - 1; ckpt >= 0; ckpt--)
			{
				KeyPoint &kpt = KeyPoints[ckpt];
				if(kpt.Status == KeyPoint::Landmark)
				{
					if(kpt.IsInlier)
					{
						kpt.Reputation = std::min(kpt.Reputation+1, MAX_REPUTATION);
					}
					else
					{
						kpt.Reputation -= 1;
						if(kpt.Reputation <= 0)
						{
							RemoveKeyPoint(ckpt);
						}
					}
				}
			}
			
			cv::Scalar col_pnp(50, 255, 200);
			
			cv::Rodrigues(out_rot_angle_axis, pnp_transform.colRange(0, 3));
			pnp_transform(0, 3) = out_trans(0);
			pnp_transform(1, 3) = out_trans(1);
			pnp_transform(2, 3) = out_trans(2);
			
			Transform = pnp_transform;
			Projection = CameraIntrinsic * Transform;
			TransformInThisFrame = true;

			Cube.Draw(FrameOut, CameraIntrinsic*pnp_transform, col_pnp);
			cv::putText(FrameOut, 
				"RANSAC+P3P " + std::to_string(inlier_count) + " / " + std::to_string(img_points.size()),
				cv::Point2i(250, 30), cv::FONT_HERSHEY_COMPLEX, 0.75, col_pnp, 2);
		}
		else
		{
			cv::putText(FrameOut, "PNP FAIL", cv::Point2i(250, 30), cv::FONT_HERSHEY_COMPLEX, 0.75, cv::Scalar(0, 0, 255), 2);
			std::cout << "PNP fail" << std::endl;
		}
	}
}

void AURTracker::StepTriangulationAttempt()
{
	// Attempt triangulation if we know the camera pose of current frame
	if(TransformInThisFrame)
	{
		std::vector< size_t > triang_ids; // keypoints to triangulate
		for(size_t ckpt = 0; ckpt < KeyPoints.size(); ckpt++)
		{
			KeyPoint &kpt = KeyPoints[ckpt];

			// Triangulate point if camera has moved far enough from the pose
			// where it first saw this keypoint
			if(kpt.Status == KeyPoint::Candidate && kpt.GetCameraDistanceSq(Transform) > CameraMovementRequiredSq)
			{
				triang_ids.push_back(ckpt);
			}
		}
		
		TriangulatePoints(triang_ids);
		
		if(LandmarkCount >= KeypointInitializationCount)
		{
			// if we have enough landmarks, stop ArUco
			UseAruco = false;
		}
	}
	
	// Draw points
	for(auto const& kpt : KeyPoints)
	{
		kpt.DrawDiagnostic(FrameOut, *this);
	}
}

void AURTracker::StepKeypointGeneration()
{
	// Create new points
	if(ShouldCreateNewPoints())
	{
		cv::Mat_< cv::Point_<uint16_t> > feature_corners;
		cv::goodFeaturesToTrack(
			FrameGray, feature_corners, 
			96,	// number of corners
			0.05, // quality level
			KeypointDistanceThreshold, // min distance between corners
			cv::noArray(), // mask
			3, // block size for non-max suppression
			true // use harris corner detector
		);

		// Store current camera pose - shared by all points created in this iteration
		std::shared_ptr<KeyPointGeneration> new_gen(new KeyPointGeneration);
		Transform.copyTo(new_gen->FirstCameraRT);
		Projection.copyTo(new_gen->FirstProjection);
		
		// Discard points which are close to existing tracked points
		for(int cid = 0; cid < feature_corners.rows; cid++)
		{
			//const cv::Point2i pt(feature_corners[cid][0]);
			const cv::Point2f ptf(feature_corners[cid][0]);

			if(IsGoodNewPoint(ptf))
			{
				AddCandidate(ptf, new_gen);
			}
		}
	}
}

bool AURTracker::ShouldCreateNewPoints()
{
	// Initial camera pose will be needed for triangulation later, 
	// so we can't create points without knowing the pose
	if(!TransformInThisFrame) 
	{
		return false;
	}
		
	// check: too few KeyPoints
	if(KeyPoints.size() < 64)
	{
		return true;
	}
	
	// check: offcenter
	// find center of mass of keypoints in image
	// if its far away from image center, create new points
	cv::Mat_<double> center(2, 1);
	center.setTo(0);
	for(auto const& kp : KeyPoints)
	{
		center(0) += kp.FramePosition.x;
		center(1) += kp.FramePosition.y;
	}
	center /= (double)KeyPoints.size();
	
	draw_cross(FrameOut, std::round(center(0)), std::round(center(1)), 2, 10, ColorGreen);
	
	center(0) = (center(0) / FrameRect.width) - 0.5;
	center(1) = (center(1) / FrameRect.height) - 0.5;
	
	if(cv::norm(center, cv::NORM_L2SQR) >= 0.2*0.2)
	{
		std::cout << "offcenter" << std::endl;
		return true;
	}

	return false;
}

bool AURTracker::IsGoodNewPoint(const cv::Point2f ptf)
{
	for(auto const& kp : KeyPoints)
	{
		if(point_distance_sq(kp.FramePosition, ptf) < KeypointDistanceThresholdSq)
		{
			return false;
		}
	}

	return true;
}

void AURTracker::ProcessFrameKeypoints()
{
	StepTracking();
	
	StepPoseEstimation();
	
	StepTriangulationAttempt();
	
	StepKeypointGeneration();
}

AURTracker::KeyPoint::KeyPoint()
	: Reputation(MAX_REPUTATION)
{
}

float AURTracker::KeyPoint::GetDisparity() const
{
	return point_distance_sq(FramePosition, FirstFramePosition);
}

float AURTracker::KeyPoint::GetCameraDistanceSq(const cv::Mat_<double>& cam_transform) const
{
	return Generation->GetCameraDistanceSq(cam_transform);
}

float AURTracker::KeyPointGeneration::GetCameraDistanceSq(const cv::Mat_<double>& cam_transform) const
{
	return cv::norm(FirstCameraRT.col(3) - cam_transform.col(3), cv::NORM_L2SQR);	
}

void AURTracker::KeyPoint::DrawDiagnostic(cv::Mat_<cv::Vec3b>& out_frame, const AURTracker& tracker) const
{
	cv::Mat_<double> world_pos(4, 1);
	cv::Mat_<double> img_pos(3, 1);
	world_pos(3, 0) = 1.0;

	const bool is_landmark = Status == KeyPoint::Landmark;
	const cv::Point2i pos_current(FramePosition.x, FramePosition.y);
	
	cv::Scalar front_color = ColorCandidate;
	if(is_landmark)
	{
		if(IsInlier)
		{
			front_color = ColorLandmarkInlier;
		}
		else
		{
			front_color = ColorLandmarkOutlier;
		}
	}
	
	if(is_landmark && tracker.TransformInThisFrame)
	{
		// reproject
		world_pos(0, 0) = WorldLocation.x;
		world_pos(1, 0) = WorldLocation.y;
		world_pos(2, 0) = WorldLocation.z;

		img_pos = tracker.Projection * world_pos;
		img_pos /= img_pos(2, 0);
		
		// reperr = Proj * W - pos_img
		// 
		// d W / d reperr = 

		const cv::Point2i pos_repro(img_pos(0, 0), img_pos(1, 0));

		cv::line(out_frame, pos_repro, pos_current, front_color, 2);
	
		cv::circle(out_frame, pos_repro, 3, ColorReprojection, 2);
	}
	
	
	cv::circle(
		out_frame,
		pos_current,
		3,
		front_color,
		2
	);
}
