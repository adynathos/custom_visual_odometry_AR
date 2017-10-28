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

#include "CubeMesh.h"
#include <opencv2/imgproc.hpp>

CubeMesh::CubeMesh(double side)
{
	cv::Mat_<double> face_bot(4, 4); // homog
	face_bot <<
		0.5, 0.5, 0, 1,
		0.5, -0.5, 0, 1,
		-0.5, -0.5, 0, 1,
		-0.5, 0.5, 0, 1;
	CubePts.create(4, 8);
	CubePts.colRange(0, 4) = face_bot.t();
	CubePts.colRange(4, 8) = face_bot.t();
	CubePts.row(2).colRange(4, 8) -= 1.0;

	// To scale, multiply everyting apart from the homogeonous balance
	CubePts.rowRange(0, 3) *= side;

	CubeEdges.create(12, 2);
	CubeEdges <<
		0, 1,
		1, 2,
		2, 3,
		3, 0,
		4, 5,
		5, 6,
		6, 7,
		7, 4,
		0, 4,
		1, 5,
		2, 6,
		3, 7;	
}

void CubeMesh::Draw(cv::Mat_<cv::Vec3b>& input_image, const cv::Mat_<double>& projection_matrix, const cv::Scalar color) const
{
	cv::Mat_<double> cube_pts_cam = projection_matrix * CubePts;
	cube_pts_cam.row(0) /= cube_pts_cam.row(2);
	cube_pts_cam.row(1) /= cube_pts_cam.row(2);

	for(int32_t edge = 0; edge < 12; edge++)
	{
		const int32_t start_id = CubeEdges[edge][0];
		const int32_t end_id = CubeEdges[edge][1];

		cv::line(input_image,
			cv::Point2i(cube_pts_cam[0][start_id], cube_pts_cam[1][start_id]),
			cv::Point2i(cube_pts_cam[0][end_id], cube_pts_cam[1][end_id]),
			color,
			4
		);
	}
}
