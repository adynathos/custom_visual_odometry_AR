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

#include <opencv2/core.hpp>

class CubeMesh
{
public:
	CubeMesh(double side = 20.0);
	
	/*
	 * projection_matrix = Intrinsic * [R | T]
	 */
	void Draw(cv::Mat_<cv::Vec3b> & input_image, cv::Mat_<double> const& projection_matrix, const cv::Scalar color = cv::Scalar(255, 50, 200)) const;
	
private:
	cv::Mat_<double> CubePts;
	cv::Mat_<int32_t> CubeEdges;
};

