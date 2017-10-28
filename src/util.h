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
#include <iostream>

std::string cvGetTypeName(const int type_id);
void printMatInfo(std::string const& name, cv::Mat const& m);
void cross_product_out(cv::Mat const& in_vector, cv::Mat_<double>& out_cross_matrix);
cv::MatExpr solve_Ax_eq_zero(cv::Mat const& A);
void test_solve();
cv::Mat_<double> linear_triangulation(cv::Mat const& proj_1, cv::Mat const& proj_2, cv::Mat const& pt_1, cv::Mat const& pt_2);
void test_triang();

void reverseRT(cv::Mat_<double> const& in_rt, cv::Mat_<double>& out_inverse_rt);

void homogenous_to_3d(cv::Mat const& homog, cv::Mat_<double>& out_pts3d);
float point_distance_sq(cv::Point2f const& a, cv::Point2f const& b);

void draw_cross(cv::Mat_<cv::Vec3b>& image, int x, int y, int thickness = 2, int length = 10, cv::Scalar const& color = cv::Scalar(0, 255, 0));
