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

#include "util.h"
#include <opencv2/imgproc.hpp>

const std::vector<int>type_codes{
	CV_8U,  CV_8UC1,  CV_8UC2,  CV_8UC3,  CV_8UC4,
	CV_8S,  CV_8SC1,  CV_8SC2,  CV_8SC3,  CV_8SC4,
	CV_16U, CV_16UC1, CV_16UC2, CV_16UC3, CV_16UC4,
	CV_16S, CV_16SC1, CV_16SC2, CV_16SC3, CV_16SC4,
	CV_32S, CV_32SC1, CV_32SC2, CV_32SC3, CV_32SC4,
	CV_32F, CV_32FC1, CV_32FC2, CV_32FC3, CV_32FC4,
	CV_64F, CV_64FC1, CV_64FC2, CV_64FC3, CV_64FC4};

const std::vector<std::string> type_names{
	"CV_8U",  "CV_8UC1",  "CV_8UC2",  "CV_8UC3",  "CV_8UC4",
	"CV_8S",  "CV_8SC1",  "CV_8SC2",  "CV_8SC3",  "CV_8SC4",
	"CV_16U", "CV_16UC1", "CV_16UC2", "CV_16UC3", "CV_16UC4",
	"CV_16S", "CV_16SC1", "CV_16SC2", "CV_16SC3", "CV_16SC4",
	"CV_32S", "CV_32SC1", "CV_32SC2", "CV_32SC3", "CV_32SC4",
	"CV_32F", "CV_32FC1", "CV_32FC2", "CV_32FC3", "CV_32FC4",
	"CV_64F", "CV_64FC1", "CV_64FC2", "CV_64FC3", "CV_64FC4"};

std::string cvGetTypeName(const int type_id)
{
	for(int i=0; i < type_codes.size(); i++)
	{
		if(type_id == type_codes[i]) return type_names[i];
	}
	return "unknown cv type";
}

void printMatInfo(std::string const& name, cv::Mat const& m)
{
	std::cout << name << ": " << m.rows << " x " << m.cols << " " << cvGetTypeName(m.type()) << std::endl;
}

void cross_product_out(cv::Mat const& in_vector, cv::Mat_<double>& out_cross_matrix)
{
	cv::MatConstIterator_<double> it = in_vector.begin<double>();

	out_cross_matrix.create(3, 3);
	out_cross_matrix <<
	0,			-*(it+2),	*(it+1),
	*(it+2),	0,			-*(it+0),
	-*(it+1),	*(it+0),	0;

	printMatInfo("out_cross_matrix", out_cross_matrix);
}

cv::MatExpr solve_Ax_eq_zero(cv::Mat const& A)
{
	cv::SVD svd(A);
	/*
		*	def solve_AX_eq_zero(A):
		*	# solve A P = 0
		*	# solution is:
		*	# smallest eigvect of A.t A
		*	# A.t A = V S.t U.t U S V.t = V S^2 V.t
		*	# so take last col of V
		*
		*	# np's svd returns U, S, V.t instead of U, S, V
		*	U, S, Vt = np.linalg.svd(A)
		*	V = Vt.T
		*
		*	# try reconstruction
		*	#A_reconstr = U @ np.diag(S) @ Vt
		*	#report_mat_size('A - USVt: ', A - A_reconstr)
		*
		*	# return: last row of V as column vector
		*/
	return svd.vt.row(svd.vt.rows-1).t();
}

void test_solve()
{
	cv::Mat_<double> A(3, 3);
	A <<
	0.73596661, 1.2553854, 1.42286162,
	0.26980368, 0.47357525, 0.53439386,
	0.58560829, 0.89018126, 1.0281464;

	cv::Mat x = solve_Ax_eq_zero(A);
	printMatInfo("x", x);
	std::cout << x << std::endl;
	std::cout << "Ax " << A * x << std::endl;
}

cv::Mat_<double> linear_triangulation(cv::Mat const& proj_1, cv::Mat const& proj_2, cv::Mat const& pt_1, cv::Mat const& pt_2)
{
	// 	mat_A = np.vstack((
	// 		cross_product_matrix(p1_pix) @ proj1,
	// 					   cross_product_matrix(p2_pix) @ proj2
	// 	))
	//
	// 	P_homog = solve_AX_eq_zero(mat_A)
	// 	#P = P_homog[:3] / P_homog[3]
	// 	P_homog /= P_homog[3]
	// 	Ps.append(P_homog)
	//

	cv::Mat_<double> mat_A(6, 4);
	cv::Mat_<double> crosspr;
	cross_product_out(pt_1, crosspr);

	// 	printMatInfo("mat_A", mat_A);
	// 	printMatInfo("proj_1", proj_1);
	// 	printMatInfo("crosspr", crosspr);
	// 	printMatInfo("p1*cpr", crosspr*proj_1);
	mat_A(cv::Range(0, 3), cv::Range::all()) = crosspr * proj_1;
	cross_product_out(pt_2, crosspr);
	mat_A(cv::Range(3, 6), cv::Range::all()) = crosspr * proj_2;

	cv::Mat_<double> pos_homog(solve_Ax_eq_zero(mat_A));
	pos_homog /= pos_homog(3, 0);

	return pos_homog.rowRange(0, 3);
}

void test_triang()
{
	cv::Mat_<double> K(3, 3);
	K <<
	500.,	0.,		320.,
	0.,		500.,	240.,
	0.,		0.,		1.;

	cv::Mat_<double> RT1(3, 4);
	RT1 <<
	1.,		0.,		0.,		0.,
	0.,		1.,		0.,		0.,
	0.,		0.,		1.,		0.;

	cv::Mat_<double> RT2(3, 4);
	RT2 <<
	0.89,	0.,		-0.45,		25.,
	0.,		1.,			0.,		0.,
	0.45,	0.,		0.89,		0.;

	cv::Mat_<double> pts3d(4, 5);
	pts3d <<
	0,		10,		5,		-8,		-3,
	5,		3,		2,		1,		0,
	0,		0,		-5,		-3,		4,
	1,		1,		1,		1,		1;

	cv::Mat_<double> M1 = K*RT1;
	cv::Mat_<double> M2 = K*RT2;

	printMatInfo("K", K);
	printMatInfo("RT1", RT1);
	printMatInfo("M1", M1);
	printMatInfo("pts3d", pts3d);
	cv::Mat_<double> pts1 = M1 * pts3d;
	cv::Mat_<double> pts2 = M2 * pts3d;
	printMatInfo("pts1", pts1);

	for(int32_t pid = 0; pid < pts3d.cols; pid++)
	{
		std::cout << "lin: " << linear_triangulation(M1, M2, pts1.col(pid), pts2.col(pid)) << std::endl;
	}
}

void reverseRT(const cv::Mat_<double>& in_rt, cv::Mat_<double>& out_inverse_rt)
{
	out_inverse_rt.create(4, 4);
	out_inverse_rt.setTo(0);
	out_inverse_rt.rowRange(0, 3).colRange(0, 3) = in_rt.rowRange(0, 3).colRange(0, 3).t();
	for(size_t r : {0, 1, 2}) out_inverse_rt(r, 3) = -in_rt(r, 3);
	out_inverse_rt(3, 3) = 1.0;
}

void homogenous_to_3d(const cv::Mat& homog, cv::Mat_<double>& out_pts3d)
{
	out_pts3d.create(3, homog.cols);
	out_pts3d.setTo(0);
	
	for(size_t r : {0, 1, 2})
	{
		for(size_t c = 0; c < homog.cols; c++)
		{
			out_pts3d(r, c) = homog.at<float>(r, c) / homog.at<float>(3, c);
		}
	}
}

float point_distance_sq(cv::Point2f const& a, cv::Point2f const& b)
{
	const cv::Point2f diff(a-b);
	return diff.dot(diff);
}

void draw_cross(cv::Mat_<cv::Vec3b>& image, int x, int y, int thickness, int length, cv::Scalar const& color)
{
	cv::line(image, cv::Point2i(x-length, y), cv::Point2i(x+length, y), color, thickness);
	cv::line(image, cv::Point2i(x, y-length), cv::Point2i(x, y+length), color, thickness);
}
