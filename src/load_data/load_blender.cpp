#include "load_blender.h"
//#include "log_torch.h"
#include <vector>
#include <fstream>
#include <boost/filesystem.hpp>
#include <iostream>
#include <nlohmann/json.hpp>


#define M_PI 3.14159265358979323846

//using namespace cv;
using json = nlohmann::json;

void LOG_Tensor_shape(const torch::Tensor& tensor) {
	auto shape = tensor.sizes();
	// Print the shape dimensions
	std::cout << "Tensor shape: [";
	for (size_t i = 0; i < shape.size(); ++i) {
		std::cout << shape[i];
		if (i < shape.size() - 1) {
			std::cout << ", ";
		}
	}
	std::cout << "]" << std::endl;
}

void LOG_Tensor_elements(const torch::Tensor& tensor) {
	for (size_t i = 0; i < tensor.sizes().at(0); i++) {
		for (size_t j = 0; j < tensor.sizes().at(1); j++) {
			std::cout << "tensor[" << i << "][" << j << "]: " << tensor[i][j].item<float>() << "\n";
		}
	}
}

//auto trans_t = [](float t) -> torch::Tensor {
//	return torch::tensor({ {1., 0., 0., 0.},
//							{0., 1., 0., 0.}, 
//							{0., 0., (double)t, 0.}, 
//							{0., 0., 0., 1.} }).to(torch::kFloat32);
//};
//
//auto rot_phi = [](float phi) -> torch::Tensor {
//	return torch::tensor({ {1., 0., 0., 0.},
//							{0., cos((double)phi), -sin((double)phi), 0.},
//							{0., sin((double)phi), cos((double)phi), 0.},
//							{0., 0., 0., 1.} }).to(torch::kFloat32);
//};
//
//auto rot_theta = [](float th) -> torch::Tensor {
//	return torch::tensor({ {cos((double)th), 0., -sin((double)th), 0.},
//							{0., 1., 0., 0.},
//							{sin((double)th), 0., cos((double)th), 0.},
//							{0., 0., 0., 1.} }).to(torch::kFloat32);
//};

torch::Tensor pose_spherical(float theta, float phi, float radius) {
	const auto trans_t = [&]() -> torch::Tensor 
	{
		return torch::tensor({ {1., 0., 0., 0.},
								{0., 1., 0., 0.},
								{0., 0., (double)theta, 0.},
								{0., 0., 0., 1.} }).to(torch::kFloat32);
	};
	const auto rot_phi = [&]() -> torch::Tensor {
		auto a = phi / 180.0 * M_PI;
		return torch::tensor({ {1., 0., 0., 0.},
								{0., cos((double)a), -sin((double)a), 0.},
								{0., sin((double)a), cos((double)a), 0.},
								{0., 0., 0., 1.} }).to(torch::kFloat32);
	};
	const auto rot_theta = [&]() -> torch::Tensor {
		auto b = theta / 180.0 * M_PI;
		return torch::tensor({ {cos((double)b), 0., -sin((double)b), 0.},
								{0., 1., 0., 0.},
								{sin((double)b), 0., cos((double)b), 0.},
								{0., 0., 0., 1.} }).to(torch::kFloat32);
	};

	torch::Tensor array = torch::tensor({ {-1, 0, 0, 0}, 
											{0, 0, 1, 0}, 
											{0, 1, 0, 1}, 
											{0, 0, 0, 1} }).to(torch::kFloat32);
	torch::Tensor c2w = trans_t();
	c2w = torch::mm(rot_phi(), c2w);
	c2w = torch::mm(rot_theta(), c2w);
	c2w = torch::mm(array, c2w);
	//LOG_Tensor_elements(c2w);
	return c2w;
}

bool Blender_data::load(std::string basedir, bool half_res, std::int64_t testskip)
{
	torch::Device dev = torch::kCPU;
	std::vector<std::string> splits{ "train", "val", "test" };
	
	std::unordered_map<std::string, json> metas;
	for (auto s : splits) {
		std::string subpath = "transforms_" + s + ".json";
		boost::filesystem::path file_path = boost::filesystem::path(basedir) / subpath;
		if (!boost::filesystem::exists(file_path)) {
			std::cout << "[ERROR] Path does not exist!" << std::endl;
			return false;
		}
		std::ifstream file(file_path.string());

		json data;
		file >> data;
		//std::cout << data.dump() <<std::endl;
		metas.insert({ s, data });
	}

	std::vector<torch::Tensor> all_imgs;
	std::vector<torch::Tensor> all_poses;
	std::vector<std::int64_t> counts{ 0 };

	std::vector<torch::Tensor> imgs;
	std::vector<torch::Tensor> poses;

	json meta;
	for (auto s : splits) {
		meta = metas[s];
		std::int64_t skip = testskip;
		if (s == "train" || testskip == 0)
			skip = 1;

		imgs.clear();
		poses.clear();
		std::int64_t step = 0;
		for (auto frame : meta["frames"]) {
			if (step % skip != 0) {
				step++;
				continue;
			}
			std::string subpath = (std::string)frame["file_path"] + ".png";
			boost::filesystem::path fname = boost::filesystem::path(basedir) / subpath;
			if (!boost::filesystem::exists(fname)) {
				std::cout << "[ERROR] Path does not exist!" << std::endl;
				return false;
			}
			// !!! In the case of color images, the decoded images will have the channels stored in B G R order.
			cv::Mat bgr_image = cv::imread(fname.string(), cv::IMREAD_UNCHANGED);	
			if (bgr_image.empty()) {
				std::cout << "[ERROR] Could not read image" << std::endl;
				return false;
			}
			// 
			cv::Mat image;
			cv::cvtColor(bgr_image, image, cv::COLOR_BGR2RGB);
			if (half_res) {
				cv::Mat resized_image;
				cv::resize(image, resized_image, cv::Size(), 0.5, 0.5, cv::INTER_AREA);
				image = resized_image;
			}
			auto options = torch::TensorOptions().dtype(torch::kByte).device(torch::kCPU);
			auto img_data = torch::from_blob(image.data, { image.rows, image.cols, image.channels() }, options);
			//img_data = img_data.permute({ 0, 3, 1, 2 });  // Change the layout from NHWC to NCHW
			img_data = img_data.to(torch::kFloat32).mul(1.0 / 255.0).to(dev);
			imgs.emplace_back(img_data);

			std::array<std::array<float, 4>, 4> data = frame["transform_matrix"].get<std::array<std::array<float, 4>, 4>>();
			options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
			auto transform_matrix = torch::from_blob(data.data(), { (std::int64_t)frame["transform_matrix"].size(), (std::int64_t)frame["transform_matrix"][0].size() }, options);
			poses.emplace_back(transform_matrix.to(dev));
			step++;
		}

		int n_imgs = imgs.size();
		int n_poses = poses.size();

		auto pose = torch::stack(poses, /*dim=*/0);	// Nx4x4						
		all_poses.emplace_back(pose);												
		auto img = torch::stack(imgs, /*dim=*/0);									
		all_imgs.emplace_back(img);													
		
		counts.emplace_back(counts.back() + n_imgs);
	}

	for (size_t split = 0; split < 3; split++) {
		std::vector<std::int64_t> new_vector;
		for (size_t i = counts[split]; i < counts[split + 1]; i++) {
			new_vector.push_back(i);
		}
		this->m_i_split.emplace_back(std::make_shared<std::vector<std::int64_t>>(new_vector));
	}
	this->m_poses = std::make_shared<torch::Tensor>(torch::cat(all_poses, /*dim=*/0));
	this->m_imgs = std::make_shared<torch::Tensor>(torch::cat(all_imgs, /*dim=*/0));

	auto shape = this->m_imgs->sizes().vec();
	int N = shape[0];
	int H = shape[1];
	int W = shape[2];
	float camera_angle_x = float(meta["camera_angle_x"]);
	float focal = 0.5 * W / tan(0.5 * camera_angle_x);
	if (half_res) {
		focal /= 2.0;
	}

	std::vector<torch::Tensor> c2ws(40);
	for (size_t i = 0; i < 40; i++) {
		float angle = -180.0 + i * 9.0;
		c2ws[i] = pose_spherical(angle, -30.0, 4.0).to(dev);
	}


	this->m_render_poses = std::make_shared<torch::Tensor>(torch::stack(c2ws, /*dim=*/0));
	
	this->m_hwf.H = H;
	this->m_hwf.W = W;
	this->m_hwf.focal = focal;

	std::cout << "Loaded blender: " << std::endl;
	std::cout << basedir << std::endl;
	LOG_Tensor_shape(*m_imgs);
	LOG_Tensor_shape(*m_render_poses);
	std::cout << "[H, W, F]: [" << H << ", " << W << ", " << focal << "]" << std::endl;
	//std::cout << "[Warning] image is BGR!!! BGR!!! BGR!!!, not RGB!!! RGB!!! RGB!!!" << std::endl;
	// right print code and wrong print code
	//std::cout << m_imgs->index({ 0, 200, 168 }) << std::endl;		
	//std::cout << (*m_imgs)[0][200][168][0].item<float>() << std::endl;
	//std::cout << (*m_imgs)[0][200][168][0].item<float>() << std::endl;
	//std::cout << (*m_imgs)[0][200][168][0].item<float>() << std::endl;
	//std::cout << (*m_imgs)[0][200][168][0].item<float>() << std::endl;
	return true;
}