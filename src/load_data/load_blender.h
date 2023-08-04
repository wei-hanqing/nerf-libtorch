#pragma once
#include <torch/torch.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
// mat --- tensor
//  |	   /
//	|	  /
//	|	 /	
// Eigen
class Blender_data {
public:
	Blender_data() {
	}

	~Blender_data() {
	}

	bool load(std::string basedir, bool half_res, std::int64_t testskip);


//private:
	std::shared_ptr<torch::Tensor> m_imgs;
	std::shared_ptr<torch::Tensor> m_poses;
	std::shared_ptr<torch::Tensor> m_render_poses;
	struct {
		int H;
		int W;
		double focal;
	}m_hwf;
	std::vector<std::shared_ptr<std::vector<std::int64_t>>> m_i_split;

};