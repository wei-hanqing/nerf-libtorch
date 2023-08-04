#pragma once
#include <torch/torch.h>
#include <boost/program_options.hpp>

class Embedder {
public:
	using Ptr = std::shared_ptr<Embedder>;

	Embedder(int multires) {
		m_include_input = true;
		m_input_dims = 3;
		m_max_freq = multires - 1;
		m_N_freqs = multires;
		m_log_sampling = true;
		m_out_dim = 0;
	};
	~Embedder() {};

	torch::Tensor embed(torch::Tensor inputs);
	int out_dim();

private:
	int m_out_dim;

	bool m_include_input;
	int m_input_dims;
	int m_max_freq;
	int m_N_freqs;
	bool m_log_sampling;

};

// Define a new Module.
struct NeRFNet : torch::nn::Module {
	using Ptr = std::shared_ptr<NeRFNet>;
	NeRFNet(int D, int W, int input_ch, int output_ch, std::vector<std::int64_t> skips, int input_ch_views, bool use_viewdirs) {
		// Construct and register two Linear submodules.
		this->D = D;
		this->W = W;
		this->input_ch = input_ch;
		this->input_ch_views = input_ch_views;
		this->output_ch = output_ch;
		this->use_viewdirs = use_viewdirs;
		this->skips = skips;
		//skips.insert(skips.end(), skips_a.begin(), skips_a.end());

		pts_linears->push_back(register_module("module" + std::to_string(0), torch::nn::Linear(input_ch, W)));
		for (size_t i = 0; i < D - 1; i++) {
			if (std::find(skips.begin(), skips.end(), i) != skips.end()) {
				pts_linears->push_back(register_module("module" + std::to_string(i+1), torch::nn::Linear(W + input_ch, W)));
			}
			else {
				pts_linears->push_back(register_module("module" + std::to_string(i+1), torch::nn::Linear(W, W)));
			}
		}
		//for (const auto& module : *pts_linears) {
		//	module->pretty_print(std::cout);
		//}
		// Convert all modules to CUDA.
		//pts_linears->to(torch::kCUDA);

		// Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
		views_linears->push_back(register_module("views_linear", torch::nn::Linear(input_ch_views + W, W / 2)));
		//for (const auto& module : *views_linears) {
		//	module->pretty_print(std::cout);
		//}
		//views_linears->to(torch::kCUDA);

		if (use_viewdirs) {
			feature_linear = register_module("feature_linear", torch::nn::Linear(W, W));
			alpha_linear = register_module("alpha_linear", torch::nn::Linear(W, 1));
			rgb_linear = register_module("rgb_linear", torch::nn::Linear(W / 2, 3));
		}
		else {
			output_linear = register_module("output_linear", torch::nn::Linear(W, output_ch));
		}
	}

	// Implement the Net's algorithm.
	torch::Tensor forward(torch::Tensor x) {
		auto inputs = torch::split_with_sizes(x, { this->input_ch, this->input_ch_views }, /*dim=*/-1);
		torch::Tensor input_pts = inputs[0];
		torch::Tensor input_views = inputs[1];
		auto h = input_pts;
		// Use one of many tensor manipulation functions.
		for (size_t i = 0; i < this->pts_linears->size(); i++) {
			h = this->pts_linears[i]->as<torch::nn::Linear>()->forward(h);
			//auto linear_i = this->pts_linears[i]->as<torch::nn::Linear>();
			//h = linear_i->forward(h);
			h = torch::relu(h);
			if (std::find(this->skips.begin(), this->skips.end(), i) != this->skips.end()) {
				h = torch::cat({ input_pts, h }, /*dim=*/-1);
			}
		}

		torch::Tensor outputs;
		if (this->use_viewdirs) {
			auto alpha = this->alpha_linear->forward(h);
			auto feature = this->feature_linear->forward(h);
			h = torch::cat({ feature, input_views },/*dim=*/-1);

			for (size_t i = 0; i < this->views_linears->size(); i++) {
				h = torch::relu(this->views_linears[i]->as<torch::nn::Linear>()->forward(h));
			}

			auto rgb = this->rgb_linear->forward(h);
			outputs = torch::cat({ rgb, alpha },/*dim=*/-1);
		}
		else {
			outputs = this->output_linear->forward(h);
		}

		return outputs;
	}

	// Use one of many "standard library" modules.
	torch::nn::Linear feature_linear{ nullptr }, alpha_linear{ nullptr }, rgb_linear{ nullptr };
	torch::nn::Linear output_linear{ nullptr };

	// member
	int D = 8;
	int W = 256;
	int input_ch = 3;
	int input_ch_views = 3;
	int output_ch = 4;
	std::vector<std::int64_t> skips{ 4 };
	bool use_viewdirs = false;

	torch::nn::ModuleList pts_linears;
	torch::nn::ModuleList views_linears;
};

class NeRF {
public:
	struct render_kwars {
		float perturb;
		std::int64_t N_importance;
		std::int64_t N_samples;
		bool use_viewdirs;
		bool white_bkgd;
		float raw_noise_std;
		bool ndc;
		bool lindisp;
		float near;
		float far;
	} render_kwars_train, render_kwars_test;
	std::shared_ptr < torch::optim::Adam> m_optimizer;

	std::int64_t create(boost::program_options::variables_map args);
	void render(int H, int W, const torch::Tensor& K, std::int64_t chunk, std::shared_ptr<torch::Tensor> rays,
		std::shared_ptr<torch::Tensor> c2w, std::shared_ptr<torch::Tensor> c2w_staticcam, bool retraw, render_kwars kwars,
		torch::Tensor& rgb, torch::Tensor& disp, torch::Tensor& acc, std::unordered_map<std::string, torch::Tensor>& extras);
	void render_path(const torch::Tensor& render_poses, int H, int W, float focal, const torch::Tensor& K, std::int64_t chunk, 
		std::shared_ptr<torch::Tensor> gt_imgs, std::string savedir, std::int64_t render_factor, render_kwars kwars,
		torch::Tensor& rgbs, torch::Tensor& disps);

private:
	// Positional encoding
	int get_embedder(int multires, int i_embed);
	int get_embedder_views(int multires, int i_embed);
	void network_query_fn(const torch::Tensor& inputs, std::shared_ptr<torch::Tensor> viewdirs, const NeRFNet::Ptr fn, torch::Tensor& outputs);
	void render_rays(torch::Tensor& ray_batch, bool retraw, render_kwars kwars, std::unordered_map<std::string, std::vector<torch::Tensor>>& ret, torch::Device dev);
	void raw2outputs(const torch::Tensor& raw, const torch::Tensor& z_vals, const torch::Tensor& rays_d, float raw_noise_std, bool white_bkgd, 
		torch::Tensor& rgb_map, torch::Tensor& disp_map, torch::Tensor& acc_map, torch::Tensor& weights, torch::Tensor& depth_map);

	std::function<torch::Tensor(torch::Tensor)> embed_fn = [](torch::Tensor a) { return a; };
	std::function<torch::Tensor(torch::Tensor)> embeddirs_fn = [](torch::Tensor a) { return a; };

	Embedder::Ptr embedder = nullptr;
	Embedder::Ptr embedder_views = nullptr;

public:
	// Model
	NeRFNet::Ptr model = nullptr;
	NeRFNet::Ptr model_fine = nullptr;
	//std::shared_ptr< torch::autograd::variable_list> m_grad_vars;

	// run_network
	std::int64_t netchunk = 1024 * 64;
};



// Ray helpers
void get_rays(int H, int W, const torch::Tensor& K, const torch::Tensor& c2w, torch::Tensor& rays_o, torch::Tensor& rays_d, torch::Device dev);
void ndc_rays(int H, int W, float focal, float near, torch::Tensor& rays_o, torch::Tensor& rays_d);
void sample_pdf(const torch::Tensor& bins, torch::Tensor& weights, std::int64_t N_samples, bool det, torch::Tensor& samples, torch::Device dev);
