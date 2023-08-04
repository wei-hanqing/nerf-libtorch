#pragma once
#include "create_nerf.h"
#include <boost/filesystem.hpp>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

torch::Tensor Embedder::embed(torch::Tensor inputs)
{
	std::vector<torch::Tensor> rlt;
	if (m_include_input) {
		rlt.emplace_back(inputs);
	}

	torch::Tensor freq_bands;
	if (m_log_sampling) {
		freq_bands = torch::pow(torch::tensor({ 2.0 }), torch::linspace(0.0, m_max_freq, m_N_freqs));
	}
	else {
		freq_bands = torch::linspace(std::pow(2.0, 0), std::pow(2.0, m_max_freq), m_N_freqs);
	}

	for (size_t j = 0; j < m_N_freqs; ++j) {
		torch::Tensor sin_val = torch::sin(inputs * freq_bands[j]);
		torch::Tensor cos_val = torch::cos(inputs * freq_bands[j]);
		rlt.emplace_back(sin_val);
		rlt.emplace_back(cos_val);
	}

	return torch::cat(rlt, 1);
}

int Embedder::out_dim()
{
	if (m_include_input) {
		m_out_dim += m_input_dims;
	}

	for (size_t j = 0; j < m_N_freqs; ++j) {
		m_out_dim += m_input_dims * 2;
	}

	return m_out_dim;
}

/// ////////////////////////////////////////////////////////////// class NeRF 

std::int64_t NeRF::create(boost::program_options::variables_map args)
{	// Instantiate NeRF's MLP model.
	int input_ch = get_embedder(args["multires"].as<std::int64_t>(), args["i_embed"].as<std::int64_t>());

	int input_ch_views = 0;
	if (args["use_viewdirs"].as<bool>()) {
		input_ch_views = get_embedder_views(args["multires_views"].as<std::int64_t>(), args["i_embed"].as<std::int64_t>());
	}

	int output_ch = 4;
	if (args["N_importance"].as<int64_t>() > 0) {
		output_ch = 5;
	}

	std::vector<std::int64_t> skips { 4 };
	this->model = std::make_shared<NeRFNet>(args["netdepth"].as<int64_t>(), args["netwidth"].as<int64_t>(),
		input_ch, output_ch, skips,
		input_ch_views, args["use_viewdirs"].as<bool>());
	this->model->to(torch::kCUDA);
	auto grad_vars = model->parameters();

	if (args["N_importance"].as<int64_t>() > 0) {
		this->model_fine = std::make_shared<NeRFNet>(args["netdepth_fine"].as<int64_t>(), args["netwidth_fine"].as<int64_t>(),
			input_ch, output_ch, skips,
			input_ch_views, args["use_viewdirs"].as<bool>());
		this->model_fine->to(torch::kCUDA);
		auto grad_vars_fine = model_fine->parameters();
		grad_vars.insert(grad_vars.end(), grad_vars_fine.begin(), grad_vars_fine.end());
	}

	// network_query_fn = run_network
	this->netchunk = args["netchunk"].as<std::int64_t>();

	// Create optimizer
	//torch::optim::Adam optimizer(
	//	grad_vars, torch::optim::AdamOptions(args["lrate"].as<float>()).betas(std::make_tuple(0.9, 0.999)));
	m_optimizer = std::make_shared<torch::optim::Adam>(grad_vars, torch::optim::AdamOptions(args["lrate"].as<float>()).betas(std::make_tuple(0.9, 0.999)));

	std::int64_t start = 0;
	auto basedir = boost::filesystem::path(args["basedir"].as<std::string>());
	auto expname = boost::filesystem::path(args["expname"].as<std::string>());

	///////////////////////////////////////
	// Load checkpoints
	std::vector<boost::filesystem::path> ckpts = { boost::filesystem::path(args["ft_path"].as<std::string>()) };
	if (args["ft_path"].as<std::string>() == "") {
		ckpts.clear();
		for (const auto& entry : boost::filesystem::directory_iterator(basedir / expname)) {
			std::string f = entry.path().filename().string();
			if (f.find(std::string("pt")) != std::string::npos) {
				ckpts.emplace_back(basedir / expname / entry.path().filename());
			}
		}
	}

	std::cout << "Found ckpts" << ckpts << std::endl;
	if (ckpts.size() > 0 && !args["no_reload"].as<bool>()) {
		std::string ckpt_path = ckpts.back().string();
		std::cout << "Reloading from" << ckpt_path << std::endl;
		auto model_fine_path = boost::filesystem::path(ckpt_path).parent_path() / "MF" / boost::filesystem::path(ckpt_path).filename();
		auto optim_path = boost::filesystem::path(ckpt_path).parent_path() / "OPTIM" / boost::filesystem::path(ckpt_path).filename();

		torch::serialize::InputArchive m_archive;
		m_archive.load_from(ckpt_path);
		this->model->load(m_archive);
		torch::IValue ivalue;
		m_archive.read(/*name=*/"global_step", /*value=*/ivalue);
		start = (int)ivalue.toInt();

		torch::load(*this->m_optimizer, optim_path.string());

		if (this->model_fine != nullptr) {
			torch::serialize::InputArchive mf_archive;
			mf_archive.load_from(model_fine_path.string());
			this->model_fine->load(mf_archive);
		}
	}

	/////////////////////////////////////////////////////////////////////////////////////
	render_kwars_train.perturb = args["perturb"].as<float>();
	render_kwars_train.N_importance = args["N_importance"].as<std::int64_t>();
	render_kwars_train.N_samples = args["N_samples"].as<std::int64_t>();
	render_kwars_train.use_viewdirs = args["use_viewdirs"].as<bool>();
	render_kwars_train.white_bkgd = args["white_bkgd"].as<bool>();
	render_kwars_train.raw_noise_std = args["raw_noise_std"].as<float>();
	// NDC only good for LLFF-style forward facing data
	if (args["dataset_type"].as<std::string>() != "llff" || args["no_ndc"].as<bool>()) {
		std::cout << "[NeRF][create nerf]: Not ndc!" << std::endl;
		render_kwars_train.ndc = false;
		render_kwars_train.lindisp = args["lindisp"].as<bool>();
	}
	
	render_kwars_test = render_kwars_train;
	render_kwars_test.perturb = false;
	render_kwars_test.raw_noise_std = 0.0;

	return start;
}

void NeRF::render(int H, int W, const torch::Tensor& K, std::int64_t chunk, std::shared_ptr<torch::Tensor> rays,
	std::shared_ptr<torch::Tensor> c2w, std::shared_ptr<torch::Tensor> c2w_staticcam, bool retraw, render_kwars kwars,
	torch::Tensor& rgb, torch::Tensor& disp, torch::Tensor& acc, std::unordered_map<std::string, torch::Tensor>& extras)
{
	torch::Tensor rays_o;
	torch::Tensor rays_d;
	if (c2w == nullptr) {
		// special case to render full image
		rays_o = rays->index({ 0 });
		rays_d = rays->index({ 1 });
	}
	else {
		get_rays(H, W, K, *c2w, rays_o, rays_d, c2w->device());
	}
	torch::Tensor viewdirs;
	if (kwars.use_viewdirs) {
		// provide ray directions as input
		viewdirs = rays_d;
		if (c2w_staticcam != nullptr) {
			get_rays(H, W, K, *c2w_staticcam, rays_o, rays_d, c2w_staticcam->device());
		}
		viewdirs = viewdirs / torch::norm(viewdirs, 2, { -1 }, true);
		viewdirs = torch::reshape(viewdirs, { -1, 3 });// .to(torch::kFloat32);
	}

	auto sh = rays_d.sizes().vec();
	if (kwars.ndc) {
		// for forward facing scenes
		ndc_rays(H, W, K[0][0].item<float>(), 1.0, rays_o, rays_d);
	}

	// Create ray batch
	rays_o = torch::reshape(rays_o, { -1, 3 }).to(torch::kFloat32);
	rays_d = torch::reshape(rays_d, { -1, 3 }).to(torch::kFloat32);

	torch::Tensor near_t = kwars.near * torch::ones_like(rays_d.index({ "...", torch::indexing::Slice(torch::indexing::None, 1) }));
	torch::Tensor far_t = kwars.far * torch::ones_like(rays_d.index({ "...", torch::indexing::Slice(torch::indexing::None, 1) }));
	auto rays_t = torch::cat({ rays_o, rays_d, near_t, far_t }, -1);
	rays = std::make_shared<torch::Tensor>(rays_t);
	if (kwars.use_viewdirs) {
		rays_t = torch::cat({ *rays, viewdirs }, -1);
		rays = std::make_shared<torch::Tensor>(rays_t);
	}

	// Render and reshape
	std::unordered_map<std::string, std::vector<torch::Tensor>> ret;
	for (size_t i = 0; i < rays->sizes().at(0); i += chunk) {
		//std::unordered_map<std::string, torch::Tensor> subret;
		render_rays(rays->index({ torch::indexing::Slice(i, i + chunk) }), retraw, kwars, ret, rays->device());
	}

	std::unordered_map<std::string, torch::Tensor> all_ret;
	for (auto& k : ret) {
		auto value = torch::cat(k.second, 0);
		std::vector<std::int64_t> arr;
		for (size_t i = 0; i < sh.size() - 1; i++) {
			arr.push_back(sh.at(i));
		}
		auto shape = value.sizes();
		for (size_t i = 1; i < shape.size(); i++) {
			arr.push_back(shape.at(i));
		}
		c10::IntArrayRef k_sh(arr.data(), arr.size());
		all_ret[k.first] = torch::reshape(value, k_sh);
	}

	for (auto& [k, value] : all_ret) {
		if (k == "rgb_map") {
			rgb = value;
		}
		else if (k == "disp_map") {
			disp = value;
		}
		else if (k == "acc_map") {
			acc = value;
		}
		else {
			extras[k] = value;
		}
	}
}

void NeRF::render_path(const torch::Tensor& render_poses, int H, int W, float focal, const torch::Tensor& K, std::int64_t chunk, 
	std::shared_ptr<torch::Tensor> gt_imgs, std::string savedir, std::int64_t render_factor, render_kwars kwars,
	torch::Tensor& rgbs, torch::Tensor& disps)
{
	if (render_factor != 0) {
		// Render downsampled for speed
		H = H / render_factor;
		W = W / render_factor;
		focal = focal / render_factor;
	}

	std::vector<torch::Tensor> rgb_vec;
	std::vector<torch::Tensor> disp_vec;
	std::int64_t N = render_poses.sizes().at(0);
	auto start = std::chrono::high_resolution_clock::now();
	for (size_t i = 0; i < N; i++) {
		auto stop = std::chrono::high_resolution_clock::now();
		auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
		std::cout << i << " Time taken by function: " << duration1.count() << " microseconds" << std::endl;
		start = std::chrono::high_resolution_clock::now();

		auto c2w = render_poses[i].index({ torch::indexing::Slice(torch::indexing::None, 3), torch::indexing::Slice(torch::indexing::None, 4) });
		torch::Tensor rgb, disp, acc;
		std::unordered_map<std::string, torch::Tensor> extras;
		this->render(H, W, K, chunk, nullptr,
			std::make_shared<torch::Tensor>(c2w), nullptr, false, kwars,
			rgb, disp, acc, extras);
		rgb_vec.emplace_back(rgb);
		disp_vec.emplace_back(disp);
		if (i == 0) {
			std::cout << rgb.sizes() << disp.sizes() << std::endl;
		}
		if (savedir != "") {
			std::ostringstream ss;
			ss << std::setw(3) << std::setfill('0') << i;
			std::string filename = ss.str() + ".png";
			//auto bgr8 = (255 * torch::clip(rgb_vec.back(), 0, 1)).to(torch::kUInt8).to(torch::kCPU);
			//std::cout << rgb_vec.back()[196] << std::endl;
			auto bgr8 = rgb_vec.back().mul(255).clamp(0, 255).to(torch::kUInt8).to(torch::kCPU);
			auto path = boost::filesystem::path(savedir) / filename;
			cv::Mat image(W, H, CV_8UC3, bgr8.data_ptr());
			//image.create(bgr8.sizes().at(1), bgr8.sizes().at(0), CV_8UC3);
			//memcpy((void*)image.data, bgr8.data_ptr(), sizeof(torch::kUInt8) * bgr8.numel());
			cv::Mat rgb8;
			cv::cvtColor(image, rgb8, cv::COLOR_BGR2RGB);
			cv::imwrite(path.string(), rgb8);
		}
	}
	rgbs = torch::stack(rgb_vec, 0);
	disps = torch::stack(rgb_vec, 0);
}

int NeRF::get_embedder(int multires, int i_embed)
{

	if (i_embed == -1) {
		this->embed_fn = torch::nn::Identity();
		//this->embed_fn = [](torch::Tensor x) -> torch::Tensor {return torch::nn::Identity(x); };
		return 3;
	}

	this->embedder = std::make_shared<Embedder>(multires);
	this->embed_fn = [&](torch::Tensor x) { return this->embedder->embed(x); };
	return this->embedder->out_dim();
}

int NeRF::get_embedder_views(int multires, int i_embed)
{

	if (i_embed == -1) {
		this->embeddirs_fn = torch::nn::Identity();
		return 3;
	}

	this->embedder_views = std::make_shared<Embedder>(multires);
	this->embeddirs_fn = [&](torch::Tensor x) { return this->embedder_views->embed(x); };
	return this->embedder_views->out_dim();
}

void NeRF::network_query_fn(const torch::Tensor& inputs, std::shared_ptr<torch::Tensor> viewdirs, const NeRFNet::Ptr fn, torch::Tensor& outputs)
{
	// Prepares inputs and applies network 'fn'.
	torch::Tensor inputs_flat = torch::reshape(inputs, { -1, inputs.sizes().back() });
	if (embed_fn == nullptr) {
		std::cout << "[NeRF]: embed_fn is nullptr!!!" << std::endl;
	}
	torch::Tensor embedded = this->embed_fn(inputs_flat);

	if (viewdirs != nullptr) {
		torch::Tensor input_dirs = viewdirs->index({ torch::indexing::Slice{}, torch::indexing::None }).expand(inputs.sizes());
		torch::Tensor input_dirs_flat = torch::reshape(input_dirs, { -1, input_dirs.sizes().back() });
		if (embeddirs_fn == nullptr) {
			std::cout << "[NeRF]: embed_fn is nullptr!!!" << std::endl;
		}
		torch::Tensor embedded_dirs = this->embeddirs_fn(input_dirs_flat);
		embedded = torch::cat({ embedded, embedded_dirs }, -1);
	}

	// outputs_flat = batchify(fn, netchunk)(embedded)
	// Constructs a version of 'fn' that applies to smaller batches.
	torch::Tensor outputs_flat;
	if (netchunk == 0) {
		outputs_flat = fn->forward(embedded);
	}
	else {
		std::vector<torch::Tensor> vec;
		for (size_t i = 0; i < embedded.sizes().at(0); i += netchunk) {
			vec.emplace_back(fn->forward(embedded.index({ torch::indexing::Slice(i, i + netchunk) })));
		}
		outputs_flat = torch::cat(vec, 0);
	}
	std::vector<std::int64_t> arr;
	auto inputs_sizes = inputs.sizes();
	for (size_t i = 0; i < inputs_sizes.size() - 1; i++) {
		arr.push_back(inputs_sizes.at(i));
	}
	arr.push_back(outputs_flat.sizes().back());
	c10::IntArrayRef re_size(arr.data(), arr.size());
	outputs = torch::reshape(outputs_flat, re_size);
}

void NeRF::render_rays(torch::Tensor& ray_batch, bool retraw, render_kwars kwars, std::unordered_map<std::string, std::vector<torch::Tensor>>& ret, torch::Device dev)
{
	std::int64_t N_rays = ray_batch.sizes().at(0);
	auto rays_o = ray_batch.index({ torch::indexing::Slice(), torch::indexing::Slice(0, 3) });
	auto rays_d = ray_batch.index({ torch::indexing::Slice(), torch::indexing::Slice(3, 6) });
	//torch::Tensor* viewdirs = nullptr;
	std::shared_ptr<torch::Tensor> viewdirs = nullptr;
	if (ray_batch.sizes().back() > 8) {
		auto viewdirs_t = ray_batch.index({ torch::indexing::Slice(), torch::indexing::Slice(-3, torch::indexing::None) }).to(dev);
		viewdirs = std::make_shared<torch::Tensor>(viewdirs_t);
	}
	auto bounds = torch::reshape(ray_batch.index({ "...", torch::indexing::Slice(6, 8) }), { -1, 1, 2 });
	auto near = bounds.index({ "...", 0 });
	auto far = bounds.index({ "...", 1 });

	auto t_vals = torch::linspace(0.0, 1.0, this->render_kwars_train.N_samples).to(dev);
	torch::Tensor z_vals;
	if (render_kwars_train.lindisp) {
		z_vals = 1.0 / (1.0 / near * (1.0 - t_vals) + 1.0 / far * (t_vals));
	}
	else {
		z_vals = near * (1.0 - t_vals) + far * (t_vals);
	}
	z_vals = z_vals.expand({ N_rays, this->render_kwars_train.N_samples });
	
	if (this->render_kwars_train.perturb > 0.0) {
		// get intervals between samples
		auto mids = 0.5 * (z_vals.index({ "...", torch::indexing::Slice(1, torch::indexing::None) }) + z_vals.index({ "...", torch::indexing::Slice(torch::indexing::None, -1) }));
		auto upper = torch::cat({ mids, z_vals.index({"...", torch::indexing::Slice(-1, torch::indexing::None)}) }, -1);
		auto lower = torch::cat({ z_vals.index({"...", torch::indexing::Slice(torch::indexing::None, 1)}), mids }, -1);
		// stratified samples in those intervals
		auto t_rand = torch::rand(z_vals.sizes()).to(dev);
		z_vals = lower + (upper - lower) * t_rand;
	}
	auto pts = rays_o.index({ "...", torch::indexing::None, torch::indexing::Slice() }) +
		rays_d.index({ "...", torch::indexing::None, torch::indexing::Slice() }) * z_vals.index({ "...", torch::indexing::Slice(), torch::indexing::None });
	torch::Tensor raw;
	network_query_fn(pts, viewdirs, this->model, raw);
	torch::Tensor rgb_map, disp_map, acc_map, weights, depth_map;
	raw2outputs(raw, z_vals, rays_d, kwars.raw_noise_std, kwars.white_bkgd, 
		rgb_map, disp_map, acc_map, weights, depth_map);

	torch::Tensor rgb_map_0;
	torch::Tensor dis_map_0;
	torch::Tensor acc_map_0;
	torch::Tensor z_samples;
	if (kwars.N_importance > 0) {
		rgb_map_0 = rgb_map;
		dis_map_0 = disp_map;
		acc_map_0 = acc_map;
		auto z_vals_mid = 0.5 * (z_vals.index({ "...", torch::indexing::Slice(1, torch::indexing::None) }) + z_vals.index({ "...", torch::indexing::Slice(torch::indexing::None, -1) }));
		sample_pdf(z_vals_mid, weights.index({"...", torch::indexing::Slice(1, -1)}), kwars.N_importance, kwars.perturb==0.0, z_samples, z_vals_mid.device());
		z_samples.detach();

		auto tuple_t = torch::sort(torch::cat({ z_vals, z_samples }, -1), -1);
		z_vals = std::get<0>(tuple_t);
		pts = rays_o.index({ "...", torch::indexing::None, torch::indexing::Slice() }) +
			rays_d.index({ "...", torch::indexing::None, torch::indexing::Slice() }) * z_vals.index({ "...", torch::indexing::Slice(), torch::indexing::None });

		if (this->model_fine == nullptr) {
			network_query_fn(pts, viewdirs, this->model, raw);
		}
		else {
			network_query_fn(pts, viewdirs, this->model_fine, raw);
		}
		raw2outputs(raw, z_vals, rays_d, kwars.raw_noise_std, kwars.white_bkgd,
			rgb_map, disp_map, acc_map, weights, depth_map);
	}

	ret["rgb_map"].emplace_back(rgb_map);
	ret["disp_map"].emplace_back(disp_map);
	ret["acc_map"].emplace_back(acc_map);
	if (retraw) {
		ret["raw"].emplace_back(raw);
	}
	if (kwars.N_importance > 0) {
		ret["rgb0"].emplace_back(rgb_map_0);
		ret["disp0"].emplace_back(dis_map_0);
		ret["acc0"].emplace_back(acc_map_0);
		ret["z_std"].emplace_back(torch::std(z_samples, { -1 }, false));
	}
	/*
	for k in ret:
		if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
			print(f"! [Numerical Error] {k} contains nan or inf.")
	*/
}

void NeRF::raw2outputs(const torch::Tensor& raw, const torch::Tensor& z_vals, const torch::Tensor& rays_d, float raw_noise_std, bool white_bkgd,
	torch::Tensor& rgb_map, torch::Tensor& disp_map, torch::Tensor& acc_map, torch::Tensor& weights, torch::Tensor& depth_map)
{
	// Transforms model's predictions to semantically meaningful values.
	auto dists = z_vals.index({ "...", torch::indexing::Slice(1, torch::indexing::None) }) + z_vals.index({ "...", torch::indexing::Slice(torch::indexing::None, -1) });
	dists = torch::cat({ dists, torch::tensor(1e10).expand(dists.index({"...", torch::indexing::Slice(torch::indexing::None, 1)}).sizes()).to(torch::kCUDA) }, -1);
	dists = dists * torch::norm(rays_d.index({ "...", torch::indexing::None, torch::indexing::Slice() }), /*dim*/-1);
	auto rgb = torch::sigmoid(raw.index({ "...", torch::indexing::Slice(torch::indexing::None, 3) }));
	torch::Tensor alpha;
	if (raw_noise_std > 0.0) {
		auto noise = torch::randn(raw.index({ "...", 3 }).sizes()) * raw_noise_std;
		alpha = 1.0 - torch::exp(-torch::relu(raw.index({ "...", 3 }) + noise) * dists);
	}
	else {
		alpha = 1.0 - torch::exp(-torch::relu(raw.index({ "...", 3 })) * dists);
	}
	weights = alpha * torch::cumprod(torch::cat({ torch::ones({alpha.sizes().at(0), 1}).to(torch::kCUDA), 1.0 - alpha + 1e-10 }, -1), -1).index({ torch::indexing::Slice(), torch::indexing::Slice(torch::indexing::None, -1) });
	rgb_map = torch::sum(weights.index({ "...", torch::indexing::None }) * rgb, -2);
	depth_map = torch::sum(weights * z_vals, -1);
	disp_map = 1.0 / torch::max(1e-10 * torch::ones_like(depth_map).to(torch::kCUDA), depth_map / torch::sum(weights, -1));
	acc_map = torch::sum(weights, -1);

	if (white_bkgd) {
		rgb_map = rgb_map + (1.0 - acc_map.index({ "...", torch::indexing::None }));
	}
}

// Ray helpers
void get_rays(int H, int W, const torch::Tensor& K, const torch::Tensor& c2w, torch::Tensor& rays_o, torch::Tensor& rays_d, torch::Device dev)
{
	std::vector<torch::Tensor> meshgrid = torch::meshgrid({ torch::linspace(0, W - 1, W), torch::linspace(0, H - 1, H) });
	torch::Tensor i = meshgrid[0];
	torch::Tensor j = meshgrid[1];
	//i = i.t();
	//j = j.t();
	torch::Tensor dirs = torch::stack({ (i - K[0][2]) / K[0][0], -(j - K[1][2]) / K[1][1], -torch::ones_like(i) }, -1).to(dev);
	// Rotate ray directions from camera frame to the world frame
	rays_d = torch::sum(dirs.index({ "...", torch::indexing::None, torch::indexing::Slice() }) *
		c2w.index({ torch::indexing::Slice(torch::indexing::None, 3), torch::indexing::Slice(torch::indexing::None, 3) }), -1);
	rays_d = torch::sum(torch::unsqueeze(dirs, -2) * c2w.index({ torch::indexing::Slice(torch::indexing::None, 3), torch::indexing::Slice(torch::indexing::None, 3) }), -1);
	// Translate camera frame's origin to the world frame. It is the origin of all rays.
	rays_o = c2w.index({ torch::indexing::Slice(torch::indexing::None, 3), -1 }).expand(rays_d.sizes());
}

void ndc_rays(int H, int W, float focal, float near, torch::Tensor& rays_o, torch::Tensor& rays_d)
{
	// Shift ray origins to near plane
	torch::Tensor t = -(near + rays_o.index({ "...", 2 })) / rays_d.index({ "...", 2 });
	rays_o = rays_o + t.index({ "...", torch::indexing::None }) * rays_d;

	// Projection
	torch::Tensor o0 = -1.0 / (W / (2.0 * focal)) * rays_o.index({ "...", 0 }) / rays_o.index({ "...", 2 });
	torch::Tensor o1 = -1.0 / (H / (2.0 * focal)) * rays_o.index({ "...", 1 }) / rays_o.index({ "...", 2 });
	torch::Tensor o2 = 1.0 + 2.0 * near / rays_o.index({ "...", 2 });

	torch::Tensor d0 = -1.0 / (W / (2.0 * focal)) * (rays_d.index({ "...", 0 }) / rays_d.index({ "...", 2 }) - rays_o.index({ "...", 0 }) / rays_o.index({ "...", 2 }));
	torch::Tensor d1 = -1.0 / (H / (2.0 * focal)) * (rays_d.index({ "...", 1 }) / rays_d.index({ "...", 2 }) - rays_o.index({ "...", 1 }) / rays_o.index({ "...", 2 }));
	torch::Tensor d2 = -2.0 * near / rays_o.index({ "...", 2 });

	rays_o = torch::stack({ o0, o1, o2 }, -1);
	rays_d = torch::stack({ d0, d1, d2 }, -1);
}

void sample_pdf(const torch::Tensor& bins, torch::Tensor& weights, std::int64_t N_samples, bool det, torch::Tensor& samples, torch::Device dev)
{
	// Get pdf
	weights = weights + 1e-5;
	auto pdf = weights / torch::sum(weights, -1, true);
	auto cdf = torch::cumsum(pdf, -1);
	cdf = torch::cat({ torch::zeros_like(cdf.index({"...", torch::indexing::Slice(torch::indexing::None, 1)})).to(dev), cdf }, -1);

	// Take uniform samples
	std::vector<std::int64_t> arr;
	auto inputs_sizes = cdf.sizes();
	for (size_t i = 0; i < inputs_sizes.size() - 1; i++) {
		arr.push_back(inputs_sizes.at(i));
	}
	arr.push_back(N_samples);
	c10::IntArrayRef re_size(arr.data(), arr.size());

	auto u = torch::rand(re_size).to(dev);
	if (det) {
		u = torch::linspace(0.0, 1.0, N_samples);
		u = u.expand(re_size).to(dev);
	}

	// Invert CDF
	u = u.contiguous();
	auto inds = torch::searchsorted(cdf, u, true);
	auto below = torch::max(torch::zeros_like(inds - 1).to(dev), inds - 1);
	auto above = torch::min((cdf.sizes().back() - 1) * torch::ones_like(inds).to(dev), inds);
	auto inds_g = torch::stack({ below, above }, -1).to(torch::kLong);

	c10::IntArrayRef matched_shape{ inds_g.sizes().at(0), inds_g.sizes().at(1), cdf.sizes().back() };
	auto cdf_g = torch::gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g);
	auto bins_g = torch::gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g);

	auto denom = (cdf_g.index({ "...", 1 }) - cdf_g.index({ "...", 0 }));
	denom = torch::where(denom < 1e-5, torch::ones_like(denom).to(dev), denom);
	auto t = (u - cdf_g.index({ "...", 0 })) / denom;
	samples = bins_g.index({ "...", 0 }) + t * (bins_g.index({ "...", 1 }) - bins_g.index({ "...", 0 }));
}

