#include <torch/torch.h>
#include <string>
#include <iostream>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <direct.h>
#include <random>
#include <algorithm>

#include "config/configs.h"
#include "load_data/load_blender.h"
#include "nerf/create_nerf.h"

namespace po = boost::program_options;
using namespace torch::indexing;

int main() {
    std::cout << "[main] run nerf is starting !!!" << std::endl;
    if (torch::cuda::is_available()) {
        std::cout << "CUDA is available! Training on GPU" << std::endl;
    }
    else
    {
        std::cout << "CUDA is not available! Training on CPU" << std::endl;
    }

    po::variables_map args;
    ConfigBuilder config_builder;
    if (!config_builder.fromfile("configs/lego.txt", args)) {
        return -1;
    }

    // Load data
    Blender_data blender_data;
    std::shared_ptr<std::vector<std::int64_t>> i_train;
    std::shared_ptr<std::vector<std::int64_t>> i_val;
    std::shared_ptr<std::vector<std::int64_t>> i_test;
    float near;
    float far;
    if (args["dataset_type"].as<std::string>() == "blender") {
        auto success = blender_data.load(args["datadir"].as<std::string>(), args["half_res"].as<bool>(), args["testskip"].as<std::int64_t>());
        if (!success) {
            std::cout << "[main] load blender data is failed" << std::endl;
            return -1;
        }

        i_train = blender_data.m_i_split[0];
        i_val = blender_data.m_i_split[1];
        i_test = blender_data.m_i_split[2];
        
        near = 2.0;
        far = 6.0;

        bool white_bkgd = args["white_bkgd"].as<bool>();
        if (white_bkgd) {
            auto a = blender_data.m_imgs->index({ "...", Slice(None, 3) });     // N, H, W, 3
            auto b = blender_data.m_imgs->index({ "...", Slice(-1, None) });    // N, H, W, 1
            auto c = a * b + (1.0 - b);                                         // N, H, W, 3
            blender_data.m_imgs = std::make_shared<torch::Tensor>(c);
        }
    }
    
    // Cast intrinsics to right types
    int H = blender_data.m_hwf.H;
    int W = blender_data.m_hwf.W;
    float focal = blender_data.m_hwf.focal;

    torch::Tensor K = torch::tensor({ {(double)focal, 0.0, 0.5 * W},
                                    {0.0, (double)focal, 0.5 * H},
                                    {0.0, 0.0, 1.0} }).to(torch::kFloat32);
    if (args["render_test"].as<bool>()) {
        std::vector<torch::Tensor> test;
        for (auto i : *i_test) {
            test.emplace_back(blender_data.m_poses->index({ i }));
        }
        blender_data.m_render_poses = std::make_shared<torch::Tensor>(torch::stack(test, /*dim=*/0));
    }

    // Create log dirand copy the config file
    boost::filesystem::path base_dir = boost::filesystem::path(args["basedir"].as<std::string>());
    std::string exp_name = args["expname"].as<std::string>();
    boost::filesystem::path log_dir = base_dir / exp_name;
    if (!boost::filesystem::exists(log_dir)) {
        int status = _mkdir(log_dir.string().c_str());
        if (status == 0) {
            std::cout << "Directory created successfully." << std::endl;
        }
        else {
            std::cout << "Failed to create directory." << std::endl;
        }
    }
    //std::string file_name = "args.txt";
    boost::filesystem::path file_dir = log_dir / "args.txt";
    try {
        // Write the options to a text file
        std::ofstream outfile(file_dir.string().c_str());
        for (auto& option : args) {
            outfile << option.first << " = ";
            boost::any value = option.second.value();
            if (value.type() == typeid(std::string)) {
                outfile << boost::any_cast<std::string>(value);
            }
            else if (value.type() == typeid(std::int64_t)) {
                outfile << boost::any_cast<std::int64_t>(value);
            }
            else if (value.type() == typeid(float)) {
                outfile << boost::any_cast<float>(value);
            }
            else if (value.type() == typeid(bool)) {
                outfile << boost::any_cast<bool>(value);
            }
            // Add more type checks as needed for other types of options
            outfile << std::endl;
        }
        outfile.close();

        std::cout << "Options written to options.txt" << std::endl;
    }
    catch (std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    if (args["config"].as<std::string>() != "") {
        std::string config_dir = args["config"].as<std::string>();
        std::cout << "String option: " << config_dir << std::endl;
        std::ifstream ifile(config_dir.c_str());
        if (!ifile) {
            std::cerr << "Failed to open file " << config_dir << std::endl;
            return 1;
        }
        file_dir = log_dir / "config.txt";
        std::ofstream ofile(file_dir.c_str());
        if (!ofile) {
            std::cerr << "Failed to open file " << file_dir << std::endl;
            return 1;
        }
        std::string line;
        while (std::getline(ifile, line)) {
            ofile << line << std::endl;
        }

        ifile.close();
        ofile.close();
    }

    /// <summary>
    ////////////////////////////////////////////////////////////// Create nerf model
    /// </summary>
    NeRF nerf;
    std::int64_t start = nerf.create(args);
    std::int64_t global_step = start;
    nerf.render_kwars_train.near = near;
    nerf.render_kwars_test.near = near;
    nerf.render_kwars_train.far = far;
    nerf.render_kwars_test.far = far;

    // Move testing data to GPU
    blender_data.m_render_poses = std::make_shared<torch::Tensor>(blender_data.m_render_poses->to(torch::kCUDA));

    // Short circuit if only rendering out from trained model
    if (args["render_only"].as<bool>()) {
        std::cout << "RENFER ONLY" << std::endl;
        torch::NoGradGuard no_grad;
        if (args["render_test"].as<bool>()) {
            // render_test switches to test poses
            c10::IntArrayRef arr(i_test->data(), i_test->size());
            torch::Tensor images_t = blender_data.m_imgs->index({ torch::tensor(arr) });
            blender_data.m_imgs = std::make_shared<torch::Tensor>(images_t);
        }
        else {
            // Default is smoother render_poses path
            blender_data.m_imgs = nullptr;
        }

        boost::filesystem::path testsavedir = log_dir;
        if (args["render_test"].as<bool>()) {
            testsavedir = testsavedir / std::string("renderonly_test_" + std::string(6 - std::to_string(start).length(), '0') + std::to_string(start));
        }
        else {
            testsavedir = testsavedir / std::string("renderonly_path_" + std::string(6 - std::to_string(start).length(), '0') + std::to_string(start));
        }
        if (!boost::filesystem::exists(testsavedir)) {
            int status = _mkdir(testsavedir.string().c_str());
            if (status == 0) {
                std::cout << "Directory created successfully." << std::endl;
            }
            else {
                std::cout << "Failed to create directory." << std::endl;
            }
        }
        std::cout << "test poses shape" << std::endl;

        // render_path();
        torch::Tensor rgbs, disps;
        nerf.render_path(*blender_data.m_render_poses, blender_data.m_hwf.H, blender_data.m_hwf.W, blender_data.m_hwf.focal, K, args["chunk"].as<std::int64_t>(),
            blender_data.m_imgs, testsavedir.string(), args["render_factor"].as<std::int64_t>(), nerf.render_kwars_test,
            rgbs, disps);
        std::cout << "Done rendering" << testsavedir.string() << std::endl;

        auto bgr8 = rgbs.mul(255).clip(0, 255).to(torch::kUInt8).to(torch::kCPU);
        cv::VideoWriter writer;
        writer.open(testsavedir.string() + "video.mp4", cv::VideoWriter::fourcc('X', '2', '6', '4'), 30, cv::Size(640, 480), true);
        if (!writer.isOpened()) {
            std::cout << "Failed to init a video" << std::endl;
            return -1;
        }
        for (size_t i = 0; i < bgr8.sizes().at(0); i++) {
            auto bgr = bgr8[i];
            cv::Mat image{ W, H, CV_8UC3, bgr.data_ptr() };
            //image.create(bgr.sizes().at(0), bgr.sizes().at(1), CV_32F);
            //memcpy(image.data, bgr.data_ptr(), sizeof(float) * bgr.numel());
            cv::Mat rgb8;
            cv::cvtColor(image, rgb8, cv::COLOR_BGR2RGB);
            cv::resize(rgb8, rgb8, cv::Size(640, 480));
            //writer.write(rgb8);
            writer << rgb8;
        }
        writer.release();
        return 0;
    }
    // Prepare raybatch tensor if batching random rays
    std::int64_t N_rand = args["N_rand"].as<std::int64_t>();
    bool use_batching = !args["no_batching"].as<bool>();
    torch::Tensor rays_rgb;
    int i_batch;
    if (use_batching) {
        // For random ray batching
        std::cout << "get rays" << std::endl;
        auto ps = blender_data.m_poses->index({ torch::indexing::Slice(), torch::indexing::Slice(torch::indexing::None, 3), torch::indexing::Slice(torch::indexing::None, 4) });
        std::vector<torch::Tensor> ray_vec;
        for (size_t i = 0; i < ps.sizes().at(0); i++) {
            auto p = ps[i];
            torch::Tensor rays_o, rays_d;
            get_rays(H, W, K, p, rays_o, rays_d, p.device());
            ray_vec.emplace_back(torch::stack({ rays_o, rays_d }, 0));
        }
        auto rays = torch::stack(ray_vec, 0);   // Nx2xHxWx3
        std::cout << "done, concats" << std::endl;
        rays_rgb = torch::concat({ rays, blender_data.m_imgs->index({torch::indexing::Slice(), torch::indexing::None}) }, 1);   // Nx3xHxWx3
        rays_rgb = torch::permute(rays_rgb, { 0, 2, 3, 1, 4 }); // NxHxWx3x3
        rays_rgb = rays_rgb.index({ torch::tensor(*i_train) });
        rays_rgb = torch::reshape(rays_rgb, { -1, 3, 3 }).to(torch::kFloat32);
        std::cout << "shuffle rays" << std::endl;
        auto rand_indx = torch::randperm(rays_rgb.sizes().at(0));
        rays_rgb = rays_rgb[rand_indx];

        std::cout << "done" << std::endl;
        i_batch = 0;
    }

    // move training data to GPU
    if (use_batching) {
        blender_data.m_imgs = std::make_shared<torch::Tensor>(blender_data.m_imgs->to(torch::kCUDA));
    }
    blender_data.m_poses = std::make_shared<torch::Tensor>(blender_data.m_poses->to(torch::kCUDA));
    if (use_batching) {
        rays_rgb = rays_rgb.to(torch::kCUDA);
    }

    std::int64_t N_iters = 200000 + 1;
    std::cout << "Begin" << std::endl;
    std::cout << "TRAIN views are" << *i_train << std::endl;
    std::cout << "TEST views are" << *i_test << std::endl;
    std::cout << "VAL views are" << *i_val << std::endl;

    start = start + 1;
    for (size_t i = start; i < N_iters; i++) {
        // Sample random ray batch
        torch::Tensor batch_rays;
        torch::Tensor target_s;
        if (use_batching) {
            // Random over all images
            auto batch = rays_rgb.index({ torch::indexing::Slice(i_batch, i_batch + N_rand) });
            batch = torch::transpose(batch, 0, 1);
            auto batch_rays = batch.index({ torch::indexing::Slice(torch::indexing::None, 2) });
            auto target_s = batch[2];

            i_batch += N_rand;
            if (i_batch >= rays_rgb.sizes().at(0)) {
                std::cout << "Shuffle data after an epoch!" << std::endl;
                auto rand_idx = torch::randperm(rays_rgb.sizes().at(0));
                rays_rgb = rays_rgb[rand_idx];
                i_batch = 0;
            }
        }
        else {
            // Random from one image
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> dis(0, i_train->size() - 1);
            std::int64_t random_index = dis(gen);
            std::int64_t img_i = i_train->at(random_index);
            torch::Tensor target = blender_data.m_imgs->index({ img_i }).to(torch::kCUDA);
            torch::Tensor pose = blender_data.m_poses->index({ img_i, torch::indexing::Slice(None, 3), torch::indexing::Slice(None, 4) }).to(torch::kCUDA);
            if (N_rand > 0) {
                torch::Tensor rays_o;
                torch::Tensor rays_d;
                get_rays(H, W, K, pose, rays_o, rays_d, pose.device());    // (H, W, 3), (H, W, 3)

                torch::Tensor coords;
                if (i < args["precrop_iters"].as<std::int64_t>()) {
                    float precrop_frac = args["precrop_frac"].as<float>();
                    int dH = int(H / 2 * precrop_frac);
                    int dW = int(W / 2 * precrop_frac);
                    coords = torch::stack(torch::meshgrid({ torch::linspace(H / 2 - dH, H / 2 + dH - 1,2 * dH), torch::linspace(W / 2 - dW, W / 2 + dW - 1,2 * dW) }), -1);
                    if (i == start) {
                        std::cout << "[Config] Center cropping of size " << std::to_string(2 * dH) << " x " << std::to_string(2 * dW) << " is enabled until iter " << std::to_string(precrop_frac) << std::endl;
                    }
                }
                else {
                    coords = torch::stack(torch::meshgrid({ torch::linspace(0, H - 1,H), torch::linspace(0, W - 1,W) }), -1);   // (H, W, 2)
                }

                coords = torch::reshape(coords, { -1, 2 }); // (H * W, 2)

                std::vector<std::int64_t> vec(coords.sizes().at(0));
                std::iota(vec.begin(), vec.end(), 0);   // range(0, N)
                std::vector<std::int64_t> out;
                size_t nelems = N_rand;
                std::sample(
                    vec.begin(),
                    vec.end(),
                    std::back_inserter(out),
                    nelems,
                    std::mt19937{ std::random_device{}() }
                );
                //std::cout << out << std::endl;
                c10::IntArrayRef select_inds(out.data(), out.size());
                torch::Tensor select_coords = coords.index({ torch::tensor(select_inds) }).to(torch::kInt64);
                rays_o = rays_o.index({ select_coords.index({torch::indexing::Slice(), 0}), select_coords.index({torch::indexing::Slice(), 1}) });
                rays_d = rays_d.index({ select_coords.index({torch::indexing::Slice(), 0}), select_coords.index({torch::indexing::Slice(), 1}) });
                batch_rays = torch::stack({ rays_o, rays_d }, 0);
                target_s = target.index({ select_coords.index({torch::indexing::Slice(), 0}), select_coords.index({torch::indexing::Slice(), 1}) });
            }

        }
        ////  Core optimization loop  ////
        torch::Tensor rgb, disp, acc;
        std::unordered_map<std::string, torch::Tensor> extras;
        nerf.render(H, W, K, args["chunk"].as<std::int64_t>(), std::make_shared<torch::Tensor>(batch_rays), 
            /*c2w*/nullptr, /*c2w_staticcam*/nullptr, /*retraw*/true, nerf.render_kwars_train,
            rgb, disp, acc, extras);

        nerf.m_optimizer->zero_grad();
        //img2mse(rgb, target_s)
        auto img_loss = torch::mean(torch::pow(rgb - target_s, 2));
        auto trans = extras["raw"].index({ "...", -1 });
        auto loss = img_loss;
        auto psnr = -10.0 * torch::log(img_loss) / torch::log(torch::tensor({ 10.0 }).to(torch::kCUDA));

        if (extras.find("rgb0") != extras.end()) {
            auto img_loss0 = torch::mean(torch::pow(extras["rgb0"] - target_s, 2));
            loss = loss + img_loss0;
            auto psnr0 = -10.0 * torch::log(img_loss0) / torch::log(torch::tensor({ 10.0 }).to(torch::kCUDA));
        }

        loss.backward();
        nerf.m_optimizer->step();

        // NOTE: IMPORTANT!
        //////   update learning rate   //////
        float decay_rate = 0.1;
        std::int64_t decay_steps = args["lrate_decay"].as<std::int64_t>();
        float new_lrate = args["lrate"].as<float>() * (std::pow(decay_rate, global_step / decay_steps));
        for (auto& param_group : nerf.m_optimizer->param_groups()) {
            param_group.options().set_lr(new_lrate);
        }

        //  Rest is logging
        if (i % args["i_weights"].as<std::int64_t>() == 0) {
            std::ostringstream ss;
            ss << std::setw(6) << std::setfill('0') << i;
            std::string filename = ss.str() + ".pt";
            auto model_path = log_dir / filename;
            auto model_fine_path = log_dir / "MF" / filename;
            if (!boost::filesystem::exists(log_dir / "MF")) {
                auto dir = log_dir / "MF";
                int status = _mkdir(dir.string().c_str());
                if (status == 0) {
                    std::cout << "Directory created successfully." << std::endl;
                }
                else {
                    std::cout << "Failed to create directory." << std::endl;
                }
            }
            auto optim_path = log_dir / "OPTIM" / filename;
            if (!boost::filesystem::exists(log_dir / "OPTIM")) {
                auto dir = log_dir / "OPTIM";
                int status = _mkdir(dir.string().c_str());
                if (status == 0) {
                    std::cout << "Directory created successfully." << std::endl;
                }
                else {
                    std::cout << "Failed to create directory." << std::endl;
                }
            }

            torch::serialize::OutputArchive m_archive;
            //nerf.model->to(torch::kCPU);
            nerf.model->save(m_archive);
            m_archive.write(/*name=*/"global_step", /*value=*/global_step);
            //torch::save(archive);
            m_archive.save_to(model_path.string());
            
            torch::serialize::OutputArchive mf_archive;
            //nerf.model_fine->to(torch::kCPU);
            nerf.model_fine->save(mf_archive);
            mf_archive.save_to(model_fine_path.string());

            torch::save(*nerf.m_optimizer, optim_path.string());
        }
        if (i % args["i_video"].as<std::int64_t>() == 0 && i > 0) {
            // Turn on testing mode
            torch::NoGradGuard no_grad;
            torch::Tensor rgbs, disps;
            //blender_data.m_render_poses->to(torch::kCUDA);
            nerf.render_path(*blender_data.m_render_poses, blender_data.m_hwf.H, blender_data.m_hwf.W, blender_data.m_hwf.focal, K, args["chunk"].as<std::int64_t>(), 
                nullptr, "", 0, nerf.render_kwars_test, 
                rgbs, disps);
            std::cout << "Done, sacing" << rgbs.sizes() << disps.sizes() << std::endl;

            std::ostringstream ss;
            ss << std::setw(6) << std::setfill('0') << i;
            auto moviebase = log_dir / (exp_name + "_spital_" + ss.str() + "_");
            auto bgr8 = rgbs.mul(255).clamp(0, 255).to(torch::kU8);
            bgr8 = bgr8.to(torch::kCPU);
            cv::VideoWriter writer_rgb;
            writer_rgb.open(moviebase.string() + "rgb.mp4", cv::VideoWriter::fourcc('X', '2', '6', '4'), 30, cv::Size(640, 480), true);
            if (!writer_rgb.isOpened()) {
                std::cout << "Failed to init a video" << std::endl;
                return -1;
            }
            for (size_t i = 0; i < bgr8.sizes().at(0); i++) {
                auto bgr = bgr8[i]; 
                cv::Mat image(cv::Size{ H, W }, CV_8UC3, bgr.data_ptr());
                cv::Mat rgb8;
                cv::cvtColor(image, rgb8, cv::COLOR_BGR2RGB);
                cv::resize(rgb8, rgb8, cv::Size(640, 480));
                writer_rgb.write(rgb8);
            }
            writer_rgb.release();

            bgr8 = torch::clip((disps / torch::max(disps)), 0, 1).mul(255).to(torch::kUInt8).to(torch::kCPU);
            cv::VideoWriter writer_disp;
            writer_disp.open(moviebase.string() + "disp.mp4", cv::VideoWriter::fourcc('X', '2', '6', '4'), 30, cv::Size(640, 480), true);
            if (!writer_disp.isOpened()) {
                std::cout << "Failed to init a video" << std::endl;
                return -1;
            }
            for (size_t i = 0; i < bgr8.sizes().at(0); i++) {
                auto bgr = bgr8[i];
                cv::Mat image(cv::Size{ H, W }, CV_8UC3, bgr.data_ptr());
                cv::Mat rgb8;
                cv::cvtColor(image, rgb8, cv::COLOR_BGR2RGB);
                cv::resize(rgb8, rgb8, cv::Size(640, 480));
                writer_disp.write(rgb8);
            }
            writer_disp.release();
        }
        if (i % args["i_testset"].as<std::int64_t>() == 0 && i > 0) {
            std::ostringstream ss;
            ss << std::setw(6) << std::setfill('0') << i;
            auto testsavedir = log_dir / ("testset_" + ss.str());
            if (!boost::filesystem::exists(testsavedir)) {
                int status = _mkdir(testsavedir.string().c_str());
                if (status == 0) {
                    std::cout << "Directory created successfully." << std::endl;
                }
                else {
                    std::cout << "Failed to create directory." << std::endl;
                }
            }
            torch::NoGradGuard no_grad;
            auto poses_t = blender_data.m_poses->index({ torch::tensor(*i_test) });
            auto images = blender_data.m_imgs->index({ torch::tensor(*i_test) });
            std::cout << "test poses shape" << poses_t.sizes() << std::endl;
            torch::Tensor rgbs, disps;
            nerf.render_path(poses_t, blender_data.m_hwf.H, blender_data.m_hwf.W, blender_data.m_hwf.focal, K, args["chunk"].as<std::int64_t>(), 
                std::make_shared<torch::Tensor>(images), testsavedir.string(), 0, nerf.render_kwars_test,
                rgbs, disps);
            std::cout << "Saved teste set" << std::endl;
        }
        if (i % args["i_print"].as<std::int64_t>() == 0) {
            std::cout << "[TRAIN] Iter: " << i << " Loss: " << loss.item() << " PSNR: " << psnr.item() << std::endl;
        }

        global_step += 1;
    }
    return 0;
}