#include "configs.h"
#include <torch/torch.h>
#include <iostream>

ConfigBuilder::ConfigBuilder()
{
    po::options_description desc("Allowed options");
    desc.add_options()
        ("config", po::value<std::string>()->default_value(""), "config file path")
        ("expname", po::value<std::string>()->default_value(""), "experiment name")
        ("basedir", po::value<std::string>()->default_value("./logs/"), "where to store ckpts and logs")
        ("datadir", po::value<std::string>()->default_value("./data/llff/fern"), "input data directory")
        // training options
        ("netdepth", po::value<std::int64_t>()->default_value(8), "layers in network")
        ("netwidth", po::value<std::int64_t>()->default_value(256), "channels per layer")
        ("netdepth_fine", po::value<std::int64_t>()->default_value(8), "layers in fine network")
        ("netwidth_fine", po::value<std::int64_t>()->default_value(256), "channels per layer in fine network")

        ("N_rand", po::value<std::int64_t>()->default_value(32 * 32 * 4), "batch size (number of random rays per gradient step)")
        ("lrate", po::value<float>()->default_value(5e-4), "learning rate")
        ("lrate_decay", po::value<std::int64_t>()->default_value(250), "exponential learning rate decay (in 1000 steps)")
        ("chunk", po::value<std::int64_t>()->default_value(1024 * 32), "number of rays processed in parallel, decrease if running out of memory")
        ("netchunk", po::value<std::int64_t>()->default_value(1024 * 64), "number of pts sent through network in parallel, decrease if running out of memory")
        ("no_batching", po::value<bool>()->default_value(false), "only take random rays from 1 image at a time")
        ("no_reload", po::value<bool>()->default_value(false), "do not reload weights from saved ckpt")
        ("ft_path", po::value<std::string>()->default_value(""), "specific weights npy file to reload for coarse network")
        // rendering options
        ("N_samples", po::value<std::int64_t>()->default_value(64), "number of coarse samples per ray")
        ("N_importance", po::value<std::int64_t>()->default_value(0), "number of additional fine samples per ray")
        ("perturb", po::value<float>()->default_value(1.0), "set to 0. for no jitter, 1. for jitter")
        ("use_viewdirs", po::value<bool>()->default_value(false), "use full 5D input instead of 3D")
        ("i_embed", po::value<std::int64_t>()->default_value(0), "set 0 for default positional encoding, -1 for none")
        ("multires", po::value<std::int64_t>()->default_value(10), "log2 of max freq for positional encoding (3D location)")
        ("multires_views", po::value<std::int64_t>()->default_value(4), "log2 of max freq for positional encoding (2D direction)")
        ("raw_noise_std", po::value<float>()->default_value(0.), "std dev of noise added to regularize sigma_a output, 1e0 recommended")

        ("render_only", po::value<bool>()->default_value(false), "do not optimize, reload weights and render out render_poses path")
        ("render_test", po::value<bool>()->default_value(false), "render the test set instead of render_poses path")
        ("render_factor", po::value<std::int64_t>()->default_value(0), "downsampling factor to speed up rendering, set 4 or 8 for fast preview")
        // training options
        ("precrop_iters", po::value<std::int64_t>()->default_value(0), "number of steps to train on central crops")
        ("precrop_frac", po::value<float>()->default_value(0.5), "fraction of img taken for central crops")
        // dataset options
        ("dataset_type", po::value<std::string>()->default_value("llff"), "options: llff / blender / deepvoxels")
        ("testskip", po::value<std::int64_t>()->default_value(8), "will load 1/N images from test/val sets, useful for large datasets like deepvoxels")
        // deepvoxels flags
        ("shape", po::value<std::string>()->default_value("greek"), "options : armchair / cube / greek / vase")
        // blender flags
        ("white_bkgd", po::value<bool>()->default_value(false), "set to render synthetic data on a white bkgd (always use for dvoxels)")
        ("half_res", po::value<bool>()->default_value(false), "load blender synthetic data at 400x400 instead of 800x800")
        // llff flags
        ("factor", po::value<std::int64_t>()->default_value(8), "downsample factor for LLFF images")
        ("no_ndc", po::value<bool>()->default_value(false), "do not use normalized device coordinates (set for non-forward facing scenes)")
        ("lindisp", po::value<bool>()->default_value(false), "sampling linearly in disparity rather than depth")
        ("spherify", po::value<bool>()->default_value(false), "set for spherical 360 scenes")
        ("llffhold", po::value<std::int64_t>()->default_value(8), "will take every 1/N images as LLFF test set, paper uses 8")
        // logging/saving options
        ("i_print", po::value<std::int64_t>()->default_value(100), "frequency of console printout and metric loggin")
        ("i_img", po::value<std::int64_t>()->default_value(500), "frequency of tensorboard image logging")
        ("i_weights", po::value<std::int64_t>()->default_value(10000), "frequency of weight ckpt saving")
        ("i_testset", po::value<std::int64_t>()->default_value(50000), "frequency of testset saving")
        ("i_video", po::value<std::int64_t>()->default_value(50000), "frequency of render_poses video saving")
        ;

    this->desc = std::make_shared<po::options_description>(desc);
}


bool ConfigBuilder::fromfile(std::string fn, po::variables_map& args)
{
    std::ifstream infile(fn);
    if (!infile) {
        std::cerr << "Failed to open file configs/xxxx.txt" << std::endl;
        return false;
    }
    po::store(po::parse_config_file(infile, *(this->desc)), args);
    po::notify(args);
    return true;
}
