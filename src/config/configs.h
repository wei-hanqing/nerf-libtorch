#include <boost/program_options.hpp>

namespace po = boost::program_options;

class ConfigBuilder {
public:
    ConfigBuilder();
    ~ConfigBuilder() {};

    bool fromfile(std::string fn, po::variables_map& args);
    
private:
    std::shared_ptr<po::options_description> desc;
};