#include <version.hpp>
using namespace pygmalion;

const std::string pygmalion::VERSION = "1.0.0";

extern "C"
{
    const char* get_version()
    {
        return pygmalion::VERSION.c_str();
    }
}
