#pragma once
#include <string>

namespace pygmalion
{
    // String representing the version of the library
    extern const std::string VERSION;

    // python bindings
    extern "C"
    {
        // Returns the version of the library
        const char* get_version();
    }
}
