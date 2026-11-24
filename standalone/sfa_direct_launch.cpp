// CPU-only preflight. The actual ASC device runner is sfa_direct_launch.asc.
#include "sfa_fixed_contract.h"
#include <iostream>
int main(int argc, char**)
{
    if (argc != 1) {
        std::cerr << "This executable only checks the host contract; use sfa_launch for device execution.\n";
        return 2;
    }
    const auto tiling = sfa_fixed::MakeTiling();
    if (!sfa_fixed::ValidateTiling(tiling)) return 1;
    sfa_fixed::WritePreflight(std::cout);
    return 0;
}
