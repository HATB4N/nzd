#include "Common/Gemm.h"

Matrix& gemm() {
    static std::unique_ptr<Matrix> gemm_instance = std::make_unique<Matrix>();
    return *gemm_instance;
}
