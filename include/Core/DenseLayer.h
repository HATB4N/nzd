#ifndef DENSELAYER_H
#define DENSELAYER_H

#include "Core/ILayer.h"
#include "Utils/Activation.h"
#include "Common/Struct.h"
#include "Common/Types.h"
#include "Initializers/Initializer.h"
#include "Optimizers/Optimizer.h"
#include <memory>
#include <cstdint>

class DenseLayer : public ILayer {
public:
    DenseLayer(uint64_t input_dim,
               uint64_t output_dim,
               ActType act_type,
               InitType init,
               OptType opt,
               uint64_t idx);

    ~DenseLayer() = default;

    // ILayer interface implementation
    void forward(const Matrix_T<fp32> &X, Matrix_T<fp32> &Z) override;
    void backward(Matrix_T<fp32> &dR, Matrix_T<fp32> &dX) override;
    void update();

    uint64_t get_in_dim() const override { return this->input_dim_; }
    uint64_t get_out_dim() const override { return this->output_dim_; }

    Matrix_T<fp32> &get_weight() { return this->weights_; }
    Matrix_T<fp32> &get_bias() { return this->biases_; }

    Matrix_T<fp32> &get_grad_weight() { return this->grad_weights_; }
    Matrix_T<fp32> &get_grad_bias() { return this->grad_biases_; }

    void set_weight(const Matrix_T<fp32> &new_weights) { this->weights_ = new_weights; }
    void set_bias(const Matrix_T<fp32> &new_biases) { this->biases_ = new_biases; }

    ActType get_act_type() { return this->act_type_; }

private:
    const uint64_t idx_;
    uint64_t input_dim_;
    uint64_t output_dim_;

    Matrix_T<fp32> weights_;
    Matrix_T<fp32> biases_;

    Matrix_T<fp32> grad_weights_;
    Matrix_T<fp32> grad_biases_;

    Matrix_T<fp32> x_cache_;
    Matrix_T<fp32> z_cache_;

    ActType act_type_;
    ActFunc act_;
    ActFunc act_difr_;
    InitFunc initializer_;
    OptFunc optimizer_;

    // runtime binding for backward one-hot-encoding exception
    using BwFunc = void (DenseLayer::*)(Matrix_T<fp32> &, Matrix_T<fp32> &);
    void bw_impl_standard_(Matrix_T<fp32> &dR, Matrix_T<fp32> &dX);
    void bw_impl_bypass_(Matrix_T<fp32> &dZ, Matrix_T<fp32> &dX);
    void compute_gradients_(Matrix_T<fp32> &dZ, Matrix_T<fp32> &dX);
    void accum_bias_grad_(const Matrix_T<fp32> &dZ);
    static constexpr BwFunc bw_table_[] = {
        &DenseLayer::bw_impl_standard_,
        &DenseLayer::bw_impl_bypass_
    };
    BwFunc runner_;
};

#endif // DENSELAYER_H
