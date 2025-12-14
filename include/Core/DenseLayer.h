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
    void backward(Matrix_T<fp32>& dR, Matrix_T<fp32>& dX) override;
    uint64_t get_in_dim() const override { return this->_input_dim; }
    uint64_t get_out_dim() const override { return this->_output_dim; }

    Matrix_T<fp32>& get_weight() { return this->_weights; }
    Matrix_T<fp32>& get_bias() { return this->_biases; }
    Matrix_T<fp32>& get_grad_weight() { return this->_grad_weights; }
    Matrix_T<fp32>& get_grad_bias() { return this->_grad_biases; }
    void set_weight(const Matrix_T<fp32>& new_weights) { this->_weights = new_weights; }
    void set_bias(const Matrix_T<fp32>& new_biases) { this->_biases = new_biases; }
    
    ActType get_act_type() { return this->_act_type; }
    void update();
    
private:
    const uint64_t _idx;
    uint64_t _input_dim;
    uint64_t _output_dim;

    Matrix_T<fp32> _weights;
    Matrix_T<fp32> _biases;

    Matrix_T<fp32> _grad_weights;
    Matrix_T<fp32> _grad_biases;

    Matrix_T<fp32> _x_cache;
    Matrix_T<fp32> _z_cache;

    ActType _act_type;
    ActFunc _act;
    ActFunc _act_difr;
    InitFunc _initializer;
    OptFunc _optimizer;

    // runtime binding for backward one-hot-encoding exception
    using BwFunc = void (DenseLayer::*)(Matrix_T<fp32>&, Matrix_T<fp32>&);
    void _bw_impl_standard(Matrix_T<fp32>& dR, Matrix_T<fp32>& dX);
    void _bw_impl_bypass(Matrix_T<fp32>& dZ, Matrix_T<fp32>& dX);
    void _compute_gradients(Matrix_T<fp32>& dZ, Matrix_T<fp32>& dX);
    void _accum_bias_grad(const Matrix_T<fp32>& dZ);
    static constexpr BwFunc _bw_table[] = {
        &DenseLayer::_bw_impl_standard,
        &DenseLayer::_bw_impl_bypass
    };
    BwFunc _runner;
};

#endif // DENSELAYER_H
