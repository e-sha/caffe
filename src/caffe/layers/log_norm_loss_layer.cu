#include <vector>

#include "caffe/layers/log_norm_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void LogNormLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  Blob<Dtype> tmp_diff;
  Blob<Dtype> tmp_log_cov;
  tmp_diff.ReshapeLike(*bottom[0]);
  tmp_log_cov.ReshapeLike(*bottom[0]);
  // difference
  // @f$ \hat{y} - y @f$ 
  caffe_gpu_sub(
      count,
      bottom[0]->gpu_data(),
      bottom[2]->gpu_data(),
      tmp_diff.mutable_gpu_data());
  // @f$ \exp{\hat{\sigma}} @f$
  caffe_gpu_exp(
      count,
      bottom[1]->gpu_data(),
      sigma_.mutable_gpu_data());
  // covariance
  // @f$ \exp{\hat{\sigma}} + \eps @f$
  caffe_gpu_add_scalar(
      count,
      eps_,
      sigma_.mutable_gpu_data());
  // weighted difference
  // @f$ \frac{\hat{y} - y}{\exp^{\hat{\sigma}} + \eps} @f$
  caffe_gpu_div(
      count,
      tmp_diff.gpu_data(),
      sigma_.gpu_data(),
      diff_.mutable_gpu_data());
  // weighted derivation
  // @f$ \frac{\left( \hat{y} - y \right)^2}{\exp^{\hat{\sigma}} + \eps} @f$
  caffe_gpu_mul(
      count,
      tmp_diff.gpu_data(),
      diff_.gpu_data(),
      sqr_diff_.mutable_gpu_data());

  // logarithm of covariance
  caffe_gpu_log(count,
      sigma_.gpu_data(),
      tmp_log_cov.mutable_gpu_data());
  // weighted derivation + logarithm of covariance
  caffe_gpu_add(count,
      tmp_log_cov.gpu_data(),
      sqr_diff_.gpu_data(),
      tmp_diff.mutable_gpu_data());
  // sum all observations
  caffe_gpu_set(count, Dtype(1), tmp_log_cov.mutable_gpu_data());
  Dtype res;
  caffe_gpu_dot(count, tmp_log_cov.gpu_data(), tmp_diff.gpu_data(), &res);

  Dtype loss = (res / bottom[0]->num() + log(2 * M_PI)) / 2;
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void LogNormLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      if (i == 1) {
        Blob<Dtype> eps_factor, sigma_factor;
        eps_factor.ReshapeLike(*bottom[1]);
        sigma_factor.ReshapeLike(*bottom[1]);
        int count = bottom[i]->count();
        
        // @f$ \exp^{\hat{y}}  @f$
        caffe_gpu_memcpy(count, 
            sigma_.gpu_data(),
            sigma_factor.mutable_gpu_data());
        caffe_gpu_add_scalar(count,
            -eps_,
            sigma_factor.mutable_gpu_data());

        // @f$ \frac{\exp^{\hat{y}}}{\exp^{\hat{y}} + \eps} @f$
        caffe_gpu_div(count,
            sigma_factor.gpu_data(),
            sigma_.gpu_data(),
            eps_factor.mutable_gpu_data());

        // @f$ \frac{\alpha}{2} \left(1 - \frac{\left( \hat{y} - y \right)^2}
        // {\exp{\hat{\sigma}} + \eps}\right) @f$
        const Dtype alpha = top[0]->cpu_diff()[0] / bottom[i]->num() / 2;
        caffe_gpu_set(count, alpha, sigma_factor.mutable_gpu_data());
        caffe_gpu_axpby(count,
            -alpha,
            sqr_diff_.gpu_data(),
            Dtype(1),
            sigma_factor.mutable_gpu_data());

        caffe_gpu_mul(count,
            eps_factor.gpu_data(),
            sigma_factor.gpu_data(),
            bottom[1]->mutable_gpu_diff());
      }
      else {
        const Dtype sign = (i == 0) ? 1 : -1;
        const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
        caffe_gpu_axpby(
            bottom[i]->count(),              // count
            alpha,                              // alpha
            diff_.gpu_data(),                   // a
            Dtype(0),                           // beta
            bottom[i]->mutable_gpu_diff());  // b
      }
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(LogNormLossLayer);

}  // namespace caffe
