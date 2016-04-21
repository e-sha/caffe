#include <vector>

#include "caffe/layers/log_norm_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void LogNormLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
      << "First and second inputs must have the same dimension.";
  CHECK_EQ(bottom[0]->count(1), bottom[2]->count(1))
      << "First and third inputs must have the same dimension.";
  diff_.ReshapeLike(*bottom[0]);
  sqr_diff_.ReshapeLike(*bottom[0]);
  sigma_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void LogNormLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  Blob<Dtype> tmp_diff;
  Blob<Dtype> tmp_log_cov;
  tmp_diff.ReshapeLike(*bottom[0]);
  tmp_log_cov.ReshapeLike(*bottom[0]);
  // difference
  // @f$ \hat{y} - y @f$ 
  caffe_sub(
      count,
      bottom[0]->cpu_data(),
      bottom[2]->cpu_data(),
      tmp_diff.mutable_cpu_data());
  // @f$ e^{\hat{\sigma}} @f$
  caffe_exp(
      count,
      bottom[1]->cpu_data(),
      sigma_.mutable_cpu_data());
  // covariance
  // @f$ e^{\hat{\sigma}} + \epsilon @f$
  caffe_add_scalar(
      count,
      eps_,
      sigma_.mutable_cpu_data());
  // weighted difference
  // @f$ \frac{\hat{y} - y}{e^{\hat{\sigma}} + \epsilon} @f$
  caffe_div(
      count,
      tmp_diff.cpu_data(),
      sigma_.cpu_data(),
      diff_.mutable_cpu_data());
  // weighted derivation
  // @f$ \frac{\left( \hat{y} - y \right)^2}{e^{\hat{\sigma}} + \epsilon} @f$
  caffe_mul(
      count,
      tmp_diff.cpu_data(),
      diff_.cpu_data(),
      sqr_diff_.mutable_cpu_data());

  // logarithm of covariance
  caffe_log(count,
      sigma_.cpu_data(),
      tmp_log_cov.mutable_cpu_data());
  // weighted derivation + logarithm of covariance
  caffe_add(count,
      tmp_log_cov.cpu_data(),
      sqr_diff_.cpu_data(),
      tmp_diff.mutable_cpu_data());
  // sum all observations
  caffe_set(count, Dtype(1), tmp_log_cov.mutable_cpu_data());
  Dtype res = caffe_cpu_dot(count, tmp_log_cov.cpu_data(), tmp_diff.cpu_data());

  Dtype loss = (res / bottom[0]->num() + log(2 * M_PI)) / 2;
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void LogNormLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < 3; ++i) {
    if (propagate_down[i]) {
      if (i == 1) {
        Blob<Dtype> eps_factor, sigma_factor;
        eps_factor.ReshapeLike(*bottom[1]);
        sigma_factor.ReshapeLike(*bottom[1]);
        int count = bottom[i]->count();
        
        // @f$ e^{\hat{y}}  @f$
        caffe_copy(count, 
            sigma_.cpu_data(),
            sigma_factor.mutable_cpu_data());
        caffe_add_scalar(count,
            -eps_,
            sigma_factor.mutable_cpu_data());

        // @f$ \frac{e^{\hat{y}}}{e^{\hat{y}} + \epsilon} @f$
        caffe_div(count,
            sigma_factor.cpu_data(),
            sigma_.cpu_data(),
            eps_factor.mutable_cpu_data());

        // @f$ \frac{\alpha}{2} \left(1 - \frac{\left( \hat{y} - y \right)^2}
        // {e^{\hat{\sigma}} + \epsilon}\right) @f$
        const Dtype alpha = top[0]->cpu_diff()[0] / bottom[i]->num() / 2;
        caffe_set(count, alpha, sigma_factor.mutable_cpu_data());
        caffe_cpu_axpby(count,
            -alpha,
            sqr_diff_.cpu_data(),
            Dtype(1),
            sigma_factor.mutable_cpu_data());

        caffe_mul(count,
            eps_factor.cpu_data(),
            sigma_factor.cpu_data(),
            bottom[i]->mutable_cpu_diff());
      }
      else {
        const Dtype sign = (i == 0) ? 1 : -1;
        const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
        caffe_cpu_axpby(
            bottom[i]->count(),              // count
            alpha,                              // alpha
            diff_.cpu_data(),                   // a
            Dtype(0),                           // beta
            bottom[i]->mutable_cpu_diff());  // b
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(LogNormLossLayer);
#endif

INSTANTIATE_CLASS(LogNormLossLayer);
REGISTER_LAYER_CLASS(LogNormLoss);

}  // namespace caffe
