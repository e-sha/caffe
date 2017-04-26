#ifndef CAFFE_LOG_NORM_LOSS_LAYER_HPP_
#define CAFFE_LOG_NORM_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

/**
 * @brief Computes the Negative Logarith of Normal Distribution probability @f$
 *          E = \frac{1}{2N} \sum\limits_{n=1}^N \left(
 *          \ln\left(e^{\hat{\sigma}_n} + \epsilon \right) +
 *          \frac{\left( \hat{y}_n - y_n \right)^2}{e^{\hat{\sigma}_n} + \epsilon}
 *          \right) + \frac{\ln(2\pi)}{2} @f$ for real-valued regression tasks.
 *
 * @param bottom input Blob vector (length 3)
 *   -# @f$ (N \times C \times H \times W) @f$
 *      the predictions @f$ \hat{y} \in [-\infty, +\infty]@f$
 *   -# @f$ (N \times C \times H \times W) @f$
 *      the confidence @f$ \hat{y} \in [-\infty, +\infty]@f$
 *   -# @f$ (N \times C \times H \times W) @f$
 *      the targets @f$ y \in [-\infty, +\infty]@f$
 * @param top output Blob vector (length 1)
 *   -# @f$ (1 \times 1 \times 1 \times 1) @f$
 *      the computed Euclidean loss: @f$ E =
 *      \frac{1}{2N} \sum\limits_{n=1}^N \left(
 *      \ln\left(e^{\hat{\sigma}_n} + \epsilon \right) +
 *      \frac{\left( \hat{y}_n - y_n \right)^2}{e^{\hat{\sigma}_n} + \epsilon}
 *      \right) + \frac{\ln(2\pi)}{2}
 *      @f$
 *
 * This can be used for least-squares regression tasks with the model confidence
 * for every prediction. 
 */
template <typename Dtype>
class LogNormLossLayer : public LossLayer<Dtype> {
 public:
  explicit LogNormLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param), diff_(), sqr_diff_(), sigma_(), eps_(1e-6) {}
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "LogNormLoss"; }
  /**
   * Unlike most loss layers, in the EuclideanLossLayer we can backpropagate
   * all inputs -- override to return true and always allow force_backward.
   */
  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return true;
  }
  
  virtual inline int ExactNumBottomBlobs() const { return 3; }

 protected:
  /// @copydoc LogNormLossLayer
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  /**
   * @brief Computes the Negative Logarithm of the Normal distribution
   * probability gradient w.r.t. the inputs.
   *
   * Unlike other children of LossLayer, LogNormLossLayer \b can compute
   * gradients with respect to the label inputs bottom[2] (but still only will
   * if propagate_down[2] is set, due to being produced by learnable parameters
   * or if force_backward is set). In fact, this layer is "commutative" -- the
   * result is the same regardless of the order of the two bottoms.
   *
   * @param top output Blob vector (length 1), providing the error gradient with
   *      respect to the outputs
   *   -# @f$ (1 \times 1 \times 1 \times 1) @f$
   *      This Blob's diff will simply contain the loss_weight* @f$ \lambda @f$,
   *      as @f$ \lambda @f$ is the coefficient of this layer's output
   *      @f$\ell_i@f$ in the overall Net loss
   *      @f$ E = \lambda_i \ell_i + \mbox{other loss terms}@f$; hence
   *      @f$ \frac{\partial E}{\partial \ell_i} = \lambda_i @f$.
   *      (*Assuming that this top Blob is not used as a bottom (input) by any
   *      other layer of the Net.)
   * @param propagate_down see Layer::Backward.
   * @param bottom input Blob vector (length 2)
   *   -# @f$ (N \times C \times H \times W) @f$
   *      the predictions @f$\hat{y}@f$; Backward fills their diff with
   *      gradients @f$
   *        \frac{\partial E}{\partial \hat{y}} =
   *            \frac{1}{n} \sum\limits_{n=1}^N (\hat{y}_n - y_n)
   *      @f$ if propagate_down[0]
   *   -# @f$ (N \times C \times H \times W) @f$
   *      the targets @f$y@f$; Backward fills their diff with gradients
   *      @f$ \frac{\partial E}{\partial y} =
   *          \frac{1}{n} \sum\limits_{n=1}^N (y_n - \hat{y}_n)
   *      @f$ if propagate_down[1]
   */
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  Blob<Dtype> diff_;
  Blob<Dtype> sqr_diff_;
  Blob<Dtype> sigma_;

  Dtype eps_;
};

}  // namespace caffe

#endif  // CAFFE_CAFFE_LOG_NORM_LOSS_LAYER_HPP_
