/* Copyright 2019-2020 Canaan Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "../tflite_importer.h"
#include <hlir/ops/conv2d.h>
#include <quantize.h>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::hlir;

DEFINE_TFLITE_LOWER(CONV_2D)
{
    auto &input = get_tensor(op.inputs(), 0);
    auto &weights = get_tensor(op.inputs(), 1);
    auto &bias = get_tensor(op.inputs(), 2);
    auto &options = *op.builtin_options_as_Conv2DOptions();

    auto weights_shape = krsc_to_kcrs(get_shape(weights.shape()));

    auto pre_trans = nhwc_to_nchw(to_data_type(input.type()), get_shape(input.shape()));

    auto in_h = pre_trans->output().shape()[2];
    auto in_w = pre_trans->output().shape()[3];
    auto f_h = weights_shape[2];
    auto f_w = weights_shape[3];
    auto stride_h = options.stride_h();
    auto stride_w = options.stride_w();
    auto dilation_h = options.dilation_h_factor();
    auto dilation_w = options.dilation_w_factor();
    auto pad_h = get_windowed_padding(in_h, f_h, stride_h, dilation_h, options.padding() == tflite::Padding_SAME);
    auto pad_w = get_windowed_padding(in_w, f_w, stride_w, dilation_w, options.padding() == tflite::Padding_SAME);
    transpose *sur_trans;

    // Quantized Conv2D
    if (input.type() == tflite::TensorType_UINT8 && weights.type() == tflite::TensorType_UINT8)
    {
        auto &output = get_tensor(op.outputs(), 0);
        auto in_p = quant::get_quant_param(to_value_range(*input.quantization()), 8);
        auto w_p = quant::get_quant_param(to_value_range(*weights.quantization()), 8);
        auto out_p = quant::get_quant_param(to_value_range(*output.quantization()), 8);
        auto mul = out_p.scale / (in_p.scale * w_p.scale);
        auto fmul_shift = quant::get_fixed_mul(mul, 32, 31, true);

        auto weights_tensor = xt::transpose(load_tensor<uint8_t, 4>(weights), { 0, 3, 1, 2 });
        auto bias_tensor = load_tensor<int32_t, 1>(bias);
        auto conv = graph_.emplace<quantized_conv2d>(pre_trans->output().shape(), std::move(weights_tensor), std::move(bias_tensor), 1,
            pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, -in_p.zero_point, -w_p.zero_point, fmul_shift.rounded_mul(), fmul_shift.shift, out_p.zero_point);
        conv->input().connect(pre_trans->output());

        sur_trans = nchw_to_nhwc(dt_uint8, conv->output().shape());
        sur_trans->input().connect(conv->output());
    }
    else
    {
        auto weights_tensor = xt::transpose(dequantize_tensor<4>(weights), { 0, 3, 1, 2 });
        auto bias_tensor = load_tensor<float, 1>(bias);
        auto conv = graph_.emplace<conv2d>(pre_trans->output().shape(), std::move(weights_tensor), std::move(bias_tensor), 1,
            pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, to_float_clamp_range(options.fused_activation_function()));
        conv->input().connect(pre_trans->output());

        sur_trans = nchw_to_nhwc(dt_float32, conv->output().shape());
        sur_trans->input().connect(conv->output());
    }

    input_tensors_.emplace(&pre_trans->input(), op.inputs()->Get(0));
    output_tensors_.emplace(op.outputs()->Get(0), &sur_trans->output());
}

DEFINE_TFLITE_LOWER(DEPTHWISE_CONV_2D)
{
    auto &input = get_tensor(op.inputs(), 0);
    auto &weights = get_tensor(op.inputs(), 1);
    auto &bias = get_tensor(op.inputs(), 2);
    auto &options = *op.builtin_options_as_DepthwiseConv2DOptions();
    auto weights_shape = dw_rsc_to_kcrs(get_shape(weights.shape()));
    auto opname = bias.name()->string_view().substr(0, bias.name()->string_view().find_last_of('/'));

    auto pre_trans = nhwc_to_nchw(to_data_type(input.type()), get_shape(input.shape()));

    auto in_h = pre_trans->output().shape()[2];
    auto in_w = pre_trans->output().shape()[3];
    auto groups = weights_shape[0];
    auto f_h = weights_shape[2];
    auto f_w = weights_shape[3];
    auto stride_h = options.stride_h();
    auto stride_w = options.stride_w();
    auto dilation_h = options.dilation_h_factor();
    auto dilation_w = options.dilation_w_factor();
    auto pad_h = get_windowed_padding(in_h, f_h, stride_h, dilation_h, options.padding() == tflite::Padding_SAME);
    auto pad_w = get_windowed_padding(in_w, f_w, stride_w, dilation_w, options.padding() == tflite::Padding_SAME);
    auto depth_mul = options.depth_multiplier();
    transpose *sur_trans;
    if (depth_mul != 1)
        throw std::runtime_error("DepthwiseConv2d " + std::string(opname) + " with depth_multiplier " + std::to_string(depth_mul) + " is not supported");

    // Quantized DepthwiseConv2D
    if (input.type() == tflite::TensorType_UINT8 && weights.type() == tflite::TensorType_UINT8)
    {
        auto &output = get_tensor(op.outputs(), 0);
        auto in_p = quant::get_quant_param(to_value_range(*input.quantization()), 8);
        auto w_p = quant::get_quant_param(to_value_range(*weights.quantization()), 8);
        auto out_p = quant::get_quant_param(to_value_range(*output.quantization()), 8);
        auto mul = out_p.scale / (in_p.scale * w_p.scale);
        auto fmul_shift = quant::get_fixed_mul(mul, 32, 31, true);

        auto weights_tensor = xt::transpose(load_tensor<uint8_t, 4>(weights), { 3, 0, 1, 2 });
        auto bias_tensor = load_tensor<int32_t, 1>(bias);
        auto conv = graph_.emplace<quantized_conv2d>(pre_trans->output().shape(), std::move(weights_tensor), std::move(bias_tensor), groups,
            pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, -in_p.zero_point, -w_p.zero_point, fmul_shift.rounded_mul(), fmul_shift.shift, out_p.zero_point);
        conv->name(opname);
        conv->input().connect(pre_trans->output());

        sur_trans = nchw_to_nhwc(dt_uint8, conv->output().shape());
        sur_trans->input().connect(conv->output());
    }
    else
    {
        auto weights_tensor = xt::transpose(dequantize_tensor<4>(weights), { 3, 0, 1, 2 });
        auto bias_tensor = load_tensor<float, 1>(bias);
        auto conv = graph_.emplace<conv2d>(pre_trans->output().shape(), std::move(weights_tensor), std::move(bias_tensor), groups,
            pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, to_float_clamp_range(options.fused_activation_function()));
        conv->name(opname);
        conv->input().connect(pre_trans->output());

        sur_trans = nchw_to_nhwc(dt_float32, conv->output().shape());
        sur_trans->input().connect(conv->output());
    }

    input_tensors_.emplace(&pre_trans->input(), op.inputs()->Get(0));
    output_tensors_.emplace(op.outputs()->Get(0), &sur_trans->output());
}