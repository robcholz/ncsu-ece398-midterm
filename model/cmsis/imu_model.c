#include "imu_model.h"
#include "imu_model_weights.h"

#include "arm_nnfunctions.h"

#define ACT_MIN (-128)
#define ACT_MAX (127)
#define RELU_MIN (0)
#define RELU_MAX (127)

static int8_t conv1_out[1 * 1 * 100 * 16];
static int8_t pool1_out[1 * 1 * 50 * 16];
static int8_t conv2_out[1 * 1 * 50 * 32];
static int8_t pool2_out[1 * 1 * 25 * 32];
static int8_t conv3_out[1 * 1 * 25 * 64];
static int8_t gap_out[64];
static int8_t scratch[8192];
static int32_t avgpool_scratch[64];

static int run_conv(const int8_t *input,
                    int32_t input_w,
                    int32_t input_c,
                    const int8_t *weights,
                    const int32_t *bias,
                    const int32_t *mult,
                    const int32_t *shift,
                    int32_t kernel_w,
                    int32_t output_c,
                    int8_t *output)
{
    cmsis_nn_context ctx = {.buf = scratch, .size = sizeof(scratch)};
    cmsis_nn_conv_params params = {0};
    cmsis_nn_per_channel_quant_params quant = {
        .multiplier = (int32_t *)mult,
        .shift = (int32_t *)shift,
    };
    cmsis_nn_dims input_dims = {.n = 1, .h = 1, .w = input_w, .c = input_c};
    cmsis_nn_dims filter_dims = {.n = output_c, .h = 1, .w = kernel_w, .c = input_c};
    cmsis_nn_dims bias_dims = {.n = 1, .h = 1, .w = 1, .c = output_c};
    cmsis_nn_dims output_dims = {.n = 1, .h = 1, .w = input_w, .c = output_c};

    params.input_offset = 0;
    params.output_offset = 0;
    params.stride.w = 1;
    params.stride.h = 1;
    params.padding.w = kernel_w / 2;
    params.padding.h = 0;
    params.dilation.w = 1;
    params.dilation.h = 1;
    params.activation.min = RELU_MIN;
    params.activation.max = RELU_MAX;

    const int32_t required =
        arm_convolve_wrapper_s8_get_buffer_size(&params, &input_dims, &filter_dims, &output_dims);
    if (required > (int32_t)sizeof(scratch))
    {
        return IMU_MODEL_SCRATCH_TOO_SMALL;
    }
    ctx.size = required;

    const arm_cmsis_nn_status status = arm_convolve_wrapper_s8(
        &ctx, &params, &quant, &input_dims, input, &filter_dims, weights, &bias_dims, bias, &output_dims, output);
    return status == ARM_CMSIS_NN_SUCCESS ? IMU_MODEL_OK : IMU_MODEL_CMSIS_ERROR;
}

static int run_maxpool(const int8_t *input, int32_t input_w, int32_t channels, int8_t *output)
{
    cmsis_nn_context ctx = {.buf = 0, .size = 0};
    cmsis_nn_pool_params params = {0};
    cmsis_nn_dims input_dims = {.n = 1, .h = 1, .w = input_w, .c = channels};
    cmsis_nn_dims filter_dims = {.n = 1, .h = 1, .w = 2, .c = 1};
    cmsis_nn_dims output_dims = {.n = 1, .h = 1, .w = input_w / 2, .c = channels};

    params.stride.w = 2;
    params.stride.h = 1;
    params.padding.w = 0;
    params.padding.h = 0;
    params.activation.min = RELU_MIN;
    params.activation.max = RELU_MAX;

    const arm_cmsis_nn_status status =
        arm_max_pool_s8(&ctx, &params, &input_dims, input, &filter_dims, &output_dims, output);
    return status == ARM_CMSIS_NN_SUCCESS ? IMU_MODEL_OK : IMU_MODEL_CMSIS_ERROR;
}

static int run_gap(const int8_t *input, int8_t *output)
{
    cmsis_nn_context ctx = {.buf = avgpool_scratch, .size = sizeof(avgpool_scratch)};
    cmsis_nn_pool_params params = {0};
    cmsis_nn_dims input_dims = {.n = 1, .h = 1, .w = 25, .c = 64};
    cmsis_nn_dims filter_dims = {.n = 1, .h = 1, .w = 25, .c = 1};
    cmsis_nn_dims output_dims = {.n = 1, .h = 1, .w = 1, .c = 64};

    params.stride.w = 25;
    params.stride.h = 1;
    params.padding.w = 0;
    params.padding.h = 0;
    params.activation.min = ACT_MIN;
    params.activation.max = ACT_MAX;

    const int32_t required = arm_avgpool_s8_get_buffer_size(output_dims.w, input_dims.c);
    if (required > (int32_t)sizeof(avgpool_scratch))
    {
        return IMU_MODEL_SCRATCH_TOO_SMALL;
    }
    ctx.size = required;

    const arm_cmsis_nn_status status =
        arm_avgpool_s8(&ctx, &params, &input_dims, input, &filter_dims, &output_dims, output);
    return status == ARM_CMSIS_NN_SUCCESS ? IMU_MODEL_OK : IMU_MODEL_CMSIS_ERROR;
}

static int run_fc(const int8_t *input, int8_t *output)
{
    cmsis_nn_context ctx = {.buf = scratch, .size = sizeof(scratch)};
    cmsis_nn_fc_params params = {0};
    cmsis_nn_per_tensor_quant_params quant = {
        .multiplier = IMU_FC_MULT,
        .shift = IMU_FC_SHIFT,
    };
    cmsis_nn_dims input_dims = {.n = 1, .h = 1, .w = 1, .c = 64};
    cmsis_nn_dims filter_dims = {.n = 8, .h = 1, .w = 1, .c = 64};
    cmsis_nn_dims bias_dims = {.n = 1, .h = 1, .w = 1, .c = 8};
    cmsis_nn_dims output_dims = {.n = 1, .h = 1, .w = 1, .c = 8};

    params.input_offset = 0;
    params.filter_offset = 0;
    params.output_offset = 0;
    params.activation.min = ACT_MIN;
    params.activation.max = ACT_MAX;

    const int32_t required = arm_fully_connected_s8_get_buffer_size(&filter_dims);
    if (required > (int32_t)sizeof(scratch))
    {
        return IMU_MODEL_SCRATCH_TOO_SMALL;
    }
    ctx.size = required;

    const arm_cmsis_nn_status status = arm_fully_connected_s8(
        &ctx, &params, &quant, &input_dims, input, &filter_dims, IMU_FC_W, &bias_dims, IMU_FC_B, &output_dims, output);
    return status == ARM_CMSIS_NN_SUCCESS ? IMU_MODEL_OK : IMU_MODEL_CMSIS_ERROR;
}

int imu_model_run(const int8_t *input_data, int8_t *output_logits)
{
    int status = run_conv(input_data, 100, 4, IMU_CONV1_W, IMU_CONV1_B, IMU_CONV1_MULT, IMU_CONV1_SHIFT, 5, 16, conv1_out);
    if (status != IMU_MODEL_OK)
        return status;
    status = run_maxpool(conv1_out, 100, 16, pool1_out);
    if (status != IMU_MODEL_OK)
        return status;
    status = run_conv(pool1_out, 50, 16, IMU_CONV2_W, IMU_CONV2_B, IMU_CONV2_MULT, IMU_CONV2_SHIFT, 5, 32, conv2_out);
    if (status != IMU_MODEL_OK)
        return status;
    status = run_maxpool(conv2_out, 50, 32, pool2_out);
    if (status != IMU_MODEL_OK)
        return status;
    status = run_conv(pool2_out, 25, 32, IMU_CONV3_W, IMU_CONV3_B, IMU_CONV3_MULT, IMU_CONV3_SHIFT, 3, 64, conv3_out);
    if (status != IMU_MODEL_OK)
        return status;
    status = run_gap(conv3_out, gap_out);
    if (status != IMU_MODEL_OK)
        return status;
    return run_fc(gap_out, output_logits);
}

float imu_model_input_scale(void)
{
    return IMU_MODEL_INPUT_SCALE;
}

const float *imu_model_norm_mean(void)
{
    return IMU_MODEL_NORM_MEAN;
}

const float *imu_model_norm_std(void)
{
    return IMU_MODEL_NORM_STD;
}

const char *imu_model_class_name(int32_t class_index)
{
    if (class_index < 0 || class_index >= IMU_MODEL_CLASSES)
    {
        return "";
    }
    return IMU_MODEL_CLASS_NAMES[class_index];
}
