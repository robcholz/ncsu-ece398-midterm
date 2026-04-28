#pragma once

#include <stdint.h>

#define IMU_MODEL_INPUT_LEN 100
#define IMU_MODEL_CHANNELS 4
#define IMU_MODEL_INPUT_SIZE (IMU_MODEL_INPUT_LEN * IMU_MODEL_CHANNELS)
#define IMU_MODEL_CLASSES 8

typedef enum
{
    IMU_MODEL_OK = 0,
    IMU_MODEL_CMSIS_ERROR = -1,
    IMU_MODEL_SCRATCH_TOO_SMALL = -2,
} imu_model_status_t;

int imu_model_run(const int8_t *input_data, int8_t *output_logits);
float imu_model_input_scale(void);
const float *imu_model_norm_mean(void);
const float *imu_model_norm_std(void);
const char *imu_model_class_name(int32_t class_index);
