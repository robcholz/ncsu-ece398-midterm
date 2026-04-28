include!(concat!(env!("OUT_DIR"), "/model_constants.rs"));

const G_TO_M_S2: f32 = 9.80665;

unsafe extern "C" {
    fn imu_model_run(input_data: *const i8, output_logits: *mut i8) -> i32;
    fn imu_model_input_scale() -> f32;
    fn imu_model_norm_mean() -> *const f32;
    fn imu_model_norm_std() -> *const f32;
    fn imu_model_class_name(class_index: i32) -> *const u8;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Error {
    CmsisError,
    ScratchTooSmall,
    Unknown(i32),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Prediction {
    pub class_index: usize,
    pub class_name: &'static str,
    pub logits: [i8; CLASSES],
}

pub struct QuantizedCnn;

impl QuantizedCnn {
    pub fn run(input: &[i8; INPUT_SIZE]) -> Result<Prediction, Error> {
        let mut logits = [0i8; CLASSES];
        let status = unsafe { imu_model_run(input.as_ptr(), logits.as_mut_ptr()) };
        match status {
            0 => {
                let class_index = argmax(&logits);
                Ok(Prediction {
                    class_index,
                    class_name: class_name(class_index),
                    logits,
                })
            }
            -1 => Err(Error::CmsisError),
            -2 => Err(Error::ScratchTooSmall),
            other => Err(Error::Unknown(other)),
        }
    }

    pub fn run_samples_g(samples: &[[f32; CHANNELS]]) -> Result<Prediction, Error> {
        assert!(samples.len() >= INPUT_LEN);

        let mut input = [0i8; INPUT_SIZE];
        for sample in 0..INPUT_LEN {
            for channel in 0..CHANNELS {
                input[sample * CHANNELS + channel] =
                    quantize_normalized(samples[sample][channel], channel);
            }
        }

        Self::run(&input)
    }
}

pub struct ImuWindow {
    input: [i8; INPUT_SIZE],
    next_sample: usize,
    filled: bool,
}

impl ImuWindow {
    pub const fn new() -> Self {
        Self {
            input: [0; INPUT_SIZE],
            next_sample: 0,
            filled: false,
        }
    }

    pub fn push_sample_m_s2(
        &mut self,
        ax: f32,
        ay: f32,
        az: f32,
    ) -> Option<Result<Prediction, Error>> {
        self.push_sample_g(ax / G_TO_M_S2, ay / G_TO_M_S2, az / G_TO_M_S2)
    }

    pub fn push_sample_g(
        &mut self,
        ax: f32,
        ay: f32,
        az: f32,
    ) -> Option<Result<Prediction, Error>> {
        let magnitude = sqrt_approx(ax * ax + ay * ay + az * az);
        let values = [ax, ay, az, magnitude];
        let base = self.next_sample * CHANNELS;
        for (channel, value) in values.iter().enumerate().take(CHANNELS) {
            self.input[base + channel] = quantize_normalized(*value, channel);
        }

        self.next_sample += 1;
        if self.next_sample == INPUT_LEN {
            self.next_sample = 0;
            self.filled = true;
        }

        if self.filled {
            Some(QuantizedCnn::run(&self.input))
        } else {
            None
        }
    }
}

impl Default for ImuWindow {
    fn default() -> Self {
        Self::new()
    }
}

fn quantize_normalized(value: f32, channel: usize) -> i8 {
    let normalized = (value - norm_mean(channel)) / norm_std(channel);
    let quantized = round_to_i32(normalized / input_scale()).clamp(-128, 127);
    quantized as i8
}

pub fn input_scale() -> f32 {
    unsafe { imu_model_input_scale() }
}

pub fn norm_mean(channel: usize) -> f32 {
    assert!(channel < CHANNELS);
    unsafe { *imu_model_norm_mean().add(channel) }
}

pub fn norm_std(channel: usize) -> f32 {
    assert!(channel < CHANNELS);
    unsafe { *imu_model_norm_std().add(channel) }
}

pub fn class_name(class_index: usize) -> &'static str {
    assert!(class_index < CLASSES);
    unsafe { c_string_to_str(imu_model_class_name(class_index as i32)) }
}

unsafe fn c_string_to_str(ptr: *const u8) -> &'static str {
    if ptr.is_null() {
        return "";
    }
    let mut len = 0usize;
    while unsafe { *ptr.add(len) } != 0 {
        len += 1;
    }
    let bytes = unsafe { core::slice::from_raw_parts(ptr, len) };
    unsafe { core::str::from_utf8_unchecked(bytes) }
}

fn round_to_i32(value: f32) -> i32 {
    if value >= 0.0 {
        (value + 0.5) as i32
    } else {
        (value - 0.5) as i32
    }
}

fn argmax(logits: &[i8; CLASSES]) -> usize {
    let mut best_idx = 0;
    let mut best = logits[0];
    for (idx, &value) in logits.iter().enumerate().skip(1) {
        if value > best {
            best = value;
            best_idx = idx;
        }
    }
    best_idx
}

fn sqrt_approx(value: f32) -> f32 {
    if value <= 0.0 {
        return 0.0;
    }
    let mut guess = if value >= 1.0 { value } else { 1.0 };
    for _ in 0..6 {
        guess = 0.5 * (guess + value / guess);
    }
    guess
}
