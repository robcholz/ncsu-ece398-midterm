#![cfg_attr(all(target_arch = "arm", target_os = "none"), no_std)]
#![cfg_attr(all(target_arch = "arm", target_os = "none"), no_main)]

include!(concat!(env!("OUT_DIR"), "/benchmark_data.rs"));

#[cfg(not(all(target_arch = "arm", target_os = "none")))]
fn main() {
    println!("benchmark target: STMBox / SensorTile Box Pro");
    println!("embedded-only benchmark binary");
    println!(
        "prepared {} windows from recordings/ with {} samples each",
        BENCHMARK_WINDOW_COUNT, BENCHMARK_WINDOW_SAMPLES
    );
    println!("build with --target thumbv8m.main-none-eabihf and flash to run measurements");
}

#[cfg(all(target_arch = "arm", target_os = "none"))]
mod embedded {
    extern crate alloc;

    use alloc::vec;
    use alloc::vec::Vec;
    use core::alloc::{GlobalAlloc, Layout};
    use core::fmt::{self, Write as _};
    use core::hint::black_box;
    use core::mem::MaybeUninit;
    use core::sync::atomic::{AtomicUsize, Ordering};

    use cortex_m::register::msp;
    use embassy_executor::Spawner;
    use embassy_futures::join::join;
    use embassy_stm32::usb::{Driver, Instance};
    use embassy_stm32::{Config, bind_interrupts, peripherals, usb};
    use embassy_time::Instant;
    use embassy_usb::Builder;
    use embassy_usb::class::cdc_acm::{CdcAcmClass, State};
    use embassy_usb::driver::EndpointError;
    use embedded_alloc::LlffHeap;
    use heapless::String;
    use panic_halt as _;
    use crate::{BENCHMARK_WINDOW_COUNT, BENCHMARK_WINDOWS, BENCHMARK_WINDOW_SAMPLES};

    const OUTPUT_CLASSES: usize = 8;
    const RUNS_PER_EXPERIMENT: usize = 64;
    const INFERENCE_FREQUENCY_HZ: f32 = 10.0;
    const HEAP_SIZE_BYTES: usize = 256 * 1024;
    const STACK_SCRATCH_WORDS: usize = 512;

    const TINY_MODEL_BYTES: usize = 6_144;
    const SMALL_MODEL_BYTES: usize = 38_912;
    const MEDIUM_MODEL_BYTES: usize = 145_408;

    static TINY_MODEL_FLASH: [u8; TINY_MODEL_BYTES] = [0x11; TINY_MODEL_BYTES];
    static SMALL_MODEL_FLASH: [u8; SMALL_MODEL_BYTES] = [0x33; SMALL_MODEL_BYTES];
    static MEDIUM_MODEL_FLASH: [u8; MEDIUM_MODEL_BYTES] = [0x55; MEDIUM_MODEL_BYTES];

    #[global_allocator]
    static ALLOCATOR: TrackingAllocator = TrackingAllocator::empty();

    #[unsafe(link_section = ".uninit.benchmark_heap")]
    static mut HEAP_MEMORY: [MaybeUninit<u8>; HEAP_SIZE_BYTES] =
        [MaybeUninit::uninit(); HEAP_SIZE_BYTES];

    bind_interrupts!(struct Irqs {
        OTG_FS => usb::InterruptHandler<peripherals::USB_OTG_FS>;
    });

    #[derive(Clone, Copy)]
    enum PreprocessKind {
        None,
        Normalize,
        LowPass20Hz,
        Augmentation,
    }

    #[derive(Clone, Copy)]
    enum ModelKind {
        Tiny,
        Small,
        Medium,
    }

    #[derive(Clone, Copy)]
    struct Experiment {
        id: &'static str,
        input_channels: usize,
        preprocess: PreprocessKind,
        model: ModelKind,
        purpose: &'static str,
    }

    const EXPERIMENTS: [Experiment; 9] = [
        Experiment {
            id: "E0",
            input_channels: 3,
            preprocess: PreprocessKind::None,
            model: ModelKind::Small,
            purpose: "baseline",
        },
        Experiment {
            id: "E1",
            input_channels: 4,
            preprocess: PreprocessKind::None,
            model: ModelKind::Small,
            purpose: "magnitude feature",
        },
        Experiment {
            id: "E2",
            input_channels: 3,
            preprocess: PreprocessKind::Normalize,
            model: ModelKind::Small,
            purpose: "normalization",
        },
        Experiment {
            id: "E3",
            input_channels: 4,
            preprocess: PreprocessKind::Normalize,
            model: ModelKind::Small,
            purpose: "best 4ch variant",
        },
        Experiment {
            id: "E4",
            input_channels: 3,
            preprocess: PreprocessKind::LowPass20Hz,
            model: ModelKind::Small,
            purpose: "filter test",
        },
        Experiment {
            id: "E5",
            input_channels: 4,
            preprocess: PreprocessKind::LowPass20Hz,
            model: ModelKind::Small,
            purpose: "filter + mag",
        },
        Experiment {
            id: "E6",
            input_channels: 4,
            preprocess: PreprocessKind::Normalize,
            model: ModelKind::Tiny,
            purpose: "edge candidate",
        },
        Experiment {
            id: "E7",
            input_channels: 4,
            preprocess: PreprocessKind::Normalize,
            model: ModelKind::Medium,
            purpose: "capacity test",
        },
        Experiment {
            id: "E8",
            input_channels: 4,
            preprocess: PreprocessKind::Augmentation,
            model: ModelKind::Small,
            purpose: "robustness",
        },
    ];

    const BATCH_SIZES: [usize; 5] = [1, 2, 4, 8, 16];

    #[derive(Clone, Copy)]
    struct ModelProfile {
        kind: ModelKind,
        label: &'static str,
        channels: [usize; 3],
        kernels: [usize; 3],
        flash_bytes: usize,
        weights: &'static [u8],
    }

    impl ModelKind {
        fn profile(self) -> ModelProfile {
            match self {
                Self::Tiny => ModelProfile {
                    kind: self,
                    label: "Tiny CNN",
                    channels: [8, 16, 16],
                    kernels: [5, 5, 3],
                    flash_bytes: TINY_MODEL_BYTES,
                    weights: &TINY_MODEL_FLASH,
                },
                Self::Small => ModelProfile {
                    kind: self,
                    label: "Small CNN",
                    channels: [16, 32, 64],
                    kernels: [5, 5, 3],
                    flash_bytes: SMALL_MODEL_BYTES,
                    weights: &SMALL_MODEL_FLASH,
                },
                Self::Medium => ModelProfile {
                    kind: self,
                    label: "Medium CNN",
                    channels: [32, 64, 128],
                    kernels: [5, 5, 3],
                    flash_bytes: MEDIUM_MODEL_BYTES,
                    weights: &MEDIUM_MODEL_FLASH,
                },
            }
        }
    }

    impl PreprocessKind {
        fn label(self) -> &'static str {
            match self {
                Self::None => "none",
                Self::Normalize => "normalize",
                Self::LowPass20Hz => "LPF20Hz",
                Self::Augmentation => "augmentation",
            }
        }
    }

    struct HeapSnapshot {
        current_bytes: usize,
        peak_bytes: usize,
    }

    struct BenchmarkResult {
        avg_inference_time_us: u32,
        p95_inference_time_us: u32,
        peak_heap_usage_bytes: usize,
        heap_watermark_bytes: usize,
        stack_watermark_bytes: usize,
        cpu_pressure_milli: u32,
        model_binary_size_bytes: usize,
    }

    struct TrackingAllocator {
        inner: LlffHeap,
        current: AtomicUsize,
        peak: AtomicUsize,
    }

    impl TrackingAllocator {
        const fn empty() -> Self {
            Self {
                inner: LlffHeap::empty(),
                current: AtomicUsize::new(0),
                peak: AtomicUsize::new(0),
            }
        }

        unsafe fn init(&self, start: usize, size: usize) {
            unsafe { self.inner.init(start, size) };
            self.current.store(0, Ordering::SeqCst);
            self.peak.store(0, Ordering::SeqCst);
        }

        fn snapshot(&self) -> HeapSnapshot {
            HeapSnapshot {
                current_bytes: self.current.load(Ordering::SeqCst),
                peak_bytes: self.peak.load(Ordering::SeqCst),
            }
        }

        fn reset_peak_to_current(&self) {
            let current = self.current.load(Ordering::SeqCst);
            self.peak.store(current, Ordering::SeqCst);
        }

        fn record_allocation(&self, size: usize) {
            let new_current = self.current.fetch_add(size, Ordering::SeqCst) + size;
            let mut peak = self.peak.load(Ordering::SeqCst);

            while new_current > peak {
                match self.peak.compare_exchange(
                    peak,
                    new_current,
                    Ordering::SeqCst,
                    Ordering::SeqCst,
                ) {
                    Ok(_) => break,
                    Err(observed) => peak = observed,
                }
            }
        }

        fn record_deallocation(&self, size: usize) {
            self.current.fetch_sub(size, Ordering::SeqCst);
        }
    }

    unsafe impl GlobalAlloc for TrackingAllocator {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            let ptr = unsafe { self.inner.alloc(layout) };
            if !ptr.is_null() {
                self.record_allocation(layout.size());
            }
            ptr
        }

        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            unsafe { self.inner.dealloc(ptr, layout) };
            self.record_deallocation(layout.size());
        }
    }

    #[embassy_executor::main]
    async fn main(_spawner: Spawner) {
        unsafe {
            ALLOCATOR.init(core::ptr::addr_of_mut!(HEAP_MEMORY) as *mut u8 as usize, HEAP_SIZE_BYTES);
        }

        let mut config = Config::default();
        {
            use embassy_stm32::rcc::*;
            config.rcc.hsi = true;
            config.rcc.pll1 = Some(Pll {
                source: PllSource::HSI,
                prediv: PllPreDiv::DIV1,
                mul: PllMul::MUL10,
                divp: None,
                divq: None,
                divr: Some(PllDiv::DIV1),
            });
            config.rcc.sys = Sysclk::PLL1_R;
            config.rcc.voltage_range = VoltageScale::RANGE1;
            config.rcc.hsi48 = Some(Hsi48Config {
                sync_from_usb: true,
            });
            config.rcc.mux.iclksel = mux::Iclksel::HSI48;
        }

        let p = embassy_stm32::init(config);

        let mut ep_out_buffer = [0u8; 256];
        let mut driver_config = embassy_stm32::usb::Config::default();
        driver_config.vbus_detection = false;
        let driver = Driver::new_fs(
            p.USB_OTG_FS,
            Irqs,
            p.PA12,
            p.PA11,
            &mut ep_out_buffer,
            driver_config,
        );

        let mut usb_config = embassy_usb::Config::new(0x0483, 0x5740);
        usb_config.manufacturer = Some("ST + Rust");
        usb_config.product = Some("MKBOXPRO Benchmark");
        usb_config.serial_number = Some("mkboxpro-benchmark");

        let mut config_descriptor = [0u8; 256];
        let mut bos_descriptor = [0u8; 256];
        let mut control_buf = [0u8; 64];
        let mut state = State::new();

        let mut builder = Builder::new(
            driver,
            usb_config,
            &mut config_descriptor,
            &mut bos_descriptor,
            &mut [],
            &mut control_buf,
        );

        let mut class = CdcAcmClass::new(&mut builder, &mut state, 64);
        let mut usb = builder.build();

        let usb_fut = usb.run();
        let app_fut = async {
            loop {
                class.wait_connection().await;
                let _ = run_benchmark(&mut class).await;
            }
        };

        join(usb_fut, app_fut).await;
    }

    async fn run_benchmark<'d, T: Instance + 'd>(
        class: &mut CdcAcmClass<'d, Driver<'d, T>>,
    ) -> Result<(), Disconnected> {
        write_line(class, "benchmark_start board=STMBOXPRO source=recordings fake_workload=cnn").await?;

        for experiment in EXPERIMENTS {
            let profile = experiment.model.profile();
            let mut header: String<128> = String::new();
            write!(
                &mut header,
                "experiment={} preprocess={} model={} channels={} purpose={}",
                experiment.id,
                experiment.preprocess.label(),
                profile.label,
                experiment.input_channels,
                experiment.purpose
            )
            .unwrap();
            write_line(class, &header).await?;

            for batch_size in BATCH_SIZES {
                match benchmark_experiment(experiment, batch_size) {
                    Ok(result) => {
                        let mut line: String<192> = String::new();
                        write!(
                            &mut line,
                            "batch={} model_binary_size_bytes={} inference_time_us={} p95_inference_time_us={} peak_heap_usage_bytes={} heap_watermark_bytes={} stack_watermark_bytes={} cpu_pressure_milli={}",
                            batch_size,
                            result.model_binary_size_bytes,
                            result.avg_inference_time_us,
                            result.p95_inference_time_us,
                            result.peak_heap_usage_bytes,
                            result.heap_watermark_bytes,
                            result.stack_watermark_bytes,
                            result.cpu_pressure_milli,
                        )
                        .unwrap();
                        write_line(class, &line).await?;
                    }
                    Err(err) => {
                        let mut line: String<128> = String::new();
                        write!(&mut line, "batch={} error={}", batch_size, err).unwrap();
                        write_line(class, &line).await?;
                    }
                }
            }
        }

        write_line(class, "benchmark_done").await
    }

    fn benchmark_experiment(
        experiment: Experiment,
        batch_size: usize,
    ) -> Result<BenchmarkResult, BenchmarkError> {
        let profile = experiment.model.profile();
        let batch_input = prepare_batch_input(experiment, batch_size);

        let before = ALLOCATOR.snapshot();
        ALLOCATOR.reset_peak_to_current();

        let mut timings_us = [0u32; RUNS_PER_EXPERIMENT];
        let mut min_stack_remaining = usize::MAX;

        for slot in timings_us.iter_mut() {
            let started = Instant::now();
            let remaining = fake_inference(&batch_input, batch_size, experiment.input_channels, profile)?;
            let elapsed = started.elapsed().as_micros() as u32;
            *slot = elapsed;
            if remaining < min_stack_remaining {
                min_stack_remaining = remaining;
            }
        }

        let after = ALLOCATOR.snapshot();
        let avg_us = average_u32(&timings_us);
        let p95_us = percentile_u32(&timings_us, 95);
        let peak_heap_usage_bytes = after.peak_bytes.saturating_sub(before.current_bytes);
        let heap_watermark_bytes = HEAP_SIZE_BYTES.saturating_sub(after.peak_bytes);
        let cpu_pressure_milli =
            ((avg_us as u64 * INFERENCE_FREQUENCY_HZ as u64 * 1000) / 1_000_000) as u32;

        Ok(BenchmarkResult {
            avg_inference_time_us: avg_us,
            p95_inference_time_us: p95_us,
            peak_heap_usage_bytes,
            heap_watermark_bytes,
            stack_watermark_bytes: min_stack_remaining,
            cpu_pressure_milli,
            model_binary_size_bytes: profile.flash_bytes,
        })
    }

    fn prepare_batch_input(experiment: Experiment, batch_size: usize) -> Vec<f32> {
        let mut batch = vec![0.0; batch_size * BENCHMARK_WINDOW_SAMPLES * experiment.input_channels];
        for batch_index in 0..batch_size {
            let window = &BENCHMARK_WINDOWS[batch_index % BENCHMARK_WINDOW_COUNT];
            let base = batch_index * BENCHMARK_WINDOW_SAMPLES * experiment.input_channels;
            write_window(&mut batch[base..base + BENCHMARK_WINDOW_SAMPLES * experiment.input_channels], window, experiment);
        }
        batch
    }

    fn write_window(
        destination: &mut [f32],
        window: &[[f32; 4]; BENCHMARK_WINDOW_SAMPLES],
        experiment: Experiment,
    ) {
        match experiment.preprocess {
            PreprocessKind::None => fill_channels(destination, window, experiment.input_channels),
            PreprocessKind::Normalize => {
                fill_channels(destination, window, experiment.input_channels);
                normalize_in_place(destination, experiment.input_channels);
            }
            PreprocessKind::LowPass20Hz => {
                fill_channels(destination, window, experiment.input_channels);
                low_pass_in_place(destination, experiment.input_channels);
            }
            PreprocessKind::Augmentation => {
                fill_channels(destination, window, experiment.input_channels);
                normalize_in_place(destination, experiment.input_channels);
                augment_in_place(destination, experiment.input_channels);
            }
        }
    }

    fn fill_channels(
        destination: &mut [f32],
        window: &[[f32; 4]; BENCHMARK_WINDOW_SAMPLES],
        channels: usize,
    ) {
        for sample_index in 0..BENCHMARK_WINDOW_SAMPLES {
            for channel_index in 0..channels {
                destination[sample_index * channels + channel_index] = window[sample_index][channel_index];
            }
        }
    }

    fn normalize_in_place(values: &mut [f32], channels: usize) {
        let mut means = [0.0f32; 4];
        let mut vars = [0.0f32; 4];

        for channel in 0..channels {
            let mut sum = 0.0;
            for sample in 0..BENCHMARK_WINDOW_SAMPLES {
                sum += values[sample * channels + channel];
            }
            means[channel] = sum / BENCHMARK_WINDOW_SAMPLES as f32;
        }

        for channel in 0..channels {
            let mut sum = 0.0;
            for sample in 0..BENCHMARK_WINDOW_SAMPLES {
                let centered = values[sample * channels + channel] - means[channel];
                sum += centered * centered;
            }
            vars[channel] = sqrt_approx(sum / BENCHMARK_WINDOW_SAMPLES as f32).max(1e-4);
        }

        for channel in 0..channels {
            for sample in 0..BENCHMARK_WINDOW_SAMPLES {
                let index = sample * channels + channel;
                values[index] = (values[index] - means[channel]) / vars[channel];
            }
        }
    }

    fn low_pass_in_place(values: &mut [f32], channels: usize) {
        const ALPHA: f32 = 0.55;
        for channel in 0..channels {
            let mut prev = values[channel];
            for sample in 1..BENCHMARK_WINDOW_SAMPLES {
                let index = sample * channels + channel;
                let filtered = prev + ALPHA * (values[index] - prev);
                values[index] = filtered;
                prev = filtered;
            }
        }
    }

    fn augment_in_place(values: &mut [f32], channels: usize) {
        for sample in 0..BENCHMARK_WINDOW_SAMPLES {
            let shifted = (sample + 3).min(BENCHMARK_WINDOW_SAMPLES - 1);
            for channel in 0..channels {
                let index = sample * channels + channel;
                let source = shifted * channels + channel;
                values[index] = values[source] * (0.97 + channel as f32 * 0.01);
            }
        }
    }

    fn fake_inference(
        input: &[f32],
        batch_size: usize,
        input_channels: usize,
        profile: ModelProfile,
    ) -> Result<usize, BenchmarkError> {
        let stack_probe = stack_probe(STACK_SCRATCH_WORDS);

        let layer1_len = BENCHMARK_WINDOW_SAMPLES / 2;
        let layer1 = temporal_block(
            input,
            batch_size,
            input_channels,
            BENCHMARK_WINDOW_SAMPLES,
            profile.channels[0],
            profile.kernels[0],
            layer1_len,
            profile.weights,
            0,
        )?;

        let layer2_len = layer1_len / 2;
        let layer2 = temporal_block(
            &layer1,
            batch_size,
            profile.channels[0],
            layer1_len,
            profile.channels[1],
            profile.kernels[1],
            layer2_len,
            profile.weights,
            17,
        )?;

        let layer3_len = layer2_len;
        let layer3 = temporal_block(
            &layer2,
            batch_size,
            profile.channels[1],
            layer2_len,
            profile.channels[2],
            profile.kernels[2],
            layer3_len,
            profile.weights,
            41,
        )?;

        let pooled = global_average_pool(&layer3, batch_size, profile.channels[2], layer3_len);
        let logits = dense_head(&pooled, batch_size, profile.channels[2], profile.weights)?;

        let checksum = logits.iter().fold(0.0f32, |acc, value| acc + *value);
        black_box((checksum, stack_probe, profile.kind as u8));

        Ok(current_stack_remaining())
    }

    fn temporal_block(
        input: &[f32],
        batch_size: usize,
        in_channels: usize,
        input_len: usize,
        out_channels: usize,
        kernel: usize,
        output_len: usize,
        weights: &[u8],
        weight_offset: usize,
    ) -> Result<Vec<f32>, BenchmarkError> {
        let mut output = vec![0.0; batch_size * out_channels * output_len];
        let stride = (input_len / output_len).max(1);

        for batch_index in 0..batch_size {
            for out_channel in 0..out_channels {
                for out_index in 0..output_len {
                    let mut acc = 0.0f32;
                    for in_channel in 0..in_channels {
                        for kernel_index in 0..kernel {
                            let src_index = (out_index * stride + kernel_index).min(input_len - 1);
                            let input_index =
                                ((batch_index * input_len + src_index) * in_channels) + in_channel;
                            let weight_index =
                                (weight_offset + (((out_channel * in_channels) + in_channel) * kernel) + kernel_index)
                                    % weights.len();
                            let weight = (weights[weight_index] as f32 - 96.0) / 255.0;
                            acc += input[input_index] * weight;
                        }
                    }
                    if acc < 0.0 {
                        acc = 0.0;
                    }
                    let output_index =
                        ((batch_index * out_channels + out_channel) * output_len) + out_index;
                    output[output_index] = acc;
                }
            }
        }

        if output.is_empty() {
            return Err(BenchmarkError::InferenceFailed("empty output from temporal block"));
        }
        Ok(output)
    }

    fn global_average_pool(
        input: &[f32],
        batch_size: usize,
        channels: usize,
        samples: usize,
    ) -> Vec<f32> {
        let mut pooled = vec![0.0; batch_size * channels];

        for batch_index in 0..batch_size {
            for channel in 0..channels {
                let mut sum = 0.0f32;
                for sample in 0..samples {
                    let index = ((batch_index * channels + channel) * samples) + sample;
                    sum += input[index];
                }
                pooled[batch_index * channels + channel] = sum / samples as f32;
            }
        }

        pooled
    }

    fn dense_head(
        pooled: &[f32],
        batch_size: usize,
        channels: usize,
        weights: &[u8],
    ) -> Result<Vec<f32>, BenchmarkError> {
        let mut logits = vec![0.0; batch_size * OUTPUT_CLASSES];
        for batch_index in 0..batch_size {
            for class_index in 0..OUTPUT_CLASSES {
                let mut acc = 0.0f32;
                for channel in 0..channels {
                    let weight_index = (class_index * channels + channel) % weights.len();
                    let weight = (weights[weight_index] as f32 - 127.0) / 511.0;
                    acc += pooled[batch_index * channels + channel] * weight;
                }
                logits[batch_index * OUTPUT_CLASSES + class_index] = acc;
            }
        }

        if logits.is_empty() {
            return Err(BenchmarkError::InferenceFailed("empty output from dense head"));
        }
        Ok(logits)
    }

    fn stack_probe(words: usize) -> usize {
        let mut scratch = [0u32; STACK_SCRATCH_WORDS];
        let limit = words.min(STACK_SCRATCH_WORDS);
        for (index, slot) in scratch.iter_mut().take(limit).enumerate() {
            *slot = 0xA500_0000 | index as u32;
        }
        black_box(scratch[limit.saturating_sub(1)]);
        current_stack_remaining()
    }

    fn current_stack_remaining() -> usize {
        unsafe extern "C" {
            static _stack_end: u32;
        }

        let stack_end = core::ptr::addr_of!(_stack_end) as usize;
        let stack_pointer = msp::read() as usize;
        stack_pointer.saturating_sub(stack_end)
    }

    fn average_u32(values: &[u32; RUNS_PER_EXPERIMENT]) -> u32 {
        let total: u64 = values.iter().map(|value| *value as u64).sum();
        (total / RUNS_PER_EXPERIMENT as u64) as u32
    }

    fn percentile_u32(values: &[u32; RUNS_PER_EXPERIMENT], percentile: usize) -> u32 {
        let mut scratch = *values;
        scratch.sort_unstable();
        let index = ((RUNS_PER_EXPERIMENT - 1) * percentile) / 100;
        scratch[index]
    }

    async fn write_line<'d, T: Instance + 'd>(
        class: &mut CdcAcmClass<'d, Driver<'d, T>>,
        line: &str,
    ) -> Result<(), Disconnected> {
        write_all(class, line.as_bytes()).await?;
        write_all(class, b"\r\n").await
    }

    async fn write_all<'d, T: Instance + 'd>(
        class: &mut CdcAcmClass<'d, Driver<'d, T>>,
        data: &[u8],
    ) -> Result<(), Disconnected> {
        for chunk in data.chunks(64) {
            class.write_packet(chunk).await?;
        }
        Ok(())
    }

    struct Disconnected;

    impl From<EndpointError> for Disconnected {
        fn from(error: EndpointError) -> Self {
            match error {
                EndpointError::BufferOverflow => panic!("USB endpoint buffer overflow"),
                EndpointError::Disabled => Self,
            }
        }
    }

    #[derive(Debug)]
    enum BenchmarkError {
        InferenceFailed(&'static str),
    }

    impl fmt::Display for BenchmarkError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                Self::InferenceFailed(message) => write!(f, "inference failed: {message}"),
            }
        }
    }

    fn sqrt_approx(value: f32) -> f32 {
        if value <= 0.0 {
            return 0.0;
        }

        let mut estimate = value;
        for _ in 0..6 {
            estimate = 0.5 * (estimate + value / estimate);
        }
        estimate
    }
}
