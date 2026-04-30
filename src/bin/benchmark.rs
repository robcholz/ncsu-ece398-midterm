#![cfg_attr(all(target_arch = "arm", target_os = "none"), no_std)]
#![cfg_attr(all(target_arch = "arm", target_os = "none"), no_main)]

include!(concat!(env!("OUT_DIR"), "/benchmark_data.rs"));

#[cfg(all(target_arch = "arm", target_os = "none"))]
mod shared;

#[cfg(all(target_arch = "arm", target_os = "none"))]
mod embedded {
    use core::fmt::Write as _;
    use core::hint::black_box;
    use core::mem::MaybeUninit;

    use bytesize::ByteSize;
    use cortex_m::register::msp;
    use cortex_m_rt::heap_start;
    use embassy_executor::Spawner;
    use embassy_futures::join::join;
    use embassy_stm32::usb::{Driver, Instance};
    use embassy_stm32::{bind_interrupts, peripherals, usb};
    use embassy_time::{Instant, Timer};
    use embassy_usb::Builder;
    use embassy_usb::class::cdc_acm::{CdcAcmClass, State};
    use embedded_alloc::LlffHeap;
    use heapless::String;
    use ncsu_ece398_spring2026::model::{CHANNELS, INPUT_LEN, QuantizedCnn};
    use panic_halt as _;

    use super::{
        BENCHMARK_WINDOW_COUNT, BENCHMARK_WINDOW_SAMPLES, BENCHMARK_WINDOWS,
    };

    use super::shared::{Disconnected, clock_config, write_line};

    const WARMUP_RUNS: usize = 8;
    const INFERENCE_FREQUENCY_HZ: u32 = 100;
    const MODEL_BINARY_SIZE_BYTES: usize = model_binary_size_bytes();
    const MODEL_STATIC_RAM_BYTES: usize =
        (100 * 16) + (50 * 16) + (50 * 32) + (25 * 32) + (25 * 64) + 64 + 8192 + (64 * 4);
    const FORMAT_HEAP_BYTES: usize = 4096;

    #[global_allocator]
    static ALLOCATOR: LlffHeap = LlffHeap::empty();

    #[unsafe(link_section = ".uninit.benchmark_format_heap")]
    static mut FORMAT_HEAP: [MaybeUninit<u8>; FORMAT_HEAP_BYTES] =
        [MaybeUninit::uninit(); FORMAT_HEAP_BYTES];

    bind_interrupts!(struct Irqs {
        OTG_FS => usb::InterruptHandler<peripherals::USB_OTG_FS>;
    });

    struct StaticMetrics {
        flash_size_bytes: usize,
        model_binary_size_bytes: usize,
        model_static_ram_bytes: usize,
    }

    #[embassy_executor::main]
    async fn main(_spawner: Spawner) {
        unsafe {
            ALLOCATOR.init(
                core::ptr::addr_of_mut!(FORMAT_HEAP) as *mut u8 as usize,
                FORMAT_HEAP_BYTES,
            );
        }

        let p = embassy_stm32::init(clock_config());

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
        usb_config.product = Some("MKBOXPRO Model Benchmark");
        usb_config.serial_number = Some("mkboxpro-model-benchmark");

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
            class.wait_connection().await;
            let _ = run_benchmark(&mut class).await;
        };

        join(usb_fut, app_fut).await;
    }

    async fn run_benchmark<'d, T: Instance + 'd>(
        class: &mut CdcAcmClass<'d, Driver<'d, T>>,
    ) -> Result<(), Disconnected> {
        write_line(class, "benchmark_connected").await?;
        Timer::after_millis(100).await;
        write_line(class, "benchmark_heartbeat tick=1").await?;
        Timer::after_millis(100).await;
        write_line(class, "benchmark_heartbeat tick=2").await?;

        let mut header: String<160> = String::new();
        write!(
            &mut header,
            "benchmark_start board=STMBOXPRO model=quantized_cmsis_nn windows={} input_len={} channels={}",
            BENCHMARK_WINDOW_COUNT,
            INPUT_LEN,
            CHANNELS,
        )
        .unwrap();
        write_line(class, &header).await?;

        let mut config: String<96> = String::new();
        write!(
            &mut config,
            "benchmark_config warmup_runs={} inference_frequency_hz={}",
            WARMUP_RUNS, INFERENCE_FREQUENCY_HZ,
        )
        .unwrap();
        write_line(class, &config).await?;

        let static_metrics = static_metrics();
        let mut size_line: String<224> = String::new();
        write!(
            &mut size_line,
            "benchmark_size flash_size={} model_binary_size={} model_static_ram={}",
            readable_bytes(static_metrics.flash_size_bytes),
            readable_bytes(static_metrics.model_binary_size_bytes),
            readable_bytes(static_metrics.model_static_ram_bytes),
        )
        .unwrap();
        write_line(class, &size_line).await?;

        if let Err(err) = warmup(class).await {
            let mut line: String<96> = String::new();
            write!(&mut line, "benchmark_error={}", err).unwrap();
            write_line(class, &line).await?;
            return Ok(());
        }

        write_line(class, "benchmark_phase=measure").await?;
        let mut sequence = 0usize;
        loop {
            match measure_once(sequence) {
                Ok(sample) => write_live_sample(class, sample).await?,
                Err(err) => {
                    let mut line: String<96> = String::new();
                    write!(&mut line, "benchmark_error={}", err).unwrap();
                    write_line(class, &line).await?;
                    return Ok(());
                }
            }
            sequence = sequence.wrapping_add(1);
        }
    }

    async fn warmup<'d, T: Instance + 'd>(
        class: &mut CdcAcmClass<'d, Driver<'d, T>>,
    ) -> Result<(), &'static str> {
        if BENCHMARK_WINDOW_SAMPLES < INPUT_LEN {
            return Err("benchmark window shorter than model input");
        }

        write_line(class, "benchmark_phase=warmup")
            .await
            .map_err(|_| "usb disconnected")?;
        for run_index in 0..WARMUP_RUNS {
            let location = benchmark_window_location(run_index);
            black_box(run_benchmark_window(location).map_err(|_| "warmup inference failed")?);
        }

        Ok(())
    }

    async fn write_live_sample<'d, T: Instance + 'd>(
        class: &mut CdcAcmClass<'d, Driver<'d, T>>,
        sample: LiveSample,
    ) -> Result<(), Disconnected> {
        let mut line: String<384> = String::new();
        write!(
            &mut line,
            "benchmark_sample sequence={} window={} start={} latency_us={} free_heap_before_bytes={} free_heap_after_bytes={} stack_remaining_bytes={} prediction={}",
            sample.sequence,
            sample.window,
            sample.start,
            sample.latency_us,
            sample.free_heap_before_bytes,
            sample.free_heap_after_bytes,
            sample.stack_remaining_bytes,
            sample.prediction,
        )
        .unwrap();
        write_line(class, &line).await
    }

    struct LiveSample {
        sequence: usize,
        window: usize,
        start: usize,
        latency_us: u32,
        free_heap_before_bytes: usize,
        free_heap_after_bytes: usize,
        stack_remaining_bytes: usize,
        prediction: &'static str,
    }

    fn measure_once(sequence: usize) -> Result<LiveSample, &'static str> {
        let before_inference_free_heap_bytes = free_heap_bytes();
        let location = benchmark_window_location(sequence);

        let started = Instant::now();
        let prediction = run_benchmark_window(location).map_err(|_| "inference failed")?;
        let latency_us = started.elapsed().as_micros() as u32;
        let free_heap_after_bytes = free_heap_bytes();
        let stack_remaining_bytes = current_stack_remaining_bytes();
        black_box(prediction);

        Ok(LiveSample {
            sequence,
            window: location.window,
            start: location.start,
            latency_us,
            free_heap_before_bytes: before_inference_free_heap_bytes,
            free_heap_after_bytes,
            stack_remaining_bytes,
            prediction: prediction.class_name,
        })
    }

    #[derive(Clone, Copy)]
    struct BenchmarkWindowLocation {
        window: usize,
        start: usize,
    }

    fn benchmark_window_location(sequence: usize) -> BenchmarkWindowLocation {
        let window = sequence % BENCHMARK_WINDOW_COUNT;
        let start = if BENCHMARK_WINDOW_SAMPLES > INPUT_LEN {
            (sequence * 7) % (BENCHMARK_WINDOW_SAMPLES - INPUT_LEN)
        } else {
            0
        };
        BenchmarkWindowLocation { window, start }
    }

    fn run_benchmark_window(
        location: BenchmarkWindowLocation,
    ) -> Result<ncsu_ece398_spring2026::model::Prediction, ncsu_ece398_spring2026::model::Error>
    {
        let window = &BENCHMARK_WINDOWS[location.window];
        let start = location.start;
        QuantizedCnn::run_samples_g(&window[start..start + INPUT_LEN])
    }

    fn static_metrics() -> StaticMetrics {
        StaticMetrics {
            flash_size_bytes: flash_size_bytes(),
            model_binary_size_bytes: MODEL_BINARY_SIZE_BYTES,
            model_static_ram_bytes: MODEL_STATIC_RAM_BYTES,
        }
    }

    fn free_heap_bytes() -> usize {
        let heap_start = heap_start() as usize;
        let stack_pointer = msp::read() as usize;
        stack_pointer.saturating_sub(heap_start)
    }

    fn readable_bytes(bytes: usize) -> impl core::fmt::Display {
        ByteSize::b(bytes as u64).display().iec()
    }

    fn current_stack_remaining_bytes() -> usize {
        unsafe extern "C" {
            static _stack_end: u32;
        }

        let stack_end = core::ptr::addr_of!(_stack_end) as usize;
        let stack_pointer = msp::read() as usize;
        stack_pointer.saturating_sub(stack_end)
    }

    fn flash_size_bytes() -> usize {
        unsafe extern "C" {
            static __vector_table: u32;
            static __erodata: u32;
            static __sdata: u32;
            static __edata: u32;
        }

        let flash_start = core::ptr::addr_of!(__vector_table) as usize;
        let rodata_end = core::ptr::addr_of!(__erodata) as usize;
        let data_start = core::ptr::addr_of!(__sdata) as usize;
        let data_end = core::ptr::addr_of!(__edata) as usize;

        rodata_end.saturating_sub(flash_start) + data_end.saturating_sub(data_start)
    }

    const fn model_binary_size_bytes() -> usize {
        let conv1 = 320 + (16 * 4 * 3);
        let conv2 = 2560 + (32 * 4 * 3);
        let conv3 = 6144 + (64 * 4 * 3);
        let fc = 512 + (8 * 4);
        let normalization = 4 * 4 * 2;
        conv1 + conv2 + conv3 + fc + normalization
    }
}
