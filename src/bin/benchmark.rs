#![no_std]
#![no_main]

use core::fmt::Write as _;

use embassy_executor::Spawner;
use embassy_stm32::i2c::{Config as I2cConfig, I2c, Master};
use embassy_stm32::mode::Blocking;
use embassy_stm32::time::Hertz;
use embassy_stm32::usb::{Driver, Instance};
use embassy_stm32::{bind_interrupts, peripherals, usb};
use embassy_time::{Instant, Timer};
use embassy_usb::Builder;
use embassy_usb::class::cdc_acm::{CdcAcmClass, State};
use heapless::String;
use ncsu_ece398_spring2026::model::ImuWindow;
use panic_halt as _;

mod shared;

use shared::{
    Disconnected, STREAM_INTERVAL_MS, clock_config, configure_sensor, probe_external_sensor,
    read_acceleration, write_line,
};

const BENCHMARK_WINDOWS: usize = 32;

bind_interrupts!(struct Irqs {
    OTG_FS => usb::InterruptHandler<peripherals::USB_OTG_FS>;
});

#[embassy_executor::main]
async fn main(_spawner: Spawner) {
    let p = embassy_stm32::init(clock_config());

    let mut i2c_config = I2cConfig::default();
    i2c_config.frequency = Hertz::khz(100);
    let mut i2c3 = I2c::new_blocking(p.I2C3, p.PG7, p.PG8, i2c_config);

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
    usb_config.product = Some("MKBOXPRO IMU Model Benchmark");
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
        loop {
            class.wait_connection().await;
            let _ = run_benchmark(&mut class, &mut i2c3).await;
        }
    };

    embassy_futures::join::join(usb_fut, app_fut).await;
}

async fn run_benchmark<'d, T: Instance + 'd>(
    class: &mut CdcAcmClass<'d, Driver<'d, T>>,
    i2c3: &mut I2c<'_, Blocking, Master>,
) -> Result<(), Disconnected> {
    write_line(class, "MKBOXPRO model benchmark online").await?;
    let probe = match probe_external_sensor(i2c3).await {
        Some(probe) => probe,
        None => {
            write_line(class, "no external LSM6DSO16IS found on I2C3 at 0x6A/0x6B").await?;
            return Ok(());
        }
    };
    if configure_sensor(i2c3, probe).await.is_err() {
        write_line(class, "configure failed").await?;
        return Ok(());
    }

    let mut window = ImuWindow::new();
    let mut predictions = 0usize;
    let mut total_us = 0u64;

    while predictions < BENCHMARK_WINDOWS {
        Timer::after_millis(STREAM_INTERVAL_MS).await;
        let acc = match read_acceleration(i2c3, probe) {
            Ok(acc) => acc,
            Err(_) => {
                write_line(class, "read failed").await?;
                return Ok(());
            }
        };

        let started = Instant::now();
        let prediction = window.push_sample_m_s2(acc[0], acc[1], acc[2]);
        let elapsed_us = Instant::now().as_micros() - started.as_micros();

        if let Some(result) = prediction {
            total_us += elapsed_us;
            predictions += 1;

            let mut line: String<128> = String::new();
            match result {
                Ok(prediction) => {
                    write!(
                        &mut line,
                        "prediction={} class={} logit={} latency_us={}",
                        predictions,
                        prediction.class_name,
                        prediction.logits[prediction.class_index],
                        elapsed_us,
                    )
                    .unwrap();
                }
                Err(_) => {
                    write!(&mut line, "prediction={} model=error", predictions).unwrap();
                }
            }
            write_line(class, &line).await?;
        }
    }

    let mut line: String<96> = String::new();
    write!(
        &mut line,
        "benchmark_done windows={} avg_latency_us={}",
        predictions,
        total_us / BENCHMARK_WINDOWS as u64,
    )
    .unwrap();
    write_line(class, &line).await
}
