#![no_std]
#![no_main]

use core::fmt::Write as _;

use embassy_executor::Spawner;
use embassy_futures::join::join;
use embassy_stm32::gpio::{Level, Output, Speed};
use embassy_stm32::i2c::{Config as I2cConfig, Error as I2cError, I2c, Master};
use embassy_stm32::mode::Blocking;
use embassy_stm32::time::Hertz;
use embassy_stm32::usb::{Driver, Instance};
use embassy_stm32::{bind_interrupts, peripherals, usb};
use embassy_time::{Instant, Timer};
use embassy_usb::Builder;
use embassy_usb::class::cdc_acm::{CdcAcmClass, State};
use heapless::String;
use panic_halt as _;

mod shared;

use shared::{
    Disconnected, ProbeResult, STREAM_INTERVAL_MS, clock_config, configure_sensor,
    probe_external_sensor, push_fixed3, read_acceleration, report_bus_imu_candidates,
    report_i2c1_candidates, write_line,
};

const CALIBRATION_SAMPLES: usize = 128;

bind_interrupts!(struct Irqs {
    OTG_FS => usb::InterruptHandler<peripherals::USB_OTG_FS>;
});

#[embassy_executor::main]
async fn main(_spawner: Spawner) {
    let p = embassy_stm32::init(clock_config());

    let mut mcu_sel = Output::new(p.PI0, Level::Low, Speed::Low);

    let mut i2c1_config = I2cConfig::default();
    i2c1_config.frequency = Hertz::khz(100);
    let mut i2c1 = I2c::new_blocking(p.I2C1, p.PB6, p.PB7, i2c1_config);

    let mut i2c2_config = I2cConfig::default();
    i2c2_config.frequency = Hertz::khz(100);
    let mut i2c2 = I2c::new_blocking(p.I2C2, p.PB13, p.PB14, i2c2_config);

    let mut i2c3_config = I2cConfig::default();
    i2c3_config.frequency = Hertz::khz(100);
    let mut i2c3 = I2c::new_blocking(p.I2C3, p.PG7, p.PG8, i2c3_config);

    let mut i2c4_config = I2cConfig::default();
    i2c4_config.frequency = Hertz::khz(100);
    let mut i2c4 = I2c::new_blocking(p.I2C4, p.PD12, p.PD13, i2c4_config);

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
    usb_config.product = Some("MKBOXPRO IMU Console");
    usb_config.serial_number = Some("mkboxpro-imu");

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
            let _ = run_console(
                &mut class,
                &mut i2c1,
                &mut i2c2,
                &mut i2c3,
                &mut i2c4,
                &mut mcu_sel,
            )
            .await;
        }
    };

    join(usb_fut, app_fut).await;
}

async fn run_console<'d, T: Instance + 'd>(
    class: &mut CdcAcmClass<'d, Driver<'d, T>>,
    i2c1: &mut I2c<'_, Blocking, Master>,
    i2c2: &mut I2c<'_, Blocking, Master>,
    i2c3: &mut I2c<'_, Blocking, Master>,
    i2c4: &mut I2c<'_, Blocking, Master>,
    mcu_sel: &mut Output<'_>,
) -> Result<(), Disconnected> {
    write_line(class, "MKBOXPRO USB CDC online").await?;
    write_line(
        class,
        "board i2c ports: I2C1=PB6/PB7 internal sensor bus, I2C2=PB13/PB14 NFC/ST25, I2C3=PG7/PG8 external DIL24, I2C4=PD12/PD13 auxiliary",
    )
    .await?;

    loop {
        write_line(class, "i2c1 pins=PB6/PB7 role=internal sensor bus").await?;
        report_i2c1_candidates(class, i2c1, mcu_sel).await?;

        write_line(class, "i2c2 pins=PB13/PB14 role=nfc/st25 bus").await?;
        report_bus_imu_candidates(class, "i2c2", i2c2).await?;

        write_line(class, "i2c3 pins=PG7/PG8 role=external DIL24 bus").await?;
        report_bus_imu_candidates(class, "i2c3", i2c3).await?;

        write_line(class, "i2c4 pins=PD12/PD13 role=auxiliary bus").await?;
        report_bus_imu_candidates(class, "i2c4", i2c4).await?;

        let probe = match probe_external_sensor(i2c3).await {
            Some(probe) => probe,
            None => {
                write_line(
                    class,
                    "external probe failed: no LSM6DSO16IS found on I2C3 (PG7/PG8) at 0x6A/0x6B; refusing to fall back to internal I2C1 LSM6DSV16X",
                )
                .await?;
                Timer::after_millis(1000).await;
                continue;
            }
        };

        let mut info: String<96> = String::new();
        write!(
            &mut info,
            "using external imu={} bus=i2c3 pins=PG7/PG8 addr=0x{:02X}",
            probe.sensor.name(),
            probe.address,
        )
        .unwrap();
        write_line(class, &info).await?;

        if let Err(err) = configure_sensor(i2c3, probe).await {
            let mut line: String<96> = String::new();
            write!(&mut line, "configure failed: {:?}", err).unwrap();
            write_line(class, &line).await?;
            Timer::after_millis(1000).await;
            continue;
        }

        write_line(class, "calibrating: keep the board still for ~3s").await?;
        let bias = match calibrate_bias(i2c3, probe).await {
            Ok(bias) => bias,
            Err(err) => {
                let mut line: String<96> = String::new();
                write!(&mut line, "calibration failed: {:?}", err).unwrap();
                write_line(class, &line).await?;
                Timer::after_millis(1000).await;
                continue;
            }
        };

        write_line(
            class,
            "streaming: acc is bias-corrected, velocity is integrated drift-prone",
        )
        .await?;

        let mut velocity = [0.0f32; 3];
        let mut last_sample = Instant::now();

        loop {
            Timer::after_millis(STREAM_INTERVAL_MS).await;

            let now = Instant::now();
            let dt = (now.as_micros() - last_sample.as_micros()) as f32 * 1e-6;
            last_sample = now;

            let raw_acc = match read_acceleration(i2c3, probe) {
                Ok(values) => values,
                Err(err) => {
                    let mut line: String<96> = String::new();
                    write!(&mut line, "read failed: {:?}; reprobing", err).unwrap();
                    write_line(class, &line).await?;
                    break;
                }
            };

            let acc = [
                raw_acc[0] - bias[0],
                raw_acc[1] - bias[1],
                raw_acc[2] - bias[2],
            ];

            velocity[0] += acc[0] * dt;
            velocity[1] += acc[1] * dt;
            velocity[2] += acc[2] * dt;

            let mut line: String<160> = String::new();
            line.push_str("acc=[").unwrap();
            push_fixed3(&mut line, acc[0]);
            line.push(',').unwrap();
            push_fixed3(&mut line, acc[1]);
            line.push(',').unwrap();
            push_fixed3(&mut line, acc[2]);
            line.push_str("] velocity=[").unwrap();
            push_fixed3(&mut line, velocity[0]);
            line.push(',').unwrap();
            push_fixed3(&mut line, velocity[1]);
            line.push(',').unwrap();
            push_fixed3(&mut line, velocity[2]);
            line.push_str("]").unwrap();

            write_line(class, &line).await?;
        }
    }
}

async fn calibrate_bias(
    i2c: &mut I2c<'_, Blocking, Master>,
    probe: ProbeResult,
) -> Result<[f32; 3], I2cError> {
    let mut sum = [0.0f32; 3];

    for _ in 0..CALIBRATION_SAMPLES {
        let sample = read_acceleration(i2c, probe)?;
        sum[0] += sample[0];
        sum[1] += sample[1];
        sum[2] += sample[2];
        Timer::after_millis(STREAM_INTERVAL_MS).await;
    }

    Ok([
        sum[0] / CALIBRATION_SAMPLES as f32,
        sum[1] / CALIBRATION_SAMPLES as f32,
        sum[2] / CALIBRATION_SAMPLES as f32,
    ])
}
