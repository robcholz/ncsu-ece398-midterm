#![no_std]
#![no_main]

use core::fmt::Write as _;

use embassy_executor::Spawner;
use embassy_stm32::gpio::{Level, Output, Speed};
use embassy_stm32::i2c::{Config as I2cConfig, Error as I2cError, I2c, Master};
use embassy_stm32::mode::Blocking;
use embassy_stm32::time::Hertz;
use embassy_stm32::usb::{Driver, Instance};
use embassy_stm32::{Config, bind_interrupts, peripherals, usb};
use embassy_time::{Instant, Timer};
use embassy_usb::Builder;
use embassy_usb::class::cdc_acm::{CdcAcmClass, State};
use embassy_usb::driver::EndpointError;
use heapless::String;
use ncsu_ece398_midterms::model::ImuWindow;
use panic_halt as _;

const WHO_AM_I_REG: u8 = 0x0F;
const ACC_OUT_START_REG: u8 = 0x28;
const ACC_SCALE_M_S2_PER_LSB: f32 = 0.000_598_205_7;
const STREAM_INTERVAL_MS: u64 = 20;
const BENCHMARK_WINDOWS: usize = 32;

bind_interrupts!(struct Irqs {
    OTG_FS => usb::InterruptHandler<peripherals::USB_OTG_FS>;
});

#[embassy_executor::main]
async fn main(_spawner: Spawner) {
    let p = embassy_stm32::init(clock_config());
    let mut mcu_sel = Output::new(p.PI0, Level::Low, Speed::Low);

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
            let _ = run_benchmark(&mut class, &mut i2c3, &mut mcu_sel).await;
        }
    };

    embassy_futures::join::join(usb_fut, app_fut).await;
}

fn clock_config() -> Config {
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
    config
}

async fn run_benchmark<'d, T: Instance + 'd>(
    class: &mut CdcAcmClass<'d, Driver<'d, T>>,
    i2c3: &mut I2c<'_, Blocking, Master>,
    _mcu_sel: &mut Output<'_>,
) -> Result<(), Disconnected> {
    write_line(class, "MKBOXPRO model benchmark online").await?;
    let address = match probe_external_sensor(i2c3) {
        Some(address) => address,
        None => {
            write_line(class, "no external LSM6DSO16IS found on I2C3 at 0x6A/0x6B").await?;
            return Ok(());
        }
    };
    if configure_sensor(i2c3, address).await.is_err() {
        write_line(class, "configure failed").await?;
        return Ok(());
    }

    let mut window = ImuWindow::new();
    let mut predictions = 0usize;
    let mut total_us = 0u64;

    while predictions < BENCHMARK_WINDOWS {
        Timer::after_millis(STREAM_INTERVAL_MS).await;
        let acc = match read_acceleration(i2c3, address) {
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

fn probe_external_sensor(i2c: &mut I2c<'_, Blocking, Master>) -> Option<u8> {
    for address in [0x6A, 0x6B] {
        if read_reg(i2c, address, WHO_AM_I_REG).ok() == Some(0x22) {
            return Some(address);
        }
    }
    None
}

async fn configure_sensor(
    i2c: &mut I2c<'_, Blocking, Master>,
    address: u8,
) -> Result<(), I2cError> {
    write_reg(i2c, address, 0x12, 0x01)?;
    Timer::after_millis(20).await;
    write_reg(i2c, address, 0x12, 0x44)?;
    write_reg(i2c, address, 0x11, 0x00)?;
    write_reg(i2c, address, 0x10, 0x30)?;
    Timer::after_millis(100).await;
    Ok(())
}

fn read_acceleration(
    i2c: &mut I2c<'_, Blocking, Master>,
    address: u8,
) -> Result<[f32; 3], I2cError> {
    let mut data = [0u8; 6];
    i2c.blocking_write_read(address, &[ACC_OUT_START_REG], &mut data)?;
    let raw_x = i16::from_le_bytes([data[0], data[1]]);
    let raw_y = i16::from_le_bytes([data[2], data[3]]);
    let raw_z = i16::from_le_bytes([data[4], data[5]]);
    Ok([
        raw_x as f32 * ACC_SCALE_M_S2_PER_LSB,
        raw_y as f32 * ACC_SCALE_M_S2_PER_LSB,
        raw_z as f32 * ACC_SCALE_M_S2_PER_LSB,
    ])
}

fn read_reg(i2c: &mut I2c<'_, Blocking, Master>, address: u8, reg: u8) -> Result<u8, I2cError> {
    let mut value = [0u8; 1];
    i2c.blocking_write_read(address, &[reg], &mut value)?;
    Ok(value[0])
}

fn write_reg(
    i2c: &mut I2c<'_, Blocking, Master>,
    address: u8,
    reg: u8,
    value: u8,
) -> Result<(), I2cError> {
    i2c.blocking_write(address, &[reg, value])
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
