#![allow(dead_code)]

use core::fmt::Write as _;

use embassy_stm32::Config;
use embassy_stm32::gpio::{Level, Output};
use embassy_stm32::i2c::{Error as I2cError, I2c, Master};
use embassy_stm32::mode::Blocking;
use embassy_stm32::usb::{Driver, Instance};
use embassy_time::Timer;
use embassy_usb::class::cdc_acm::CdcAcmClass;
use embassy_usb::driver::EndpointError;
use heapless::String;

pub const WHO_AM_I_REG: u8 = 0x0F;
pub const ACC_OUT_START_REG: u8 = 0x28;
pub const ACC_SCALE_M_S2_PER_LSB: f32 = 0.000_598_205_7;
pub const STREAM_INTERVAL_MS: u64 = 20;

#[derive(Clone, Copy)]
pub enum SensorKind {
    Lsm6dsv16x,
    Lsm6dso16is,
}

impl SensorKind {
    pub fn name(self) -> &'static str {
        match self {
            Self::Lsm6dsv16x => "LSM6DSV16X",
            Self::Lsm6dso16is => "LSM6DSO16IS",
        }
    }

    pub fn expected_whoami(self) -> u8 {
        match self {
            Self::Lsm6dsv16x => 0x70,
            Self::Lsm6dso16is => 0x22,
        }
    }

    pub fn from_whoami(value: u8) -> Option<Self> {
        match value {
            0x70 => Some(Self::Lsm6dsv16x),
            0x22 => Some(Self::Lsm6dso16is),
            _ => None,
        }
    }
}

#[derive(Clone, Copy)]
pub struct ProbeResult {
    pub sensor: SensorKind,
    pub address: u8,
}

pub fn clock_config() -> Config {
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

pub async fn probe_external_sensor(i2c: &mut I2c<'_, Blocking, Master>) -> Option<ProbeResult> {
    for address in [0x6A, 0x6B] {
        if let Ok(value) = read_reg(i2c, address, WHO_AM_I_REG) {
            if value == SensorKind::Lsm6dso16is.expected_whoami() {
                return Some(ProbeResult {
                    sensor: SensorKind::Lsm6dso16is,
                    address,
                });
            }
        }
    }

    None
}

pub async fn configure_sensor(
    i2c: &mut I2c<'_, Blocking, Master>,
    probe: ProbeResult,
) -> Result<(), I2cError> {
    write_reg(i2c, probe.address, 0x12, 0x01)?;
    Timer::after_millis(20).await;

    match probe.sensor {
        SensorKind::Lsm6dsv16x => {
            write_reg(i2c, probe.address, 0x12, 0x44)?;
            write_reg(i2c, probe.address, 0x11, 0x00)?;
            write_reg(i2c, probe.address, 0x17, 0x00)?;
            write_reg(i2c, probe.address, 0x10, 0x05)?;
        }
        SensorKind::Lsm6dso16is => {
            write_reg(i2c, probe.address, 0x12, 0x44)?;
            write_reg(i2c, probe.address, 0x11, 0x00)?;
            write_reg(i2c, probe.address, 0x10, 0x30)?;
        }
    }

    Timer::after_millis(100).await;
    Ok(())
}

pub fn read_acceleration(
    i2c: &mut I2c<'_, Blocking, Master>,
    probe: ProbeResult,
) -> Result<[f32; 3], I2cError> {
    let mut data = [0u8; 6];
    i2c.blocking_write_read(probe.address, &[ACC_OUT_START_REG], &mut data)?;

    let raw_x = i16::from_le_bytes([data[0], data[1]]);
    let raw_y = i16::from_le_bytes([data[2], data[3]]);
    let raw_z = i16::from_le_bytes([data[4], data[5]]);

    Ok([
        raw_x as f32 * ACC_SCALE_M_S2_PER_LSB,
        raw_y as f32 * ACC_SCALE_M_S2_PER_LSB,
        raw_z as f32 * ACC_SCALE_M_S2_PER_LSB,
    ])
}

pub fn read_reg(i2c: &mut I2c<'_, Blocking, Master>, address: u8, reg: u8) -> Result<u8, I2cError> {
    let mut value = [0u8; 1];
    i2c.blocking_write_read(address, &[reg], &mut value)?;
    Ok(value[0])
}

pub fn write_reg(
    i2c: &mut I2c<'_, Blocking, Master>,
    address: u8,
    reg: u8,
    value: u8,
) -> Result<(), I2cError> {
    i2c.blocking_write(address, &[reg, value])
}

pub fn push_fixed3<const N: usize>(buf: &mut String<N>, value: f32) {
    let scaled_f = value * 1000.0;
    let scaled = if scaled_f >= 0.0 {
        (scaled_f + 0.5) as i32
    } else {
        (scaled_f - 0.5) as i32
    };
    let sign = if scaled < 0 { "-" } else { "" };
    let abs = scaled.abs();
    write!(buf, "{}{}.{:03}", sign, abs / 1000, abs % 1000).unwrap();
}

pub async fn write_line<'d, T: Instance + 'd>(
    class: &mut CdcAcmClass<'d, Driver<'d, T>>,
    line: &str,
) -> Result<(), Disconnected> {
    write_all(class, line.as_bytes()).await?;
    write_all(class, b"\r\n").await
}

pub async fn write_all<'d, T: Instance + 'd>(
    class: &mut CdcAcmClass<'d, Driver<'d, T>>,
    data: &[u8],
) -> Result<(), Disconnected> {
    for chunk in data.chunks(64) {
        class.write_packet(chunk).await?;
    }
    Ok(())
}

pub struct Disconnected;

impl From<EndpointError> for Disconnected {
    fn from(error: EndpointError) -> Self {
        match error {
            EndpointError::BufferOverflow => panic!("USB endpoint buffer overflow"),
            EndpointError::Disabled => Self,
        }
    }
}

pub async fn report_bus_imu_candidates<'d, T: Instance + 'd>(
    class: &mut CdcAcmClass<'d, Driver<'d, T>>,
    bus_name: &str,
    i2c: &mut I2c<'_, Blocking, Master>,
) -> Result<(), Disconnected> {
    for address in [0x6A, 0x6B] {
        let mut line: String<128> = String::new();
        write!(&mut line, "{} addr=0x{:02X} ", bus_name, address).unwrap();

        match read_reg(i2c, address, WHO_AM_I_REG) {
            Ok(value) => {
                if let Some(sensor) = SensorKind::from_whoami(value) {
                    write!(&mut line, "whoami=0x{:02X} sensor={}", value, sensor.name()).unwrap();
                } else {
                    write!(&mut line, "whoami=0x{:02X} sensor=unknown", value).unwrap();
                }
            }
            Err(_) => {
                line.push_str("no response").unwrap();
            }
        }

        write_line(class, &line).await?;
    }

    Ok(())
}

pub async fn report_i2c1_candidates<'d, T: Instance + 'd>(
    class: &mut CdcAcmClass<'d, Driver<'d, T>>,
    i2c: &mut I2c<'_, Blocking, Master>,
    mcu_sel: &mut Output<'_>,
) -> Result<(), Disconnected> {
    for mux_high in [false, true] {
        mcu_sel.set_level(if mux_high { Level::High } else { Level::Low });
        Timer::after_millis(10).await;
        report_bus_imu_candidates(
            class,
            if mux_high {
                "i2c1 mux=high"
            } else {
                "i2c1 mux=low"
            },
            i2c,
        )
        .await?;
    }

    Ok(())
}
