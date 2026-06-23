"""
lidar/urg_driver.py
Hokuyo URG-04LX-UG-01 LiDAR driver.

Protocol: SCIP 2.0 over USB-CDC (appears as /dev/ttyACM0)

Commands used:
  VV  — version / device info
  PP  — sensor parameters
  MD  — multi-echo distance data (active scan mode)
  GD  — single scan acquisition
  QT  — stop scanning

Sensor specs (URG-04LX-UG-01):
  Range:          60 mm – 5,600 mm
  Angular range:  240° (−120° to +120°)
  Angular res.:   0.3515625° (1024 steps / 360°, 682 steps used)
  Scan speed:     10 Hz (100 ms / scan)
  Distance acc.:  ±30 mm

Design:
  - pyserial-asyncio for non-blocking serial I/O
  - Reconnect logic with configurable retry threshold
  - Decoded scan → dict with angles_deg, distances_m, intensities
  - latest_scan property for REST API access
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from config.config_loader import LidarConfig

logger = logging.getLogger(__name__)

# SCIP 2.0 constants for URG-04LX
_URG_STEPS_TOTAL   = 1024
_URG_FRONT_STEP    = 512           # Step index pointing straight ahead
_URG_STEPS_USED    = 682           # −120° to +120°
_URG_STEP_MIN      = 44            # First valid step
_URG_STEP_MAX      = 725           # Last valid step
_URG_MIN_RANGE_MM  = 60
_URG_MAX_RANGE_MM  = 5600
_CRLF              = b"\n"


class URGLidarDriver:
    """
    Async driver for the Hokuyo URG-04LX-UG-01.

    Usage:
        driver = URGLidarDriver(cfg)
        await driver.connect()
        scan = driver.latest_scan    # dict
        await driver.disconnect()
    """

    def __init__(self, cfg: LidarConfig) -> None:
        self._cfg          = cfg
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._connected    = False
        self._scan_task: asyncio.Task | None = None
        self._error_count  = 0
        self._scan_count   = 0
        self.latest_scan: dict[str, Any] = {
            "angles":     [],
            "distances":  [],
            "intensities": [],
            "timestamp":  0.0,
            "scan_count": 0,
        }

    async def connect(self) -> None:
        """Open serial connection and verify device identity."""
        import serial_asyncio  # type: ignore  # pyserial-asyncio

        logger.info("Connecting to LiDAR on %s @ %d baud", self._cfg.port, self._cfg.baudrate)
        self._reader, self._writer = await serial_asyncio.open_serial_connection(
            url=self._cfg.port,
            baudrate=self._cfg.baudrate,
        )
        self._connected = True

        await self._send_cmd(b"SCIP2.0")
        await asyncio.sleep(0.1)

        info = await self._get_version_info()
        logger.info("LiDAR connected: %s", info.get("model", "unknown"))

        await self._send_cmd(b"MD0044072500000")   # Start continuous scanning
        logger.info("URG-04LX scanning started.")

    async def disconnect(self) -> None:
        """Stop scanning and close the serial port."""
        if self._writer:
            try:
                await self._send_cmd(b"QT")          # Quit / stop laser
                self._writer.close()
                await self._writer.wait_closed()
            except Exception as exc:
                logger.debug("LiDAR disconnect error: %s", exc)
        self._connected = False
        logger.info("LiDAR disconnected. Total scans: %d", self._scan_count)

    async def read_scan(self) -> dict[str, Any] | None:
        """
        Read and decode one complete MD scan response.

        Returns:
            Decoded scan dict, or None on read error.
        """
        if not self._connected or self._reader is None:
            return None

        try:
            raw = await asyncio.wait_for(
                self._read_scan_block(), timeout=1.0 / self._cfg.scan_frequency_hz * 3
            )
            if raw is None:
                return None

            scan = self._decode_scan(raw)
            if scan:
                self.latest_scan = scan
                self._scan_count += 1
                self._error_count = 0
            return scan

        except TimeoutError:
            self._error_count += 1
            logger.warning("LiDAR scan timeout (count=%d)", self._error_count)
            return None
        except Exception as exc:
            self._error_count += 1
            logger.error("LiDAR read error: %s", exc)
            return None

    # ── Internal helpers ────────────────────────────────────────────────────

    async def _send_cmd(self, cmd: bytes) -> None:
        """Send a SCIP2.0 command terminated by LF."""
        if self._writer:
            self._writer.write(cmd + _CRLF)
            await self._writer.drain()

    async def _read_line(self) -> bytes:
        """Read one LF-terminated line from the serial port."""
        if self._reader is None:
            raise RuntimeError("LiDAR connection not established.")
        line = await self._reader.readline()
        return line.rstrip(b"\n\r")

    async def _read_scan_block(self) -> list[bytes] | None:
        """
        Read lines until an empty line (end of SCIP block) is found.

        A valid MD block looks like:
            MD0044072500000\n
            00P\n
            <timestamp>\n
            <data line 1>\n
            ...
            \n              ← empty line = end of block
        """
        lines: list[bytes] = []
        while True:
            line = await self._read_line()
            if line == b"":
                break
            lines.append(line)
        return lines if lines else None

    def _decode_scan(self, lines: list[bytes]) -> dict[str, Any] | None:
        """
        Decode SCIP 2.0 3-char encoded distance data.

        SCIP 2.0 uses base-64 encoding: each group of 3 ASCII chars
        encodes one 18-bit distance value in mm.
        """
        # Find the timestamp line (after command echo + status)
        data_lines: list[bytes] = []
        data_started = False
        _timestamp_raw = 0

        for i, line in enumerate(lines):
            if i == 0:
                continue   # Command echo
            if i == 1:
                continue   # Status code (00 = OK)
            if i == 2:
                # Timestamp (4-char SCIP encoded)
                try:
                    _timestamp_raw = self._decode_scip_chars(line[:4])
                except Exception:
                    pass
                data_started = True
                continue
            if data_started:
                data_lines.append(line)

        if not data_lines:
            return None

        # Concatenate all data chars (strip the checksum byte at end of each line)
        raw_data = b"".join(line[:-1] for line in data_lines)

        # Decode 3-char groups → distances in mm
        distances_mm: list[int] = []
        for idx in range(0, len(raw_data) - 2, 3):
            chunk = raw_data[idx:idx + 3]
            if len(chunk) < 3:
                break
            dist = self._decode_scip_chars(chunk)
            distances_mm.append(dist)

        if not distances_mm:
            return None

        # Map step indices to angles and filter
        angles_deg:   list[float] = []
        distances_m:  list[float] = []
        intensities:  list[float] = []   # URG-04LX does not provide intensity → fill 1.0

        step_start = _URG_STEP_MIN
        min_mm = max(self._cfg.min_range_m * 1000, _URG_MIN_RANGE_MM)
        max_mm = min(self._cfg.max_range_m * 1000, _URG_MAX_RANGE_MM)

        for step_offset, dist_mm in enumerate(distances_mm):
            step = step_start + step_offset

            # Reject invalid / out-of-range readings
            if dist_mm < min_mm or dist_mm > max_mm:
                continue

            # Convert step to angle in degrees
            angle = (step - _URG_FRONT_STEP) * 360.0 / _URG_STEPS_TOTAL

            if not (self._cfg.angle_min_deg <= angle <= self._cfg.angle_max_deg):
                continue

            angles_deg.append(round(angle, 3))
            distances_m.append(round(dist_mm / 1000.0, 4))
            intensities.append(1.0)

        return {
            "angles":      angles_deg,
            "distances":   distances_m,
            "intensities": intensities,
            "timestamp":   time.time(),
            "scan_count":  self._scan_count + 1,
        }

    @staticmethod
    def _decode_scip_chars(data: bytes) -> int:
        """
        Decode SCIP 2.0 base-64 encoded value.

        Each byte: value = (byte - 0x30)
        Accumulated MSB-first (6 bits per byte for 2/3-char, 18 bits total).
        """
        val = 0
        for byte in data:
            val = (val << 6) | (byte - 0x30)
        return val

    async def _get_version_info(self) -> dict[str, str]:
        """Query device version/info string."""
        await self._send_cmd(b"VV")
        info: dict[str, str] = {}
        try:
            for _ in range(20):
                line = await asyncio.wait_for(self._read_line(), timeout=1.0)
                if line == b"":
                    break
                if b":" in line:
                    key, _, val = line.partition(b":")
                    info[key.decode(errors="replace").strip()] = val.decode(errors="replace").strip()
        except TimeoutError:
            pass
        return info
