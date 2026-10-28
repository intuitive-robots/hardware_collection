"""Change the bus baudrate (and return delay time) of all GELLO servos.

The GELLO default of 57600 baud limits the leader to ~30 samples/s (one sync read of 8
servos is ~25 ms on the wire). At 1 Mbps a read takes a few ms. Both values live in the
servos' EEPROM, so this is a one-time change; everything that opens the GELLO port must use
the new baudrate afterwards (gello node --baudrate, gello_get_offset.py --baudrate).

Only X-series servos (XL330, XL430, XC330, XM/XH430, ...) are supported: Baud Rate is
register 8 and Return Delay Time register 9 on all of them. Torque must be off (it is for GELLO).

    python hardware_collection/gello/scripts/gello_set_baudrate.py --from-baud 57600 --to-baud 1000000
    # undo:
    python hardware_collection/gello/scripts/gello_set_baudrate.py --from-baud 1000000 --to-baud 57600
"""

import time
from dataclasses import dataclass
from typing import Tuple

import tyro

from hardware_collection.gello.third_party.DynamixelSDK.python.src.dynamixel_sdk.packet_handler import PacketHandler
from hardware_collection.gello.third_party.DynamixelSDK.python.src.dynamixel_sdk.port_handler import PortHandler
from hardware_collection.gello.third_party.DynamixelSDK.python.src.dynamixel_sdk.robotis_def import COMM_SUCCESS

ADDR_MODEL_NUMBER = 0
ADDR_BAUD_RATE = 8
ADDR_RETURN_DELAY_TIME = 9
ADDR_TORQUE_ENABLE = 64
# X-series Baud Rate register values
BAUD_VALUES = {9600: 0, 57600: 1, 115200: 2, 1000000: 3, 2000000: 4, 3000000: 5, 4000000: 6}
# model numbers of X-series servos sharing the control table layout used here
X_SERIES_MODELS = {
    1190: "XL330-M077", 1200: "XL330-M288", 1240: "XC330-T181", 1230: "XC330-T288",
    1220: "XC330-M288", 1210: "XC330-M181", 1060: "XL430-W250", 1090: "2XL430-W250",
    1070: "XC430-W150", 1080: "XC430-W240", 1160: "2XC430-W250", 1020: "XM430-W350",
    1030: "XM430-W210", 1120: "XM540-W270", 1130: "XM540-W150", 1010: "XH430-W350",
    1000: "XH430-W210",
}


@dataclass
class Args:
    port: str = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT94EVRT-if00-port0"
    ids: Tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8)
    """Servo IDs (GELLO: 7 joints + gripper)."""
    from_baud: int = 57600
    """Baudrate the servos currently use."""
    to_baud: int = 1000000
    return_delay_us: int = 0
    """Return Delay Time in us (servo default 500). 0 is safe for a single-master bus."""


def _open(port: str, baud: int) -> PortHandler:
    ph = PortHandler(port)
    if not ph.openPort() or not ph.setBaudRate(baud):
        raise RuntimeError(f"Could not open {port} at {baud}")
    return ph


def _ping_all(ph: PortHandler, pk, ids) -> dict:
    found = {}
    for dxl_id in ids:
        model, res, err = pk.ping(ph, dxl_id)
        if res == COMM_SUCCESS and err == 0:
            found[dxl_id] = model
    return found


def main(args: Args) -> None:
    if args.to_baud not in BAUD_VALUES:
        raise SystemExit(f"--to-baud must be one of {sorted(BAUD_VALUES)}")
    pk = PacketHandler(2.0)

    ph = _open(args.port, args.from_baud)
    found = _ping_all(ph, pk, args.ids)
    missing = [i for i in args.ids if i not in found]
    if missing:
        ph.closePort()
        raise SystemExit(f"No answer at {args.from_baud} baud from IDs {missing} (found {sorted(found)}); nothing written")
    unknown = {i: m for i, m in found.items() if m not in X_SERIES_MODELS}
    if unknown:
        ph.closePort()
        raise SystemExit(f"Unsupported (non X-series) model numbers {unknown}; nothing written")
    for dxl_id, model in found.items():
        torque, res, err = pk.read1ByteTxRx(ph, dxl_id, ADDR_TORQUE_ENABLE)
        if res != COMM_SUCCESS or torque != 0:
            ph.closePort()
            raise SystemExit(f"ID {dxl_id}: torque is on or unreadable, EEPROM is write-protected; nothing written")
        print(f"ID {dxl_id}: {X_SERIES_MODELS[model]}")

    # Return delay first (still talking at the old baudrate), then baudrate: the servo switches
    # as soon as the baudrate write is acknowledged.
    for dxl_id in args.ids:
        res, err = pk.write1ByteTxRx(ph, dxl_id, ADDR_RETURN_DELAY_TIME, args.return_delay_us // 2)
        if res != COMM_SUCCESS or err != 0:
            raise RuntimeError(f"ID {dxl_id}: writing Return Delay Time failed ({res}, {err})")
    for dxl_id in args.ids:
        res, err = pk.write1ByteTxRx(ph, dxl_id, ADDR_BAUD_RATE, BAUD_VALUES[args.to_baud])
        # the acknowledgement may already be sent at the new rate, so only report here
        print(f"ID {dxl_id}: baudrate written (response {res}, {err})")
    ph.closePort()
    time.sleep(0.5)

    ph = _open(args.port, args.to_baud)
    found = _ping_all(ph, pk, args.ids)
    for dxl_id in found:
        baud, _, _ = pk.read1ByteTxRx(ph, dxl_id, ADDR_BAUD_RATE)
        delay, _, _ = pk.read1ByteTxRx(ph, dxl_id, ADDR_RETURN_DELAY_TIME)
        print(f"ID {dxl_id}: answers at {args.to_baud}, baud register {baud}, return delay {2 * delay} us")
    ph.closePort()
    missing = [i for i in args.ids if i not in found]
    if missing:
        raise SystemExit(
            f"IDs {missing} do not answer at {args.to_baud}. Rerun with --from-baud {args.from_baud} "
            f"--to-baud {args.to_baud} --ids {' '.join(map(str, missing))} or scan with Dynamixel Wizard."
        )
    print(f"All {len(found)} servos now run at {args.to_baud} baud.")


if __name__ == "__main__":
    main(tyro.cli(Args))
