import serial

ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)

try:
    while True:
        raw = ser.readline()      # bir satır ham bayt oku (b'\xD2\x07…\r\n' vs.)
        if not raw:
            continue
        # Sadece decimal değerler:
        print(" ".join(str(b) for b in raw))
except KeyboardInterrupt:
    pass
finally:
    ser.close()
