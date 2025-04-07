def openSerialPort(ser):
    ser.close()
    ser.open()
    # time.sleep(1)
    ser.write(b'\r')
    ser.write(b'\r')
    # time.sleep(1)
    data = ser.readline()

    while (not data):
        ser.write(b'\r\r')
        print("connecting...")
    data = ser.readline()
    ser.write(b'les\r')
    status = ""
    print("Port is open")


def readSensorData(ser):
    if ser.in_waiting:
        data = ser.readline().decode('ascii').strip()
        if (data.find("est[") == -1):
            print("", end="")
            return None
        
        start_index = data.find("est[") + len("est[")
        end_index = data.find("]", start_index)
        est_content = data[start_index:end_index]
        values = est_content.split(",")

        x = float(values[0])
        y = float(values[1])
        return [x, y]


# while True:
#     data = readSerial(ser)
#     if data is not None:
#         print(data)
        