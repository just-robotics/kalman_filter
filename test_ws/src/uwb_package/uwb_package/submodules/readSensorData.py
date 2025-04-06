import serial
import time
import asyncio
from logging import *


async def readSensorData(port='/dev/ttyACM0', baudrate=115200, reconnect_interval=5, max_no_data_time=5):
    ser = serial.Serial()
    ser.baudrate = 115200
    ser.port = '/dev/ttyACM0'
    ser.open()
    ser.timeout = 1
    ser.write(b'\r\r')
    data = ser.readline()
    ser.write(b'les\n')
    while True:
        try:
            ser.write(b'\r\r')
            data = ser.readline()
            ser.write(b'les\n')
            print("Подключение к UWB датчику установлено")
            # sendLog("Подключение к UWB датчику установлено")
            last_valid_data_time = time.time()
            while True:
                if ser.in_waiting:
                    new_str = ser.readline().decode('ascii').strip()
                    if len(new_str) > 10:
                        data = str(new_str)

                        # Поиск подстроки, начинающейся с "est[" и заканчивающейся "]"
                        start_index = data.find("est[") + len("est[")  # Начало содержимого внутри скобок
                        end_index = data.find("]", start_index)       # Конец содержимого внутри скобок
                        est_content = data[start_index:end_index]
                        values = est_content.split(",")

                        x = float(values[0])
                        y = float(values[1])
                        print(f"x = {x}, y = {y}")
    
                    else:
                        print("Получены неверные данные")
                # if time.time() - last_valid_data_time > max_no_data_time:
                #     print("Долгое отсутствие валидных данных, переподключение датчика...")
                #     ser.close()
                #     ser.open()
                #     break
                await asyncio.sleep(0.1)
        except (serial.SerialException, OSError) as e:
            print(f"Ошибка соединения с датчиком: {e}, повтор через {reconnect_interval} сек...")
        await asyncio.sleep(reconnect_interval)

def run_sensor_loop():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(readSensorData(max_no_data_time=10))


run_sensor_loop()