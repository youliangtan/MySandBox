import serial
import time


def drinkDispense():
    ser = serial.Serial('COM5', 9600)
    string = ""

    # send data to arduino
    # data = ser.write('U'.encode())
    data = 1
    print(data)

    if data:  # data = 1
        print(ser.readable())
        ini = ser.readline()
        ser.flush()
        print(ini.decode("ascii"))

    while True:
        string = ser.readline().decode("ascii")
        # wait for return DONE string
        if string[0] == "D":
            break
    print("end")
    return 0  # call this func to dispense 


drinkDispense()
