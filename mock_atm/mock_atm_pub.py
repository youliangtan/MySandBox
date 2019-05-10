#!/usr/bin/env python3

# license removed for brevity
# This scripts runs a node that perform as a payload manager task
# Created by Tan You Liang, at Nov 2018
# 
# Juz a simple mock atm pub, working along side with payload controller

import rclpy
from rclpy.node import Node
import _thread
import time
import signal
import sys

from termcolor import colored

from atm_msgs.msg import PayloadCommand  # Need to compile ATM msgs package, which creates custom made msg
from atm_msgs.msg import PayloadStatus
from std_msgs.msg import Int8

class MockAtmPub(Node):
    def __init__(self):
        super().__init__('mock_atm_node')
        
        # For Communicating with ATM (AGV Task Manager)
        self.atm_pub = self.create_publisher(PayloadCommand, '/payload/command')
        self.atm_sub = self.create_subscription(PayloadStatus, '/payload/status', self.atmStatus_callback)
        self.test_sub = self.create_subscription(Int8, '/test', self.test_callback)

        # For patient mode
        self.patientID="FA,9F,D1,83"  # PARAM
        self.Payload_ID = 'rmf1234' # PARAM
        self.Task_ID = 'test123'    # PARAM
        self.door_command = 9
        self.agv_state = 9
        self.stop_flag = False
                
        try:
            print(" ------------ Running Main ------------")
            _thread.start_new_thread(self.MockPublisher, tuple())

        except:
            print("Error in thread creation!!!")
    

    def test_callback(self, msg):
        print("test callback: {} ".format(msg.data))


    def doorStatus_callback(self, msg):
        self.doorStatus = msg.data     # -1 is unknown
        print("[CALLBACK]::Door current state: {}".format(msg.data))


    # Publish Payload Status to atm&payload topic
    def pubInput(self):
        print ("preparing data: state and command: ", self.agv_state, ",", self.door_command)
        payload_command = PayloadCommand()
        payload_command.payload_id = self.Payload_ID
        payload_command.task_id = self.Task_ID
        payload_command.state = self.agv_state
        payload_command.patient_rfid = self.patientID
        payload_command.control_command = self.door_command
        
        self.atm_pub.publish(payload_command)


    def atmStatus_callback(self, msg):
        print (colored(" ------------ ATM Callback -------------- ", "red"))
        print (colored(msg, 'red'))
        print (colored(" ------------ End Callback -------------- \n", 'red'))


    # manage main pub
    def MockPublisher(self):
        print ("Running pub thread.....")
        
        while(1):
            print ("\n## =========== start new command ==========")
            self.getUserInput()
            self.pubInput()
            print ("##     !!!!! Done !!!!!!\n")
            if (self.stop_flag == True):
                print ("##     !!!!! Done !!!!!!\n")
                exit(0)
            time.sleep(1)
            

    def getUserInput(self):

        print ("\n## AGV State [ 0: moving, 1: stopped at Nurse, 2: stopped at Patient, 3: FM Controlzz ]")
        self.agv_state = int( input("## Enter Input >") )

        if (self.agv_state == 3):

            print ("\n### Door Control [ control Payload Door, 1: left open, 2: total close, 3: right open ]")
            self.door_command = int(input("## Enter Input >"))


def main(args=None):
    rclpy.init(args=args)
    node = MockAtmPub()
    print("DONE WITH INIT!!! start main")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print(" Keyboard intterupted!! End mock pub now!")
        exit(0)


if  __name__ == "__main__":
    main()

