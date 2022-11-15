import sys
import os
fsds_lib_path = os.path.join("~", "Formula-Student-Driverless-Simulator", "python")
sys.path.insert(0, fsds_lib_path)

import time
import fsds

# connect to the AirSim simulator 
client = fsds.FSDSClient()

# Check network connection
client.confirmConnection()

# After enabling api controll only the api can controll the car. 
# Direct keyboard and joystick into the simulator are disabled.
# If you want to still be able to drive with the keyboard while also 
# controll the car using the api, call client.enableApiControl(False)
"""client.enableApiControl(True)

# Instruct the car to go full-speed forward
car_controls = fsds.CarControls()
car_controls.throttle = 1
client.setCarControls(car_controls)


client.setCarControls(car_controls)
time.sleep(5)
state = client.getCarState()

# velocity in m/s in the car's reference frame
print(state.speed)

# nanosecond timestamp of the latest physics update
print(state.timestamp)

# position (meter) in global reference frame.
print(state.kinematics_estimated.position)

# orientation (Quaternionr) in global reference frame.
print(state.kinematics_estimated.orientation)

# m/s
print(state.kinematics_estimated.linear_velocity)

# rad/s
print(state.kinematics_estimated.angular_velocity)

# m/s^2
print(state.kinematics_estimated.linear_acceleration)

# rad/s^2
print(state.kinematics_estimated.angular_acceleration)

# Places the vehicle back at it's original position
client.reset()
"""
[image] = client.simGetImages([fsds.ImageRequest(camera_name = 'cam_left_RGB', image_type = fsds.ImageType.Scene, pixels_as_float = False, compress = True)], vehicle_name = 'FSCar')

print("Image width: ", image.width)
print("Image height: ", image.height)

# write to png 
fsds.write_file(os.path.normpath('example.png'), image.image_data_uint8)
