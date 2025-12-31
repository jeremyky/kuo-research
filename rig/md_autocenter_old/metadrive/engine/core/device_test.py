import pygame

# pygame.init()
# pygame.joystick.init()

# if pygame.joystick.get_count() == 0:
#     print("No joysticks connected!")
# else:
#     print(pygame.joystick.get_count())
#     joystick = pygame.joystick.Joystick(0)
#     joystick.init()
#     print(f"Joystick name: {joystick.get_name()}")
    

from evdev import InputDevice, list_devices

devices = [InputDevice(path) for path in list_devices()]
for device in devices:
    print(device.name, device.path)