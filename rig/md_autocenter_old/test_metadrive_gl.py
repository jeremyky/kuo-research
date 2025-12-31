import os
import evdev
import time

def check_device_permissions(device_path):
    """Check if we have proper permissions to access the device."""
    try:
        with open(device_path, 'rb') as f:
            return True
    except PermissionError:
        print(f"\nPermission denied for {device_path}")
        print("Try running: sudo chmod a+rw " + device_path)
        return False
    except Exception as e:
        print(f"\nError accessing {device_path}: {str(e)}")
        return False

def list_input_devices():
    """List all available input devices and their capabilities."""
    devices = [evdev.InputDevice(path) for path in evdev.list_devices()]
    print("\nAvailable Input Devices:")
    print("-" * 50)
    for device in devices:
        try:
            print(f"Device: {device.name}")
            print(f"Path: {device.path}")
            print(f"Info: {device.info}")
            caps = device.capabilities()
            print("Raw capabilities:", caps)
            
            # Print detailed axis info
            if evdev.ecodes.EV_ABS in caps:
                print("\nDetailed axis information:")
                for axis, info in caps[evdev.ecodes.EV_ABS]:
                    print(f"Axis {axis}: {info}")
                    if axis == 0:  # Usually axis 0 is steering
                        print(f"  Steering axis found: {info}")
            
            # Check device permissions
            has_permission = check_device_permissions(device.path)
            print(f"Has read/write permission: {has_permission}")
            
            # Test if we can grab the device
            try:
                device.grab()
                print("Successfully grabbed device")
                device.ungrab()
            except Exception as e:
                print(f"Could not grab device: {str(e)}")
            
            print("-" * 50)
        except Exception as e:
            print(f"Error reading device {device.path}: {str(e)}")
    return devices

def monitor_wheel_input(device_path="/dev/input/event18", duration=10):
    """Monitor and print wheel input events for a specified duration."""
    try:
        device = evdev.InputDevice(device_path)
        print(f"\nMonitoring {device.name} for {duration} seconds...")
        print("Move the wheel and press buttons to see events")
        
        start_time = time.time()
        while time.time() - start_time < duration:
            try:
                events = device.read()
                for event in events:
                    print(f"Event: {event}")
            except BlockingIOError:
                # No events available
                time.sleep(0.1)
                continue
            
    except Exception as e:
        print(f"Error monitoring device: {str(e)}")

def main():
    print("\nSteering Wheel Diagnostic Tool")
    print("=" * 50)
    
    # List all input devices
    devices = list_input_devices()
    
    # Monitor the wheel if found
    wheel_path = "/dev/input/event18"
    if os.path.exists(wheel_path) and check_device_permissions(wheel_path):
        print("\nFound steering wheel, starting input monitoring...")
        monitor_wheel_input(wheel_path)
    else:
        print(f"\nSteering wheel not found at {wheel_path} or permission denied")

if __name__ == "__main__":
    main() 