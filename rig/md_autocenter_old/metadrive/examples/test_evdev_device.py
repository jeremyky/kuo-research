from evdev import list_devices, InputDevice

def list_hid_devices():
    print("Listing all HID devices:")
    devices = list_devices()
    if not devices:
        print("No HID devices found.")
        return

    for device_path in devices:
        device = InputDevice(device_path)
        print(f"Device Path: {device.path}")
        print(f"Name: {device.name}")
        print(f"Phys: {device.phys}")
        # print(f"Unique ID: {device.info.id}")
        print(f"Bus Type: {device.info.bustype}")
        print(f"Vendor ID: {device.info.vendor}")
        print(f"Product ID: {device.info.product}")
        print()

if __name__ == "__main__":
    list_hid_devices()
