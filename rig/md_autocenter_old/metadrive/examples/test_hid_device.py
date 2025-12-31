import subprocess

def list_hid_devices():
    try:
        # List all USB devices
        print("Listing all HID USB devices:")
        result = subprocess.run(['lsusb'], capture_output=True, text=True, check=True)
        usb_devices = result.stdout

        # Filter for HID devices
        hid_devices = [line for line in usb_devices.splitlines() if 'HID' in line.upper()]
        for device in hid_devices:
            print(device)

        print()

        # List all input devices
        print("Listing all input devices:")
        result = subprocess.run(['xinput', 'list', '--short'], capture_output=True, text=True, check=True)
        input_devices = result.stdout
        print(input_devices)

        print()

        # Detailed information about each input device
        print("Detailed information about each input device:")
        device_ids = [line.split()[0] for line in input_devices.splitlines() if line]
        for device_id in device_ids:
            try:
                print(f"Device ID {device_id}:")
                result = subprocess.run(['xinput', 'list-props', device_id], capture_output=True, text=True, check=True)
                device_props = result.stdout
                print(device_props)
                print()
            except subprocess.CalledProcessError:
                print(f"Failed to get properties for device ID {device_id}")

    except subprocess.CalledProcessError as e:
        print(f"An error occurred while listing devices: {e}")

if __name__ == "__main__":
    list_hid_devices()
