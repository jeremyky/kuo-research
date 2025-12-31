import pygame

# Initialize Pygame
pygame.init()

# Initialize the joystick module
pygame.joystick.init()

# Get the number of joysticks connected
num_joysticks = pygame.joystick.get_count()

# Check if there are any joysticks connected
if num_joysticks == 0:
    print("No joysticks found.")
else:
    print(f"Found {num_joysticks} joystick(s):")
    
    # Iterate over each joystick
    for i in range(num_joysticks):
        # Get the joystick instance
        joystick = pygame.joystick.Joystick(i)
        
        # Initialize the joystick
        joystick.init()
        
        # Get joystick details
        name = joystick.get_name()
        num_axes = joystick.get_numaxes()
        num_buttons = joystick.get_numbuttons()
        num_hats = joystick.get_numhats()
        
        # Print joystick details
        print(f"Joystick {i}:")
        print(f"  Name: {name}")
        print(f"  Number of Axes: {num_axes}")
        print(f"  Number of Buttons: {num_buttons}")
        print(f"  Number of Hats: {num_hats}")
        
        # Clean up joystick
        joystick.quit()

# Quit Pygame
pygame.quit()
