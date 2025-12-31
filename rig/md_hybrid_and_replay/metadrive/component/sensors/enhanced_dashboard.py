from panda3d.core import NodePath, PGTop, TextNode, CardMaker, Vec3, OrthographicLens

from metadrive.component.sensors.base_sensor import BaseSensor
from metadrive.constants import CamMask
from metadrive.engine.core.image_buffer import ImageBuffer


class EnhancedDashBoard(ImageBuffer, BaseSensor):
    """
    Enhanced Dashboard for showing raw steering wheel input values and detailed numerical displays
    """
    def perceive(self, *args, **kwargs):
        """
        This is only used for GUI and won't provide any observation result
        """
        raise NotImplementedError

    PARA_VIS_LENGTH = 12
    PARA_VIS_HEIGHT = 1
    MAX_SPEED = 120
    BUFFER_W = 2
    BUFFER_H = 1
    CAM_MASK = CamMask.PARA_VIS
    GAP = 4.1
    TASK_NAME = "update panel"

    def __init__(self, engine, *, cuda):
        if engine.win is None:
            return
        self.aspect2d_np = NodePath(PGTop("aspect2d"))
        self.aspect2d_np.show(self.CAM_MASK)
        self.para_vis_np = {}
        self.value_text_np = {}  # For numerical value displays

        tmp_node_path_list = []
        # make_buffer_func, make_camera_func = engine.win.makeTextureBuffer, engine.makeCamera

        # don't delete the space in word, it is used to set a proper position
        for i, np_name in enumerate(["Steering", " Throttle", "     Brake", "    Speed"]):
            text = TextNode(np_name)
            text.setText(np_name)
            text.setSlant(0.1)
            textNodePath = self.aspect2d_np.attachNewNode(text)
            tmp_node_path_list.append(textNodePath)

            textNodePath.setScale(0.052)
            text.setFrameColor(0, 0, 0, 1)
            text.setTextColor(0, 0, 0, 1)
            text.setFrameAsMargin(-self.GAP, self.PARA_VIS_LENGTH, 0, 0)
            text.setAlign(TextNode.ARight)
            textNodePath.setPos(-1.125111, 0, 0.9 - i * 0.08)
            if i != 0:
                cm = CardMaker(np_name)
                cm.setFrame(0, self.PARA_VIS_LENGTH - 0.21, -self.PARA_VIS_HEIGHT / 2 + 0.1, self.PARA_VIS_HEIGHT / 2)
                cm.setHasNormals(True)
                card = textNodePath.attachNewNode(cm.generate())
                tmp_node_path_list.append(card)

                card.setPos(0.21, 0, 0.22)
                self.para_vis_np[np_name.lstrip()] = card
            else:
                # left
                name = "Left"
                cm = CardMaker(name)
                cm.setFrame(
                    0, (self.PARA_VIS_LENGTH - 0.4) / 2, -self.PARA_VIS_HEIGHT / 2 + 0.1, self.PARA_VIS_HEIGHT / 2
                )
                cm.setHasNormals(True)
                card = textNodePath.attachNewNode(cm.generate())
                tmp_node_path_list.append(card)

                card.setPos(0.2 + self.PARA_VIS_LENGTH / 2, 0, 0.22)
                self.para_vis_np[name] = card
                # right
                name = "Right"
                cm = CardMaker(np_name)
                cm.setFrame(
                    -(self.PARA_VIS_LENGTH - 0.1) / 2, 0, -self.PARA_VIS_HEIGHT / 2 + 0.1, self.PARA_VIS_HEIGHT / 2
                )
                cm.setHasNormals(True)
                card = textNodePath.attachNewNode(cm.generate())
                tmp_node_path_list.append(card)

                card.setPos(0.2 + self.PARA_VIS_LENGTH / 2, 0, 0.22)
                self.para_vis_np[name] = card
        
        # Add numerical value displays
        self._create_value_displays(tmp_node_path_list)
        
        super(EnhancedDashBoard, self).__init__(
            self.BUFFER_W, self.BUFFER_H, self.BKG_COLOR, parent_node=self.aspect2d_np, engine=engine
        )
        self.origin = NodePath("EnhancedDashBoard")
        self._node_path_list.extend(tmp_node_path_list)

    def _create_value_displays(self, tmp_node_path_list):
        """Create numerical value displays for steering, throttle, brake, and speed"""
        # Position values to the right of the visual bars
        value_positions = [
            ("Steering", -0.85, 0.9),      # Steering angle in degrees
            ("Throttle", -0.85, 0.82),     # Throttle percentage
            ("Brake", -0.85, 0.74),        # Brake percentage  
            ("Speed", -0.85, 0.66),        # Speed in km/h
            ("RawSteer", -0.85, 0.58),     # Raw steering input
            ("RawThrottle", -0.85, 0.50),  # Raw throttle input
            ("RawBrake", -0.85, 0.42),     # Raw brake input
        ]
        
        for name, x, y in value_positions:
            # Create text node for the value
            text = TextNode(f"{name}_value")
            text.setText("0.0")  # Initial value
            text.setSlant(0.1)
            textNodePath = self.aspect2d_np.attachNewNode(text)
            tmp_node_path_list.append(textNodePath)
            
            textNodePath.setScale(0.045)  # Slightly smaller than labels
            text.setFrameColor(0, 0, 0, 0.8)  # Semi-transparent background
            text.setTextColor(1, 1, 1, 1)  # White text
            text.setFrameAsMargin(-0.5, 0.5, -0.1, 0.1)
            text.setAlign(TextNode.ACenter)
            textNodePath.setPos(x, 0, y)
            
            self.value_text_np[name] = textNodePath

    def _create_camera(self, parent_node, bkg_color):
        """
        Create orthogonal camera for the buffer
        """
        self.cam = cam = self.engine.makeCamera(self.buffer, clearColor=bkg_color)
        cam.node().setCameraMask(self.CAM_MASK)

        self.cam.reparentTo(parent_node)
        self.cam.setPos(Vec3(-0.9, -1.01, 0.78))

    def update_vehicle_state(self, vehicle):
        """
        Update the dashboard result given a vehicle - shows RAW values from steering wheel
        """
        # Store vehicle reference for raw value access
        self.vehicle = vehicle
        
        # Use raw values if available, otherwise fall back to processed values
        if (hasattr(vehicle, 'raw_steering') and hasattr(vehicle, 'raw_throttle') and hasattr(vehicle, 'raw_brake') 
            and vehicle.raw_steering is not None):
            steering = vehicle.raw_steering
            # Convert raw throttle/brake to throttle_brake format (positive = throttle, negative = brake)
            throttle_brake = vehicle.raw_throttle - vehicle.raw_brake
        else:
            steering, throttle_brake = vehicle.steering, vehicle.throttle_brake
        
        speed = vehicle.speed_km_h
        
        # Update visual bars with raw values
        if throttle_brake < 0:
            self.para_vis_np["Throttle"].setScale(0, 1, 1)
            self.para_vis_np["Brake"].setScale(-throttle_brake, 1, 1)
        elif throttle_brake > 0:
            self.para_vis_np["Throttle"].setScale(throttle_brake, 1, 1)
            self.para_vis_np["Brake"].setScale(0, 1, 1)
        else:
            self.para_vis_np["Throttle"].setScale(0, 1, 1)
            self.para_vis_np["Brake"].setScale(0, 1, 1)

        steering_value = abs(steering)
        if steering < 0:
            self.para_vis_np["Left"].setScale(steering_value, 1, 1)
            self.para_vis_np["Right"].setScale(0, 1, 1)
        elif steering > 0:
            self.para_vis_np["Right"].setScale(steering_value, 1, 1)
            self.para_vis_np["Left"].setScale(0, 1, 1)
        else:
            self.para_vis_np["Right"].setScale(0, 1, 1)
            self.para_vis_np["Left"].setScale(0, 1, 1)
        speed_value = speed / self.MAX_SPEED
        self.para_vis_np["Speed"].setScale(speed_value, 1, 1)
        
        # Update numerical value displays
        self._update_value_displays(steering, throttle_brake, speed)

    def _update_value_displays(self, steering, throttle_brake, speed):
        """Update the numerical value displays"""
        # Convert steering to degrees (assuming max steering is ±1.0, max wheel angle is 35 degrees)
        steering_degrees = steering * 35.0
        
        # Calculate throttle and brake percentages
        if throttle_brake > 0:
            throttle_pct = throttle_brake * 100
            brake_pct = 0.0
        else:
            throttle_pct = 0.0
            brake_pct = abs(throttle_brake) * 100
        
        # Update text displays
        if "Steering" in self.value_text_np:
            self.value_text_np["Steering"].node().setText(f"{steering_degrees:.1f}°")
        
        if "Throttle" in self.value_text_np:
            self.value_text_np["Throttle"].node().setText(f"{throttle_pct:.0f}%")
        
        if "Brake" in self.value_text_np:
            self.value_text_np["Brake"].node().setText(f"{brake_pct:.0f}%")
        
        if "Speed" in self.value_text_np:
            self.value_text_np["Speed"].node().setText(f"{speed:.1f} km/h")
        
        # Update raw value displays if available
        if hasattr(self, 'vehicle') and self.vehicle:
            if hasattr(self.vehicle, 'raw_steering') and "RawSteer" in self.value_text_np:
                raw_steer_degrees = self.vehicle.raw_steering * 35.0
                self.value_text_np["RawSteer"].node().setText(f"Raw: {raw_steer_degrees:.1f}°")
            
            if hasattr(self.vehicle, 'raw_throttle') and "RawThrottle" in self.value_text_np:
                raw_throttle_pct = self.vehicle.raw_throttle * 100
                self.value_text_np["RawThrottle"].node().setText(f"Raw: {raw_throttle_pct:.0f}%")
            
            if hasattr(self.vehicle, 'raw_brake') and "RawBrake" in self.value_text_np:
                raw_brake_pct = self.vehicle.raw_brake * 100
                self.value_text_np["RawBrake"].node().setText(f"Raw: {raw_brake_pct:.0f}%")

    def remove_display_region(self):
        self.buffer.set_active(False)
        super(EnhancedDashBoard, self).remove_display_region()

    def add_display_region(self, display_region, keep_height=False):
        super(EnhancedDashBoard, self).add_display_region(display_region, False)
        self.buffer.set_active(True)
        self.origin.reparentTo(self.aspect2d_np)

    def destroy(self):
        super(EnhancedDashBoard, self).destroy()
        for para in self.para_vis_np.values():
            para.removeNode()
        for text_np in self.value_text_np.values():
            text_np.removeNode()
        self.aspect2d_np.removeNode()

    def track(self, *args, **kwargs):
        # for compatibility
        pass
