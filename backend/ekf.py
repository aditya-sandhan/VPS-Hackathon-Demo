import numpy as np
import random

class SensorFusionEKF:
    def __init__(self):
        # Initial state: [x, y, z]
        self.fused_position = np.zeros(3)
        self.velocity = np.zeros(3)

    def generate_simulated_imu(self):
        # Hackathon trick: Since we lack hardware, we simulate IMU noise/acceleration
        # A real drone vibrates and drifts. This mimics that raw sensor noise.
        noise_x = random.uniform(-0.05, 0.05)
        noise_y = random.uniform(-0.05, 0.05)
        noise_z = random.uniform(-0.01, 0.01)
        return np.array([noise_x, noise_y, noise_z])

    def update(self, visual_pos, dt=0.1):
        """
        Fuses the highly accurate (but slow) visual position from OpenCV 
        with the fast (but noisy) simulated IMU data.
        """
        visual_array = np.array(visual_pos)
        imu_accel = self.generate_simulated_imu()

        # 1. IMU Prediction Step (Dead Reckoning)
        self.velocity += imu_accel * dt
        predicted_pos = self.fused_position + (self.velocity * dt)

        # 2. Kalman Update Step (Correcting IMU drift with Camera data)
        # We trust the camera 85%, and the IMU prediction 15%
        self.fused_position = (0.85 * visual_array) + (0.15 * predicted_pos)

        return {
            "fused_x": round(self.fused_position[0], 3),
            "fused_y": round(self.fused_position[1], 3),
            "fused_z": round(self.fused_position[2], 3)
        }