"""
Real-Time Depth Estimation & 3D Point Cloud Generation
Uses MiDaS (small) for monocular depth estimation, then unprojects
pixels into 3D world coordinates using camera intrinsics + EKF pose.
"""

import numpy as np
import cv2
import torch
import time


class DepthEngine:
    """
    Lightweight 3D reconstruction engine using MiDaS monocular depth.
    Designed for real-time hackathon demo: ~4800 points per frame at stride=8.
    """

    def __init__(self, camera_matrix, depth_scale=5.0, stride=8):
        """
        Args:
            camera_matrix: 3x3 numpy array of camera intrinsics
            depth_scale: max depth in meters (MiDaS outputs relative depth)
            stride: pixel sampling stride (8 = every 8th pixel)
        """
        self.fx = camera_matrix[0, 0]
        self.fy = camera_matrix[1, 1]
        self.cx = camera_matrix[0, 2]
        self.cy = camera_matrix[1, 2]
        self.depth_scale = depth_scale
        self.stride = stride

        # --- Load MiDaS Small Model ---
        print("[DepthEngine] Loading MiDaS small model...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[DepthEngine] Using device: {self.device}")

        self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
        self.model.to(self.device)
        self.model.eval()

        # Load MiDaS transforms
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        self.transform = midas_transforms.small_transform

        # Pre-compute pixel grid for unprojection (will be set on first frame)
        self._pixel_grid = None
        self._grid_shape = None

        print("[DepthEngine] Model loaded successfully.")

    def estimate_depth(self, frame_bgr):
        """
        Run MiDaS inference to get a relative depth map.

        Args:
            frame_bgr: BGR image from OpenCV (H x W x 3, uint8)

        Returns:
            depth_map: H x W float32 array, normalized to [0, depth_scale] meters
        """
        # Convert BGR -> RGB for MiDaS
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Apply MiDaS transform
        input_batch = self.transform(frame_rgb).to(self.device)

        # Run inference
        with torch.no_grad():
            prediction = self.model(input_batch)

            # Resize back to original frame dimensions
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=frame_bgr.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth_map = prediction.cpu().numpy()

        # MiDaS outputs inverse depth (higher = closer). Invert and normalize.
        # Clamp to avoid division by zero
        depth_map = np.clip(depth_map, 1e-3, None)
        depth_map = 1.0 / depth_map  # Now: higher = farther

        # Normalize to [0, depth_scale] meters
        d_min, d_max = depth_map.min(), depth_map.max()
        if d_max - d_min > 1e-6:
            depth_map = (depth_map - d_min) / (d_max - d_min) * self.depth_scale
        else:
            depth_map = np.full_like(depth_map, self.depth_scale / 2)

        return depth_map.astype(np.float32)

    def _build_pixel_grid(self, h, w):
        """Pre-compute the (u, v) pixel coordinate grid for the stride-sampled image."""
        vs = np.arange(0, h, self.stride)
        us = np.arange(0, w, self.stride)
        uu, vv = np.meshgrid(us, vs)
        self._pixel_grid = (uu.flatten(), vv.flatten())
        self._grid_shape = (h, w)

    def unproject_to_pointcloud(self, frame_bgr, depth_map, pose_matrix_4x4):
        """
        Unproject sampled pixels into 3D world-space points.

        Args:
            frame_bgr: Original BGR frame for color extraction
            depth_map: H x W depth map from estimate_depth()
            pose_matrix_4x4: 4x4 numpy array [R|t] world pose of the camera

        Returns:
            dict with:
                "positions": list of [x, y, z] floats (world coordinates)
                "colors": list of [r, g, b] ints (0-255)
                "count": number of points
        """
        h, w = depth_map.shape[:2]

        # Build pixel grid if not yet computed or size changed
        if self._pixel_grid is None or self._grid_shape != (h, w):
            self._build_pixel_grid(h, w)

        us, vs = self._pixel_grid

        # Sample depth values at stride positions
        depths = depth_map[vs, us]

        # Filter out zero/near-zero depth (invalid)
        valid_mask = depths > 0.05
        us_valid = us[valid_mask]
        vs_valid = vs[valid_mask]
        depths_valid = depths[valid_mask]

        # Unproject: pixel (u,v,d) -> camera-space (X, Y, Z)
        x_cam = (us_valid.astype(np.float32) - self.cx) * depths_valid / self.fx
        y_cam = (vs_valid.astype(np.float32) - self.cy) * depths_valid / self.fy
        z_cam = depths_valid

        # Stack into Nx4 homogeneous coordinates
        n_points = len(x_cam)
        local_pts = np.stack([x_cam, y_cam, z_cam, np.ones(n_points)], axis=1)  # Nx4

        # Transform to world coordinates
        world_pts = (pose_matrix_4x4 @ local_pts.T).T[:, :3]  # Nx3

        # Extract colors from frame (BGR -> RGB)
        colors_bgr = frame_bgr[vs_valid, us_valid]  # Nx3
        colors_rgb = colors_bgr[:, ::-1]  # BGR -> RGB

        # Round for JSON serialization efficiency
        positions = np.round(world_pts, 3).tolist()
        colors = colors_rgb.tolist()

        return {
            "positions": positions,
            "colors": colors,
            "count": n_points
        }

    @staticmethod
    def build_pose_matrix(x, y, z, yaw=0.0):
        """
        Build a simple 4x4 pose matrix from EKF position + optional yaw.
        Since we don't have full rotation from the EKF, we use identity
        rotation or a simple yaw rotation around the Y axis.

        Args:
            x, y, z: EKF position in meters
            yaw: rotation around Y axis in radians (default 0)

        Returns:
            4x4 numpy pose matrix
        """
        cos_y = np.cos(yaw)
        sin_y = np.sin(yaw)

        pose = np.array([
            [cos_y,  0, sin_y, x],
            [0,      1, 0,     y],
            [-sin_y, 0, cos_y, z],
            [0,      0, 0,     1]
        ], dtype=np.float64)

        return pose
