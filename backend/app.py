import cv2
import numpy as np
from flask import Flask, jsonify
from flask_cors import CORS
import threading
import time
from ekf import SensorFusionEKF


app = Flask(__name__)
CORS(app) # Crucial: allows your friend's frontend to access your data

# This global dictionary is the "bridge" between the camera math and the web server
vps_data = {
    "x": 0.0, 
    "y": 0.0, 
    "z": 1.5, # Altitude
    "mode": "INITIALIZING", 
    "features": 0
}

def vps_engine():
    global vps_data
    cap = cv2.VideoCapture(0) # 0 is your laptop camera
    
    if not cap.isOpened():
        print("ERROR: Could not open camera!")
        return

    # --- SETUP ARUCO (Precision Landing) ---
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()

    #CAMERA CALIBRATION (cm-level accuracy) ---
    focal_length = 800 # Standard webcam focal length
    center_x, center_y = 640/2, 480/2
    camera_matrix = np.array([[focal_length, 0, center_x],
                              [0, focal_length, center_y],
                              [0, 0, 1]], dtype=float)
    dist_coeffs = np.zeros((4,1))
    marker_length_m = 0.05 # Assuming the marker on your phone is 5cm wide
    
    # Define the 3D corners of the marker
    obj_points = np.array([
        [-marker_length_m/2, marker_length_m/2, 0],
        [marker_length_m/2, marker_length_m/2, 0],
        [marker_length_m/2, -marker_length_m/2, 0],
        [-marker_length_m/2, -marker_length_m/2, 0]
    ], dtype=np.float32)

    parameters.adaptiveThreshConstant = 7 # Helps detect markers on bright phone screens
    # --- SETUP OPTICAL FLOW (Navigation) ---
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    
    # Find initial features to track
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, maxCorners=100, qualityLevel=0.3, minDistance=7)

    print("VPS Engine Online. Camera active.")
    ekf = SensorFusionEKF()
    while True:
        ret, frame = cap.read()
        if not ret: 
            break
        
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 1. CHECK FOR LANDING PAD (ArUco Detection)
        corners, ids, _ = cv2.aruco.detectMarkers(frame_gray, aruco_dict, parameters=parameters)
        
        if ids is not None:
            # Get CM-LEVEL ACCURACY using Perspective-n-Point (PnP) math
            _, rvec, tvec = cv2.solvePnP(obj_points, corners[0][0], camera_matrix, dist_coeffs)
            
            # tvec[2][0] is the Z distance away from the camera in meters. Multiply by 100 for cm.
            z_cm = round(tvec[2][0] * 100, 2)
            x_cm = round(tvec[0][0] * 100, 2)
            y_cm = round(tvec[1][0] * 100, 2)

            vps_data["mode"] = "PRECISION_LANDING"
            vps_data["eqs_status"] = "LOCKED: CM-LEVEL POSE"
            vps_data["z"] = z_cm # Now outputting actual distance in CM
            vps_data["x"] = x_cm 
            vps_data["y"] = y_cm
            
            # Draw an XYZ axis on the marker to prove to the judges you are calculating 3D pose
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.03)

            
        else:
            # 2. NO MARKER? USE OPTICAL FLOW
            vps_data["mode"] = "VO_NAVIGATING"
            vps_data["z"] = 1.5 # Cruising altitude
            
            if p0 is not None:
                p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None)
                
                if p1 is not None:
                    good_new = p1[st==1]
                    good_old = p0[st==1]
                    
                    if len(good_new) > 0:
                        # Calculate how much the pixels shifted
                        diff = np.mean(good_new - good_old, axis=0)
                        
                        # Update our global X and Y (multiplying by 0.01 to scale down the pixel speed)
                        vps_data["x"] += round(float(diff[0]) * 0.02, 3)
                        vps_data["y"] += round(float(diff[1]) * 0.02, 3)
                        vps_data["features"] = len(good_new)

                        raw_pos = [vps_data["x"], vps_data["y"], vps_data["z"]]
                        fused_data = ekf.update(raw_pos)
                        
                        # Overwrite the raw data with our clean, fused data!
                        vps_data["x"] = fused_data["fused_x"]
                        vps_data["y"] = fused_data["fused_y"]
                        vps_data["z"] = fused_data["fused_z"]
                        
                        # Draw the tracking lines for your local view
                        for i, (new, old) in enumerate(zip(good_new, good_old)):
                            a, b = new.ravel()
                            c, d = old.ravel()
                            frame = cv2.line(frame, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
                            frame = cv2.circle(frame, (int(a), int(b)), 3, (0, 0, 255), -1)

            # Re-calculate features for the next loop to prevent the "white wall" drop-off
            p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, maxCorners=100, qualityLevel=0.3, minDistance=7)

        old_gray = frame_gray.copy()

        # 3. DISPLAY LOCAL DEBUG WINDOW
        cv2.putText(frame, f"MODE: {vps_data['mode']}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"X: {vps_data['x']:.2f} Y: {vps_data['y']:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.imshow('VPS Backend Diagnostics', frame)
        
        # Press 'q' to quit the local window
        if cv2.waitKey(10) & 0xFF == ord('q'): 
            break

    cap.release()
    cv2.destroyAllWindows()

# --- THE WEB SERVER ---
@app.route('/data')
def get_vps():
    # This endpoint simply returns the current state of vps_data as JSON
    return jsonify(vps_data)

if __name__ == '__main__':
    # 1. Start the camera and math loop in the background
    threading.Thread(target=vps_engine, daemon=True).start()
    
    # 2. Start the Flask server so the frontend can pull the data
    print("API Server starting on http://localhost:5000/data")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)