#my_functions.py
import mediapipe as mp
import cv2
import numpy as np

def draw_landmarks(image, results):
    """
    Draw the landmarks on the image.

    Args:
        image (numpy.ndarray): The input image.
        results: The landmarks detected by Mediapipe.

    Returns:
        None
    """
    # Draw landmarks for left hand
    mp.solutions.drawing_utils.draw_landmarks(image, results.left_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS)
    # Draw landmarks for right hand
    mp.solutions.drawing_utils.draw_landmarks(image, results.right_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS)

def image_process(image, model):
    """
    Process the image and obtain sign landmarks.

    Args:
        image (numpy.ndarray): The input image.
        model: The Mediapipe holistic object.

    Returns:
        results: The processed results containing sign landmarks.
        image_bgr: The processed image (writeable).
    """
    # Create a copy of the image to avoid modifying the original
    image_copy = image.copy()
    
    # Set the image to read-only mode
    image_copy.flags.writeable = False
    
    # Convert the image from BGR to RGB
    image_rgb = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)
    
    # Add contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    lab[:,:,0] = clahe.apply(lab[:,:,0])
    image_rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    # Process the image using the model
    results = model.process(image_rgb)
    
    # Set the image back to writeable mode
    image_copy.flags.writeable = True
    
    # Convert the image back from RGB to BGR
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    
    # Make sure to return the writeable BGR image
    return results, image_bgr

def keypoint_extraction(results):
    """Enhanced keypoint extraction with normalization, relative positioning and hand orientation features"""
    # Extract basic keypoints
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(63)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(63)
    
    # Add hand presence flags
    lh_present = 1.0 if results.left_hand_landmarks else 0.0
    rh_present = 1.0 if results.right_hand_landmarks else 0.0
    
    # Add advanced normalization and hand orientation features
    if results.left_hand_landmarks:
        # Basic normalization relative to wrist
        wrist_x, wrist_y, wrist_z = results.left_hand_landmarks.landmark[0].x, results.left_hand_landmarks.landmark[0].y, results.left_hand_landmarks.landmark[0].z
        lh_normalized = np.array([[res.x - wrist_x, res.y - wrist_y, res.z - wrist_z] for res in results.left_hand_landmarks.landmark]).flatten()
        
        # Calculate hand orientation vectors (improves gesture recognition)
        middle_mcp = results.left_hand_landmarks.landmark[9]  # Middle finger MCP joint
        pinky_mcp = results.left_hand_landmarks.landmark[17]  # Pinky MCP joint
        hand_direction = np.array([middle_mcp.x - wrist_x, middle_mcp.y - wrist_y, middle_mcp.z - wrist_z])
        hand_normal = np.array([pinky_mcp.x - middle_mcp.x, pinky_mcp.y - middle_mcp.y, pinky_mcp.z - middle_mcp.z])
        
        # Combine features
        lh = np.concatenate([lh, lh_normalized, hand_direction, hand_normal])
    else:
        lh = np.concatenate([lh, np.zeros(63), np.zeros(3), np.zeros(3)])
    
    if results.right_hand_landmarks:
        # Basic normalization relative to wrist
        wrist_x, wrist_y, wrist_z = results.right_hand_landmarks.landmark[0].x, results.right_hand_landmarks.landmark[0].y, results.right_hand_landmarks.landmark[0].z
        rh_normalized = np.array([[res.x - wrist_x, res.y - wrist_y, res.z - wrist_z] for res in results.right_hand_landmarks.landmark]).flatten()
        
        # Calculate hand orientation vectors
        middle_mcp = results.right_hand_landmarks.landmark[9]
        pinky_mcp = results.right_hand_landmarks.landmark[17]
        hand_direction = np.array([middle_mcp.x - wrist_x, middle_mcp.y - wrist_y, middle_mcp.z - wrist_z])
        hand_normal = np.array([pinky_mcp.x - middle_mcp.x, pinky_mcp.y - middle_mcp.y, pinky_mcp.z - middle_mcp.z])
        
        # Combine features
        rh = np.concatenate([rh, rh_normalized, hand_direction, hand_normal])
    else:
        rh = np.concatenate([rh, np.zeros(63), np.zeros(3), np.zeros(3)])
    
    # Combine all features
    keypoints = np.concatenate([lh, rh, np.array([lh_present, rh_present])])
    return keypoints