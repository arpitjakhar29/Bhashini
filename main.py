#main.py

from flask import Flask, request, jsonify
import numpy as np
import cv2
import mediapipe as mp
import os
import base64
from flask_cors import CORS
import logging
import time
import string
import threading
import language_tool_python
from collections import Counter

# Configure Flask app
app = Flask(__name__)
# Enable CORS for frontend integration
CORS(app, resources={r"/*": {"origins": ["http://localhost:5173", "http://127.0.0.1:5173"]}})

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize language tool for grammar correction
language_tool = None
language_tool_ready = False

def load_language_tool():
    global language_tool, language_tool_ready
    try:
        print("Loading LanguageTool for grammar correction...")
        language_tool = language_tool_python.LanguageToolPublicAPI('en-UK')
        language_tool_ready = True
        print("LanguageTool loaded successfully")
    except Exception as e:
        print(f"Error loading LanguageTool: {str(e)}")
        language_tool_ready = False

# Load language tool in a separate thread to avoid blocking startup
threading.Thread(target=load_language_tool).start()

# Load the model
MODEL_PATH = 'my_model.keras'
try:
    from tensorflow.keras.models import load_model
    model = load_model(MODEL_PATH)
    logger.info("Model loaded successfully from: %s", MODEL_PATH)
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    model = None

# Set the path to the data directory
PATH = os.path.join('data')

# Create an array of actions (signs) labels by listing the contents of the data directory
try:
    actions = np.array(os.listdir(PATH))
    logger.info(f"Loaded {len(actions)} actions from data directory")
except Exception as e:
    logger.error(f"Error loading actions: {str(e)}")
    actions = np.array([])

# Initialize holistic model
holistic = mp.solutions.holistic.Holistic(min_detection_confidence=0.75, min_tracking_confidence=0.75)

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

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "ok", 
        "model_loaded": model is not None,
        "actions_loaded": len(actions) > 0,
        "num_actions": len(actions),
        "grammar_tool_ready": language_tool_ready
    })

@app.route('/api/ai-status', methods=['GET'])
def ai_status():
    """Check if the grammar tool is loaded and ready"""
    return jsonify({
        'success': True,
        'ready': language_tool_ready
    })

@app.route('/api/translate-frame', methods=['POST'])
def translate_frame():
    """
    Endpoint for processing a single frame for sign language translation
    
    Expects JSON with:
    {
        "image": "base64_encoded_image"
    }
    """
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
        
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
        
    data = request.get_json()
    
    if 'image' not in data:
        return jsonify({"error": "No image data provided"}), 400
    
    try:
        # Decode the base64 image
        image_data = base64.b64decode(data['image'].split(',')[1] if ',' in data['image'] else data['image'])
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Process the image and obtain sign landmarks
        results, _ = image_process(image, holistic)
        
        # Extract keypoints
        keypoint = keypoint_extraction(results)
        
        # Return the keypoints for further processing
        return jsonify({
            "success": True,
            "keypoints": keypoint.tolist(),
            "timestamp": time.time()
        })
        
    except Exception as e:
        logger.error(f"Error processing frame: {str(e)}")
        return jsonify({"error": f"Error processing frame: {str(e)}"}), 500

# Store prediction history for stability analysis
prediction_history = []
last_prediction = None
last_detection_time = time.time()
cooldown_period = 2.0  # seconds

@app.route('/api/translate-sequence', methods=['POST'])
def translate_sequence():
    """
    Endpoint for processing a sequence of frames/keypoints for sign prediction
    """
    global prediction_history, last_prediction, last_detection_time
    
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
        
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
        
    data = request.get_json()
    
    # Check if we received keypoints or images
    if 'keypoints' in data and len(data['keypoints']) > 0:
        keypoints = np.array(data['keypoints'])
    elif 'images' in data and len(data['images']) > 0:
        # Process each image to get keypoints
        keypoints = []
        for image_data in data['images']:
            try:
                # Decode the base64 image
                img_data = base64.b64decode(image_data.split(',')[1] if ',' in image_data else image_data)
                nparr = np.frombuffer(img_data, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                # Process the image and obtain sign landmarks
                results, _ = image_process(image, holistic)
                
                # Extract keypoints
                keypoint = keypoint_extraction(results)
                keypoints.append(keypoint)
            except Exception as e:
                logger.error(f"Error processing image: {str(e)}")
                return jsonify({"error": f"Error processing image: {str(e)}"}), 500
        
        keypoints = np.array(keypoints)
    else:
        return jsonify({"error": "No keypoints or images provided"}), 400
    
    # Check if we have enough frames
    if len(keypoints) < 10:
        return jsonify({"error": f"Not enough frames. Got {len(keypoints)}, need 10"}), 400
    
    try:
        # Take the last 10 keypoints if we have more
        if len(keypoints) > 10:
            keypoints = keypoints[-10:]
        
        # Make prediction
        prediction = model.predict(keypoints[np.newaxis, :, :])
        prediction_confidence = np.amax(prediction)
        predicted_class = np.argmax(prediction)
        predicted_sign = actions[predicted_class]
        latest_keypoints = keypoints[-1]
        
        left_hand_present = False
        right_hand_present = False
        
        # Check if any hand landmarks are present in the last frame
        # This is a more direct check than using the flags
        sum_left_hand = np.sum(np.abs(latest_keypoints[:63]))  # First 63 values are left hand landmarks
        sum_right_hand = np.sum(np.abs(latest_keypoints[63:126]))  # Next 63 values are right hand landmarks
        
        # If the sum is very small, no hand is present
        left_hand_present = sum_left_hand > 0.1
        right_hand_present = sum_right_hand > 0.1
        
        # Add current prediction to history
        prediction_history.append(predicted_class)
        
        # Keep only the most recent predictions
        if len(prediction_history) > 7:
            prediction_history.pop(0)
        
        # Dynamic confidence threshold based on prediction stability
        prediction_counts = Counter(prediction_history)
        most_common = prediction_counts.most_common(1)[0]
        most_common_class = most_common[0]
        occurrence_count = most_common[1]
        
        # Adjust threshold based on prediction stability
        base_threshold = 0.85
        stability_factor = occurrence_count / len(prediction_history)
        adaptive_threshold = base_threshold * (1.0 - (stability_factor * 0.1))
        
        # Get the current time for cooldown check
        current_time = time.time()
        
        # Default response with is_reliable set to false
        response = {
            "success": True,
            "sign": actions[predicted_class],
            "confidence": float(prediction_confidence),
            "timestamp": time.time(),
            "stability": 0,
            "threshold": 0,
            "is_reliable": False
        }
        if predicted_sign == "thank you" and not (left_hand_present or right_hand_present):
            response["notes"] = "Filtered: 'thank you' detected with no hands present"
            response["is_reliable"] = False
            response["sign"] = ""
            return jsonify(response)
            
        # Check if prediction meets our criteria
        if prediction_confidence > adaptive_threshold and most_common_class == predicted_class:
            # Check cooldown period
            if current_time - last_detection_time > cooldown_period:
                response["is_reliable"] = True
                if last_prediction != actions[most_common_class]:
                    last_prediction = actions[most_common_class]
                    last_detection_time = current_time  # Reset the cooldown timer
            
        # Return the result
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        return jsonify({"error": f"Error making prediction: {str(e)}"}), 500

@app.route('/api/available-signs', methods=['GET'])
def available_signs():
    """Return list of available signs that can be recognized"""
    return jsonify({
        "success": True,
        "signs": actions.tolist()
    })

@app.route('/api/process-sentence', methods=['POST'])
def process_sentence():
    """
    Process a list of signs to form a sentence with grammar corrections
    
    Expects JSON with:
    {
        "signs": ["sign1", "sign2", "sign3", ...],
        "useAI": true/false
    }
    """
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
        
    data = request.get_json()
    
    if 'signs' not in data or not data['signs']:
        return jsonify({"error": "No signs provided"}), 400
    
    use_ai = data.get('useAI', True)
    
    try:
        # Create a copy of the signs list to avoid modifying the original
        sentence = data['signs'].copy()
        
        # Process the sentence
        # Capitalize the first word
        if sentence:
            sentence[0] = sentence[0].capitalize()
        
        # Process letters to form words
        for i in range(1, len(sentence)):
            if sentence[i] in string.ascii_lowercase or sentence[i] in string.ascii_uppercase:
                if sentence[i-1] in string.ascii_lowercase or sentence[i-1] in string.ascii_uppercase or (
                    sentence[i-1] not in ['hello', 'yes', 'no'] and 
                    sentence[i-1] not in actions.tolist() and 
                    sentence[i-1] not in list(x.capitalize() for x in actions.tolist())
                ):
                    # Combine letters to form words
                    sentence[i] = sentence[i-1] + sentence[i]
                    sentence[i-1] = ""
        
        # Remove empty strings
        sentence = [s for s in sentence if s]
        
        # Join the processed sentence
        basic_sentence = ' '.join(sentence)
        
        # Use language tool for grammar correction if available and requested
        if use_ai and language_tool_ready and language_tool:
            try:
                # Apply grammar correction
                grammar_result = language_tool.correct(basic_sentence)
                
                # Ensure first letter is capitalized
                processed_sentence = grammar_result[0].upper() + grammar_result[1:] if grammar_result else basic_sentence
                
                # Calculate simple confidence score (0-100)
                confidence = min(95, max(70, 100 - (len(sentence) * 2)))
                
                return jsonify({
                    "success": True,
                    "raw_signs": data['signs'],
                    "basic_sentence": basic_sentence,
                    "processed_sentence": processed_sentence,
                    "method": "language_tool",
                    "confidence": confidence
                })
                
            except Exception as e:
                logger.error(f"Error in language tool processing: {str(e)}")
                # Fall back to basic processing
                return jsonify({
                    "success": True,
                    "raw_signs": data['signs'],
                    "basic_sentence": basic_sentence,
                    "processed_sentence": basic_sentence,
                    "method": "basic",
                    "confidence": 0
                })
        else:
            # Return basic processing result
            return jsonify({
                "success": True, 
                "raw_signs": data['signs'],
                "basic_sentence": basic_sentence,
                "processed_sentence": basic_sentence,
                "method": "basic",
                "confidence": 0
            })
        
    except Exception as e:
        logger.error(f"Error processing sentence: {str(e)}")
        return jsonify({"error": f"Error processing sentence: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)