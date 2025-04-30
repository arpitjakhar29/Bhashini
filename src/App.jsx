//App.jsx

import React, { useState, useEffect, useRef } from "react";
import AITranslator from "./components/AITranslator";
import TranslationDisplay from "./components/TranslationDisplay";
import "./App.css";

function App() {
  // State definitions
  const [isRecording, setIsRecording] = useState(false);
  const [translatedSigns, setTranslatedSigns] = useState([]);
  const [translatedSentence, setTranslatedSentence] = useState("");
  // const [availableSigns, setAvailableSigns] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [confidence, setConfidence] = useState(0);
  const [error, setError] = useState(null);
  const [backendStatus, setBackendStatus] = useState({ status: "unknown" });
  const [framesCaptured, setFramesCaptured] = useState(0);
  const [predictionsAttempted, setPredictionsAttempted] = useState(0);
  const [darkMode, setDarkMode] = useState(false);
  const [rawSentence, setRawSentence] = useState("");
  const [isAIEnhanced, setIsAIEnhanced] = useState(false);
  const [aiConfidence, setAIConfidence] = useState(0);

  // Ref definitions - ALL refs must be defined here
  const videoRef = useRef(null);
  const mediaStreamRef = useRef(null);
  const canvasRef = useRef(null);
  const keyPointsBuffer = useRef([]);
  const animationFrameRef = useRef(null);
  const captureIntervalRef = useRef(null);
  const predictionIntervalRef = useRef(null);
  const processingRef = useRef(false);

  // Constants
  const API_BASE_URL = "http://127.0.0.1:5000/api";
  const CAPTURE_INTERVAL = 100; // Capture frame every 100ms
  const PREDICTION_INTERVAL = 1000; // Make prediction every 1000ms

  const handleTranslationComplete = (translationResult) => {
    // Update states based on AI translation results
    setRawSentence(translationResult.raw);
    setTranslatedSentence(translationResult.processed);
    setIsAIEnhanced(translationResult.method === "ai");
    setAIConfidence(translationResult.confidence);
  };

  // Initialize dark mode from localStorage
  useEffect(() => {
    const savedDarkMode = localStorage.getItem("darkMode") === "true";
    setDarkMode(savedDarkMode);

    if (savedDarkMode) {
      document.documentElement.classList.add("dark");
    } else {
      document.documentElement.classList.remove("dark");
    }
  }, []);

  // Update dark mode class when state changes
  useEffect(() => {
    if (darkMode) {
      document.documentElement.classList.add("dark");
      localStorage.setItem("darkMode", "true");
    } else {
      document.documentElement.classList.remove("dark");
      localStorage.setItem("darkMode", "false");
    }
  }, [darkMode]);
  const toggleDarkMode = () => {
    setDarkMode((prevMode) => !prevMode);
  };

  // Check backend health on component mount
  useEffect(() => {
    const checkHealth = async () => {
      try {
        console.log("Checking backend health...");
        const response = await fetch(`${API_BASE_URL}/health`);
        console.log("Health check response:", response);
        if (response.ok) {
          const data = await response.json();
          console.log("Health data:", data);
          setBackendStatus(data);
        } else {
          console.error("Health check failed with status:", response.status);
          setBackendStatus({ status: "error" });
        }
      } catch (err) {
        console.error("Health check error:", err);
        setError(
          "Failed to connect to the backend. Please make sure the server is running."
        );
        setBackendStatus({ status: "error" });
      }
    };

    checkHealth();

    // Load available signs
    // fetchAvailableSigns();

    // Clean up on unmount
    return () => {
      stopRecording();
    };
  }, []);

  // const fetchAvailableSigns = async () => {
  //   try {
  //     console.log("Fetching available signs...");
  //     const response = await fetch(`${API_BASE_URL}/available-signs`);
  //     const data = await response.json();
  //     console.log("Available signs response:", data);
  //     if (data.success) {
  //       console.log("Signs loaded:", data.signs);
  //       setAvailableSigns(data.signs);
  //     } else {
  //       console.error("Failed to load signs:", data);
  //     }
  //   } catch (err) {
  //     console.error("Failed to fetch available signs:", err);
  //   }
  // };

  const startRecording = async () => {
    try {
      // Reset state
      setError(null);
      setTranslatedSigns([]);
      setTranslatedSentence("");
      setFramesCaptured(0);
      setPredictionsAttempted(0);
      keyPointsBuffer.current = [];

      // Access webcam
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: 640,
          height: 480,
          facingMode: "user",
        },
      });

      // Store stream reference
      mediaStreamRef.current = stream;

      // Display video
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
        console.log("Video is now playing");
      }

      // Update state
      setIsRecording(true);

      // Start capture loop with a small delay to ensure state is updated
      setTimeout(() => {
        console.log("Starting capture loop after delay");
        startCaptureLoop();
      }, 100);
    } catch (err) {
      setError(
        "Failed to access camera. Please ensure you have granted camera permissions."
      );
      console.error("Error accessing camera:", err);
    }
  };

  const stopRecording = () => {
    setIsRecording(false);

    // Stop animation frame
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }

    // Clear capture interval
    if (captureIntervalRef.current) {
      clearInterval(captureIntervalRef.current);
      captureIntervalRef.current = null;
    }

    // Clear prediction interval
    if (predictionIntervalRef.current) {
      clearInterval(predictionIntervalRef.current);
      predictionIntervalRef.current = null;
    }

    // Stop and release camera
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((track) => track.stop());
      mediaStreamRef.current = null;
    }

    // Clear video source
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
  };

  const startCaptureLoop = () => {
    console.log("Starting capture loop");

    // Clear any existing intervals first
    if (captureIntervalRef.current) clearInterval(captureIntervalRef.current);
    if (predictionIntervalRef.current)
      clearInterval(predictionIntervalRef.current);

    // Set up interval for frame capture
    captureIntervalRef.current = setInterval(() => {
      captureFrame();
    }, CAPTURE_INTERVAL);

    // Set up interval for predictions
    predictionIntervalRef.current = setInterval(() => {
      console.log("Prediction interval triggered");
      console.log("Current buffer size:", keyPointsBuffer.current.length);
      if (keyPointsBuffer.current.length >= 10 && !processingRef.current) {
        processPrediction();
      } else {
        console.log(
          "Skipping prediction - need more frames or processing in progress"
        );
      }
    }, PREDICTION_INTERVAL);
  };

  const captureFrame = async () => {
    console.log("captureFrame called");

    // Only check for the refs, ignore isRecording state for now
    if (!videoRef.current || !canvasRef.current) {
      console.log("No video or canvas ref");
      return;
    }

    const video = videoRef.current;
    const canvas = canvasRef.current;

    console.log("Video state:", {
      readyState: video.readyState,
      videoWidth: video.videoWidth,
      videoHeight: video.videoHeight,
    });

    // Make sure video dimensions are available
    if (video.videoWidth === 0 || video.videoHeight === 0) {
      console.log("Video dimensions not ready");
      return;
    }

    try {
      const context = canvas.getContext("2d");

      // Set canvas size
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;

      // Draw frame
      context.drawImage(video, 0, 0, canvas.width, canvas.height);
      console.log("Frame drawn to canvas");

      // Get image data
      const imageData = canvas.toDataURL("image/jpeg", 0.8);
      console.log("Image data obtained, length:", imageData.length);

      // Send to backend
      const response = await fetch(`${API_BASE_URL}/translate-frame`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ image: imageData }),
      });

      console.log("Backend response status:", response.status);

      const data = await response.json();
      console.log("Frame processing complete:", data);

      if (data.success) {
        setFramesCaptured((prev) => prev + 1);
        keyPointsBuffer.current.push(data.keypoints);
        console.log(
          "Keypoints added to buffer, new length:",
          keyPointsBuffer.current.length
        );
      }
    } catch (err) {
      console.error("Error in captureFrame:", err);
    }
  };

  const processPrediction = async () => {
    console.log("Starting prediction process");
    console.log("Buffer length:", keyPointsBuffer.current.length);
    console.log("Processing flag:", processingRef.current);

    if (!keyPointsBuffer.current.length || processingRef.current) {
      console.log("Skipping prediction - conditions not met");
      return;
    }

    processingRef.current = true;
    setIsLoading(true);
    setPredictionsAttempted((prev) => prev + 1);

    try {
      // Take the most recent 10 frames for prediction
      const recentKeypoints = keyPointsBuffer.current.slice(-10);

      console.log("Sending keypoints to backend for prediction...");
      const response = await fetch(`${API_BASE_URL}/translate-sequence`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ keypoints: recentKeypoints }),
      });

      const data = await response.json();
      console.log("Prediction response:", data);

      if (data.success) {
        console.log("Prediction confidence:", data.confidence);
        // Only add the sign if confidence is above threshold
        if (data.confidence > 0.75) {
          // Lowered threshold for testing
          setConfidence(Math.round(data.confidence * 100));

          // Add detected sign to the list if it's not a repeat of the last sign
          setTranslatedSigns((prev) => {
            if (prev.length === 0 || prev[prev.length - 1] !== data.sign) {
              console.log("Adding new sign:", data.sign);
              const updatedSigns = [...prev, data.sign];

              // Update translated sentence
              processSentence(updatedSigns);

              return updatedSigns;
            }
            return prev;
          });
        } else {
          console.log("Prediction confidence too low:", data.confidence);
        }
      } else {
        console.error("Prediction failed:", data.error);
      }
    } catch (err) {
      console.error("Error making prediction:", err);
    } finally {
      setIsLoading(false);
      processingRef.current = false;
    }
  };

  const processSentence = async (signs) => {
    try {
      const response = await fetch(`${API_BASE_URL}/process-sentence`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ signs }),
      });

      const data = await response.json();
      console.log("Sentence processing response:", data);

      if (data.success) {
        setTranslatedSentence(data.processed_sentence);
      }
    } catch (err) {
      console.error("Error processing sentence:", err);
    }
  };

  const clearTranslation = () => {
    setTranslatedSigns([]);
    setTranslatedSentence("");
    setRawSentence("");
    setIsAIEnhanced(false);
    setAIConfidence(0);
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Sign Language Translator</h1>
        <div className="flex items-center gap-4">
          <button
            className="theme-toggle"
            onClick={toggleDarkMode}
            aria-label="Toggle dark mode"
          >
            {darkMode ? "🌙" : "☀️"}
          </button>
          <div className="status-indicator">
            Backend Status:
            <span
              className={`status ${
                backendStatus.status === "ok" ? "active" : "inactive"
              }`}
            >
              {backendStatus.status === "ok" ? "Connected" : "Disconnected"}
            </span>
          </div>
        </div>
      </header>

      <main>
        <div className="video-container">
          <video ref={videoRef} className="video-preview" playsInline muted />
          <canvas ref={canvasRef} style={{ display: "none" }} />

          {isLoading && (
            <div className="loading-indicator">
              <div className="spinner"></div>
              <p>Processing...</p>
            </div>
          )}

          {isRecording && (
            <div className="recording-indicator">
              <div className="recording-dot"></div>
              <span>Recording</span>
            </div>
          )}

          {confidence > 0 && (
            <div className="confidence-meter">
              <div
                className="confidence-bar"
                style={{ width: `${confidence}%` }}
              ></div>
              <span>{confidence}% Confidence</span>
            </div>
          )}

          {isRecording && (
            <div
              className="debug-panel"
              style={{
                position: "absolute",
                bottom: "10px",
                left: "10px",
                backgroundColor: "rgba(0,0,0,0.7)",
                color: "white",
                padding: "10px",
                borderRadius: "5px",
                fontSize: "12px",
              }}
            >
              <div>Frames: {framesCaptured}</div>
              <div>Predictions: {predictionsAttempted}</div>
              <div>Buffer: {keyPointsBuffer.current.length}</div>
            </div>
          )}
        </div>

        <div className="controls">
          {!isRecording ? (
            <button className="start-btn" onClick={startRecording}>
              Start Translating
            </button>
          ) : (
            <button className="stop-btn" onClick={stopRecording}>
              Stop Translating
            </button>
          )}
          <button
            className="clear-btn"
            onClick={clearTranslation}
            disabled={translatedSigns.length === 0}
          >
            Clear Translation
          </button>
        </div>

        <>
          {/* AI Translator component (invisible) */}
          <AITranslator
            rawSigns={translatedSigns}
            onTranslationComplete={handleTranslationComplete}
            apiBaseUrl={API_BASE_URL}
          />

          {/* Translation Display */}
          <TranslationDisplay
            translatedSentence={translatedSentence}
            rawSentence={rawSentence}
            isAIEnhanced={isAIEnhanced}
            confidence={aiConfidence}
            isLoading={isLoading}
          />

          <div className="signs-history">
            <h3>Detected Signs</h3>
            <div className="signs-list">
              {translatedSigns.length > 0 ? (
                translatedSigns.map((sign, index) => (
                  <span key={index} className="sign-chip">
                    {sign}
                  </span>
                ))
              ) : (
                <em>No signs detected</em>
              )}
            </div>
          </div>
        </>
      </main>

      {error && (
        <div className="error-banner">
          <span>{error}</span>
          <button onClick={() => setError(null)}>✕</button>
        </div>
      )}

      <footer>
        <p>Sign Language Translator • Real-time Translation</p>
      </footer>
    </div>
  );
}

export default App;
