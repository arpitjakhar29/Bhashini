import React, { useState, useEffect, useRef } from "react";

// Constants
const API_URL = "http://127.0.0.1:5000/api";

const AITranslator = ({
  rawSigns,
  onTranslationComplete,
  apiBaseUrl = API_URL,
}) => {
  // State
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState(null);
  const [modelReady, setModelReady] = useState(false);

  // Process signs when they change
  useEffect(() => {
    // Skip processing if no signs or already processing
    if (!rawSigns.length || isProcessing) return;

    // Process the latest batch of signs
    processWithAI(rawSigns);
  }, [rawSigns]);

  // Check AI model status on mount
  useEffect(() => {
    checkAIModelStatus();
  }, []);

  // Check if the AI model is ready on the backend
  const checkAIModelStatus = async () => {
    try {
      const response = await fetch(`${apiBaseUrl}/ai-status`);
      const data = await response.json();

      if (data.success && data.ready) {
        setModelReady(true);
      } else {
        console.log("AI model not ready yet, will use basic processing");
      }
    } catch (err) {
      console.error("Could not check AI model status:", err);
      // Continue without AI enhancement if we can't reach the endpoint
    }
  };

  // Process signs with AI
  const processWithAI = async (signs) => {
    if (!signs.length) return;

    setIsProcessing(true);
    setError(null);

    try {
      const response = await fetch(`${apiBaseUrl}/process-sentence`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          signs,
          useAI: modelReady, // Tell backend whether to use AI
        }),
      });

      if (!response.ok) {
        throw new Error(`Server responded with ${response.status}`);
      }

      const data = await response.json();

      if (data.success) {
        // Call the callback with both raw and processed results
        onTranslationComplete({
          raw: signs.join(" "),
          processed: data.processed_sentence,
          confidence: data.confidence || 100,
          method: data.method || "basic", // "ai" or "basic"
        });
      } else {
        throw new Error(data.error || "Failed to process translation");
      }
    } catch (err) {
      setError(err.message);
      console.error("Error in AI processing:", err);

      // Fall back to basic processing
      onTranslationComplete({
        raw: signs.join(" "),
        processed: signs.join(" "),
        confidence: 0,
        method: "fallback",
      });
    } finally {
      setIsProcessing(false);
    }
  };

  // No UI is needed since this is a logic component
  return null;
};

export default AITranslator;
