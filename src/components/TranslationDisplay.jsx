import React, { useState, useEffect } from "react";
import "./TranslationDisplay.css";

function TranslationDisplay({
  translatedSentence,
  rawSentence,
  isAIEnhanced,
  confidence,
  isLoading,
  onManualCorrection,
}) {
  const [isEditing, setIsEditing] = useState(false);
  const [editedText, setEditedText] = useState("");

  useEffect(() => {
    // Update the edited text when translation changes
    setEditedText(translatedSentence);
  }, [translatedSentence]);

  const handleEditClick = () => {
    setIsEditing(true);
  };

  const handleSaveClick = () => {
    onManualCorrection(editedText);
    setIsEditing(false);
  };

  const handleCancelClick = () => {
    setEditedText(translatedSentence);
    setIsEditing(false);
  };

  return (
    <div className="translation-display">
      <h3>Translation</h3>

      {isLoading ? (
        <div className="translation-loading">
          <div className="loading-dots">
            <span></span>
            <span></span>
            <span></span>
          </div>
          <p>Translating signs...</p>
        </div>
      ) : (
        <div className="translation-content">
          {isEditing ? (
            <div className="editing-container">
              <textarea
                value={editedText}
                onChange={(e) => setEditedText(e.target.value)}
                className="translation-editor"
                rows={4}
              />
              <div className="editing-controls">
                <button className="save-btn" onClick={handleSaveClick}>
                  Save
                </button>
                <button className="cancel-btn" onClick={handleCancelClick}>
                  Cancel
                </button>
              </div>
            </div>
          ) : (
            <>
              <p className="translated-text">
                {translatedSentence || "No translation yet"}
              </p>
              {translatedSentence && (
                <button className="edit-btn" onClick={handleEditClick}>
                  Edit Translation
                </button>
              )}
            </>
          )}

          {isAIEnhanced && translatedSentence && (
            <div className="ai-badge">
              <span className="ai-icon">🧠</span>
              <span>AI Enhanced</span>
              {confidence > 0 && (
                <span className="ai-confidence">{confidence}%</span>
              )}
            </div>
          )}

          {rawSentence && translatedSentence !== rawSentence && (
            <div className="raw-translation">
              <p>
                <small>Original: {rawSentence}</small>
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default TranslationDisplay;
