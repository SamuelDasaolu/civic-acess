import { useEffect, useCallback } from "react";
import { X } from "lucide-react";

function GradingModal({ message, isOpen, onClose }) {
  const getScoreColor = useCallback((score) => {
    if (score === null || isNaN(score)) return "bg-gray-100 border-gray-300";
    if (score >= 90) return "bg-green-100 border-green-300";
    if (score >= 70) return "bg-yellow-100 border-yellow-300";
    return "bg-red-100 border-red-300";
  }, []);

  const getScoreBadgeColor = useCallback((score) => {
    if (score === null || isNaN(score)) return "bg-gray-500 text-white";
    if (score >= 90) return "bg-green-500 text-white";
    if (score >= 70) return "bg-yellow-500 text-white";
    return "bg-red-500 text-white";
  }, []);

  const getScoreLabel = useCallback((score) => {
    if (score === null || isNaN(score)) return "Score Unavailable";
    if (score >= 90) return "Excellent";
    if (score >= 70) return "Good";
    return "Needs Improvement";
  }, []);

  const score = message
    ? message.judge_score !== null
      ? message.judge_score
      : "N/A"
    : "N/A";
  const reason = message
    ? message.judge_reason || "No evaluation reason provided."
    : "No evaluation reason provided.";

  const translatedQuery = message
    ? message.translatedQuery || message.text
    : "N/A";

  useEffect(() => {
    const handleEscape = (event) => {
      if (event.key === "Escape") {
        onClose();
      }
    };

    if (isOpen) {
      document.addEventListener("keydown", handleEscape);
    }

    return () => {
      document.removeEventListener("keydown", handleEscape);
    };
  }, [isOpen, onClose]);

  if (!isOpen || !message) return null;

  return (
    <div
      className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50"
      onClick={onClose}
    >
      <div
        className={`bg-white rounded-lg shadow-xl max-w-md w-full transform transition-all ${getScoreColor(
          score
        )} border-2`}
        role="dialog"
        aria-modal="true"
        aria-labelledby="modal-title"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between p-6 border-b border-gray-200">
          <h2 id="modal-title" className="text-xl font-bold text-gray-800">
            Response Evaluation
          </h2>
          <button
            onClick={onClose}
            className="text-gray-500 hover:text-gray-700 transition-colors p-1 rounded-full hover:bg-gray-100"
            aria-label="Close evaluation modal"
          >
            <X size={24} />
          </button>
        </div>

        {message.translatedQuery &&
          message.translatedQuery !== message.text && (
            <div className="bg-gray-50 p-6 border-b border-gray-200">
              <p className="text-sm font-semibold text-gray-700 mb-1">
                Translated Query (for legal search):
              </p>
              <p className="text-sm italic text-gray-600">
                "{translatedQuery}"
              </p>
            </div>
          )}

        <div className="p-6 space-y-6">
          <div className="flex items-center gap-4">
            <div
              className={`${getScoreBadgeColor(
                score
              )} rounded-full w-20 h-20 flex items-center justify-center text-2xl font-bold`}
            >
              {score}
            </div>
            <div>
              <p className="text-gray-600 text-sm font-semibold">JUDGE SCORE</p>
              <p className="text-gray-500 text-xs mt-1">
                {getScoreLabel(score)}
              </p>
            </div>
          </div>

          <div>
            <label className="block text-sm font-semibold text-gray-700 mb-2">
              Evaluation Reason
            </label>
            <div
              className="bg-white border border-gray-300 rounded-lg p-3 max-h-40 overflow-y-auto"
              tabIndex="0"
            >
              <p className="text-sm text-gray-700 leading-relaxed whitespace-pre-wrap">
                {reason}
              </p>
            </div>
          </div>
        </div>

        <div className="bg-gray-50 px-6 py-4 rounded-b-lg border-t border-gray-200">
          <button
            onClick={onClose}
            className="w-full bg-gray-300 hover:bg-gray-400 text-gray-800 font-semibold py-2 px-4 rounded transition-colors focus:outline-none focus:ring-2 focus:ring-gray-400"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
}

export default GradingModal;
