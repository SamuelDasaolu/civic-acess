import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { ArrowLeft, Mail } from "lucide-react";

const translations = {
  en: {
    forgotPassword: "Reset Your Password",
    email: "Email Address",
    sendReset: "Send Reset Link",
    backToLogin: "Back to Login",
    checkEmail: "Check your email for reset instructions",
    back: "Back",
    error: "Failed to send reset email. Please try again.",
  },
  ha: {
    forgotPassword: "Sake Saita Kalmar sirri",
    email: "Adireshin Email",
    sendReset: "Aika Hanyar Sake Saitawa",
    backToLogin: "Koma Shiga",
    checkEmail: "Duba email don umarnin sake saiti",
    back: "Koma baya",
    error: "Aika email bai yi nasara ba.",
  },
  ig: {
    forgotPassword: "Megharia Okwu ichebe",
    email: "Ọrụ Email",
    sendReset: "Zite Njikọ Megharia",
    backToLogin: "Laghachi na Nbanye",
    checkEmail: "Lelee email gị maka ntuziaka megharia",
    back: "Laghachi azụ",
    error: "Izite email megharia adịghị ezi.",
  },
  yo: {
    forgotPassword: "Tun Awo-ọrọ Rẹ Ṣe",
    email: "Ọrọ Email",
    sendReset: "Firanṣẹ Ọna asopọ Itunṣe",
    backToLogin: "Pada si Wiwole",
    checkEmail: "Ṣayẹwo email rẹ fun itọsọna atunṣe",
    back: "Lọ padà",
    error: "Firanṣẹ email atunṣe kò rọ́pọ̀.",
  },
  pid: {
    forgotPassword: "Reset Your Password",
    email: "Email Address",
    sendReset: "Send Reset Link",
    backToLogin: "Go Back to Login",
    checkEmail: "Check your email for reset link",
    back: "Go Back",
    error: "Reset email no send. Try again.",
  },
};

function ForgotPassword() {
  const navigate = useNavigate();
  const [email, setEmail] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState(false);

  const langCode = localStorage.getItem("language") || "en";
  const t = translations[langCode];

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
    setSuccess(false);

    if (!email) {
      setError("Please enter your email address");
      return;
    }

    if (!/\S+@\S+\.\S+/.test(email)) {
      setError("Please enter a valid email address");
      return;
    }

    setLoading(true);
    try {
      const response = await fetch(
        "http://localhost:8000/auth/forgot-password",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            email: email.trim(),
          }),
        }
      );

      if (response.ok) {
        setSuccess(true);

        setTimeout(() => {
          navigate("/login");
        }, 5000);
      } else {
        const data = await response.json();
        setError(data.detail || t.error);
      }
    } catch (err) {
      console.error("Forgot password error:", err);
      setError("Network error. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col min-h-screen bg-linear-to-b from-green-50 to-white">
      <header className="bg-green-600 text-white p-4">
        <button
          onClick={() => navigate("/login")}
          className="flex items-center gap-2 hover:bg-green-700 px-3 py-2 rounded-lg transition-colors"
        >
          <ArrowLeft size={20} />
          <span>{t.back}</span>
        </button>
      </header>

      <div className="flex-1 flex items-center justify-center p-4">
        <div className="w-full max-w-md">
          <div className="text-center mb-8">
            <div className="inline-flex items-center justify-center size-14 bg-green-100 rounded-full mb-4">
              <Mail className="w-7 h-7 text-green-600" />
            </div>
            <h1 className="text-3xl font-bold text-gray-900 mb-2">
              {t.forgotPassword}
            </h1>
            <p className="text-gray-600">
              Enter your email and we'll send you reset instructions
            </p>
          </div>

          <div className="bg-white rounded-xl shadow-lg p-6 space-y-4">
            {success ? (
              <div className="text-center space-y-4">
                <div className="inline-flex items-center justify-center size-12 bg-green-100 rounded-full">
                  <Mail className="w-6 h-6 text-green-600" />
                </div>
                <h3 className="text-lg font-semibold text-gray-900">
                  Check Your Email
                </h3>
                <p className="text-gray-600">{t.checkEmail}</p>
                <p className="text-sm text-gray-500">
                  Redirecting to login in 5 seconds...
                </p>
                <button
                  onClick={() => navigate("/login")}
                  className="w-full bg-green-600 text-white py-3 rounded-lg font-semibold hover:bg-green-700 transition-colors mt-4"
                >
                  {t.backToLogin}
                </button>
              </div>
            ) : (
              <>
                {error && (
                  <div className="bg-red-50 border border-red-200 text-red-800 px-4 py-3 rounded-lg text-sm">
                    {error}
                  </div>
                )}

                <div>
                  <label className="block text-sm font-semibold text-gray-700 mb-2">
                    {t.email}
                  </label>
                  <input
                    type="email"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    className="w-full px-4 py-3 border-2 border-green-200 rounded-lg focus:outline-none focus:border-green-600 focus:ring-2 focus:ring-green-100 transition-colors"
                    placeholder="you@example.com"
                    disabled={loading}
                  />
                </div>

                <button
                  onClick={handleSubmit}
                  disabled={loading}
                  className="w-full bg-green-600 text-white py-3 rounded-lg font-semibold hover:bg-green-700 transition-colors disabled:bg-gray-400 disabled:cursor-not-allowed mt-2"
                >
                  {loading ? "Sending..." : t.sendReset}
                </button>

                <div className="pt-4 border-t border-gray-200">
                  <button
                    onClick={() => navigate("/login")}
                    className="w-full text-green-600 py-2 rounded-lg font-semibold hover:bg-green-50 transition-colors"
                  >
                    ← {t.backToLogin}
                  </button>
                </div>
              </>
            )}
          </div>

          <p className="mt-6 text-center text-xs text-gray-500">
            If you don't see the email, check your spam folder
          </p>
        </div>
      </div>
    </div>
  );
}

export default ForgotPassword;
