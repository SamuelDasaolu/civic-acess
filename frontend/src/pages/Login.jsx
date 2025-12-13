import { useState } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import { ArrowLeft, Eye, EyeOff } from "lucide-react";

const translations = {
  en: {
    login: "Welcome Back",
    email: "Email Address",
    password: "Password",
    loginBtn: "Log In",
    noAccount: "Don't have an account?",
    signup: "Sign Up",
    back: "Back",
    error: "Login failed. Please check your credentials.",
    forgotPassword: "Forgot Password?",
    loginMessage: "Login to your CivicAccess account",
  },
  ha: {
    login: "Sannu da zagaye",
    email: "Adireshin Email",
    password: "Kalmar sirri",
    loginBtn: "Shiga",
    noAccount: "Ba ku da akaun?",
    signup: "Zama Memba",
    back: "Koma baya",
    error: "Shiga bai yi nasara ba.",
    forgotPassword: "Manta Kalmar sirri?",
    loginMessage: "Shiga cikin asusun CivicAccess ɗinku",
  },
  ig: {
    login: "Nnoo azụ",
    email: "Ọrụ Email",
    password: "Okwu ichebe",
    loginBtn: "Banye",
    noAccount: "Enweghị akauntụ?",
    signup: "Debanye Aha",
    back: "Laghachi azụ",
    error: "Ibata adịghị ezi.",
    forgotPassword: "I Chefuo Okwu ichebe?",
    loginMessage: "Banye na akaụntụ CivicAccess gị",
  },
  yo: {
    login: "Kaabo pada",
    email: "Ọrọ Email",
    password: "Awo-ọrọ",
    loginBtn: "Wọle",
    noAccount: "O ko ni akaùnti?",
    signup: "Forukọsilẹ",
    back: "Lọ padà",
    error: "Wiwole kò rọ́pọ̀.",
    forgotPassword: "Gbagbe Awo-ọrọ?",
    loginMessage: "Wọle si àkọọlẹ CivicAccess rẹ",
  },
  pid: {
    login: "Welcome Back",
    email: "Email Address",
    password: "Password",
    loginBtn: "Log In",
    noAccount: "You no get account?",
    signup: "Register",
    back: "Go Back",
    error: "Login no work. Check your email and password.",
    forgotPassword: "You been Forget Password?",
    loginMessage: "Login for your CivicAccess account.",
  },
};

function Login() {
  const navigate = useNavigate();
  const location = useLocation();
  const [email, setEmail] = useState(location.state?.email || "");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [successMessage, setSuccessMessage] = useState(
    location.state?.message || ""
  );

  const langCode = localStorage.getItem("language") || "en";
  const t = translations[langCode];

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");

    if (!email || !password) {
      setError("All fields cannot be empty!");
      return;
    }

    setLoading(true);
    const apiUrl = import.meta.env.VITE_AUTH_URL;

    try {
      const response = await fetch(`${apiUrl}/auth/login`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password }),
      });

      if (response.ok) {
        const data = await response.json();
        localStorage.setItem("token", data.access_token);
        localStorage.setItem("user", JSON.stringify(data.user));
        navigate("/chatbot");
      } else {
        setError(t.error);
      }
    } catch (err) {
      console.error("Login error:", err);
      setError("Network error. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col min-h-screen bg-linear-to-b from-green-50 to-white">
      <header className="bg-green-600 text-white p-4 sm:p-5">
        <button
          onClick={() => navigate("/")}
          className="flex items-center gap-2 hover:bg-green-700 px-3 py-2 rounded-lg transition-colors text-sm sm:text-base"
        >
          <ArrowLeft className="w-5 h-5 sm:w-6 sm:h-6" />
          <span className="font-medium">{t.back}</span>
        </button>
      </header>

      <main className="flex-1 flex items-center justify-center p-4 sm:p-6 lg:p-8">
        <div className="w-full max-w-md mx-auto">
          <div className="text-center mb-6 sm:mb-8 md:mb-10">
            <div className="inline-flex items-center justify-center w-12 h-12 sm:w-14 sm:h-14 md:w-16 md:h-16 bg-green-100 rounded-full mb-3 sm:mb-4">
              <svg
                className="w-6 h-6 sm:w-7 sm:h-7 md:w-8 md:h-8 text-green-600"
                fill="currentColor"
                viewBox="0 0 48 48"
              >
                <path d="M36.7273 44C33.9891 44 31.6043 39.8386 30.3636 33.69C29.123 39.8386 26.7382 44 24 44C21.2618 44 18.877 39.8386 17.6364 33.69C16.3957 39.8386 14.0109 44 11.2727 44C7.25611 44 4 35.0457 4 24C4 12.9543 7.25611 4 11.2727 4C14.0109 4 16.3957 8.16144 17.6364 14.31C18.877 8.16144 21.2618 4 24 4C26.7382 4 29.123 8.16144 30.3636 14.31C31.6043 8.16144 33.9891 4 36.7273 4C40.7439 4 44 12.9543 44 24C44 35.0457 40.7439 44 36.7273 44Z" />
              </svg>
            </div>
            <h1 className="text-2xl sm:text-3xl md:text-4xl font-bold text-gray-900 mb-1 sm:mb-2">
              {t.login}
            </h1>
            <p className="text-sm sm:text-base text-gray-600">
              {t.loginMessage}
            </p>
          </div>

          <div className="bg-white rounded-xl sm:rounded-2xl shadow-lg sm:shadow-xl p-4 sm:p-6 md:p-8 space-y-4 sm:space-y-5">
            {successMessage && (
              <div className="bg-green-50 border border-green-200 text-green-800 px-3 py-2 sm:px-4 sm:py-3 rounded-lg text-xs sm:text-sm">
                {successMessage}
              </div>
            )}

            {error && (
              <div className="bg-red-50 border border-red-200 text-red-800 px-3 py-2 sm:px-4 sm:py-3 rounded-lg text-xs sm:text-sm">
                {error}
              </div>
            )}

            <div>
              <label className="block text-sm sm:text-base font-semibold text-gray-700 mb-2">
                {t.email}
              </label>
              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="w-full px-3 py-2.5 sm:px-4 sm:py-3 border-2 border-green-200 rounded-lg focus:outline-none focus:border-green-600 focus:ring-2 focus:ring-green-100 transition-colors text-sm sm:text-base"
                placeholder="you@example.com"
                required
              />
            </div>

            <div>
              <label className="block text-sm sm:text-base font-semibold text-gray-700 mb-2">
                {t.password}
              </label>
              <div className="relative">
                <input
                  type={showPassword ? "text" : "password"}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="w-full px-3 py-2.5 sm:px-4 sm:py-3 border-2 border-green-200 rounded-lg focus:outline-none focus:border-green-600 focus:ring-2 focus:ring-green-100 transition-colors text-sm sm:text-base pr-12"
                  placeholder="••••••••"
                  required
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-500 hover:text-gray-700"
                >
                  {showPassword ? <EyeOff size={20} /> : <Eye size={20} />}
                </button>
              </div>
            </div>

            <button
              onClick={handleSubmit}
              disabled={loading}
              className="w-full bg-green-600 text-white py-2.5 sm:py-3 md:py-3.5 rounded-lg font-semibold hover:bg-green-700 transition-colors disabled:bg-gray-400 disabled:cursor-not-allowed mt-4 sm:mt-6 text-sm sm:text-base"
            >
              {loading ? "Logging in..." : t.loginBtn}
            </button>
          </div>

          <div className="mt-4 sm:mt-6 text-center items-center justify-center flex-col gap-2">
            <p className="text-sm sm:text-base text-gray-600">
              {t.noAccount}{" "}
              <button
                onClick={() => navigate("/signup")}
                className="text-green-600 font-semibold hover:underline focus:outline-none focus:ring-2 focus:ring-green-100 rounded"
              >
                {t.signup}
              </button>
            </p>

            <p className="mt-4 text-center text-sm text-gray-600">
              <button
                onClick={() => navigate("/forgot-password")}
                className="text-green-600 font-medium hover:underline"
              >
                {t.forgotPassword}
              </button>
            </p>
          </div>
        </div>
      </main>

      <div className="h-4 sm:h-0"></div>
    </div>
  );
}

export default Login;
