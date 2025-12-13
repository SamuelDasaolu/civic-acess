import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { ArrowLeft, Eye, EyeOff } from "lucide-react";

const translations = {
  en: {
    signup: "Create Account",
    email: "Email Address",
    password: "Password",
    confirmPassword: "Confirm Password",
    createAccount: "Create Account",
    alreadyHaveAccount: "Already have an account?",
    login: "Log In",
    back: "Back",
    error: "Sign up failed. Please try again.",
    signupMessage: "Join CivicAccess today",
  },
  ha: {
    signup: "Ƙirƙiri Akaun",
    email: "Adireshin Email",
    password: "Kalmar sirri",
    confirmPassword: "Tabbatar Kalmar sirri",
    createAccount: "Ƙirƙiri Akaun",
    alreadyHaveAccount: "Kuna da akaun?",
    login: "Shiga",
    back: "Koma baya",
    error: "Kukarin ƙirƙiri akaun bai yi nasara ba.",
    signupMessage: "Zama CivicAccess ɗinku",
  },
  ig: {
    signup: "Mewe Akauntụ",
    email: "Ọrụ Email",
    password: "Okwu ichebe",
    confirmPassword: "Kwadebe Okwu ichebe",
    createAccount: "Mewe Akauntụ",
    alreadyHaveAccount: "Nwere akauntụ?",
    login: "Banye",
    back: "Laghachi azụ",
    error: "Mmewe akauntụ adịghị ezi.",
    signupMessage: "Zama CivicAccess ɗinku",
  },
  yo: {
    signup: "Ṣẹda Akaùnti",
    email: "Ọrọ Email",
    password: "Awo-ọrọ",
    confirmPassword: "Firisi Awo-ọrọ",
    createAccount: "Ṣẹda Akaùnti",
    alreadyHaveAccount: "O ni akaùnti?",
    login: "Wọle",
    back: "Lọ padà",
    error: "Ṣíṣẹda akaùnti kò rọ́pọ̀.",
    signupMessage: "Darapọ mọ CivicAccess loni",
  },
  pid: {
    signup: "Register",
    email: "Email Address",
    password: "Password",
    confirmPassword: "Confirm Your Password",
    createAccount: "Create Account",
    alreadyHaveAccount: "You get account already?",
    login: "Log In",
    back: "Go Back",
    error: "Register no work. Try again.",
    signupMessage: "Join CivicAccess E too Sure!!!",
  },
};

function SignUp() {
  const navigate = useNavigate();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const langCode = localStorage.getItem("language") || "en";
  const t = translations[langCode];

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");

    if (!email || !password || !confirmPassword) {
      setError("All fields are required");
      return;
    }

    if (password !== confirmPassword) {
      setError("Passwords do not match");
      return;
    }

    if (password.length < 6) {
      setError("Password must be at least 6 characters");
      return;
    }

    setLoading(true);
    const apiUrl = import.meta.env.VITE_AUTH_URL;

    try {
      const response = await fetch(`${apiUrl}/auth/signup`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          email: email.trim(),
          password: password.trim(),
        }),
      });

      const data = await response.json();

      if (response.ok) {
        
        navigate("/login", { 
          state: { 
            email: email,
            message: "Account created! Please log in with your credentials."
          } 
        });
      } else {
        setError(data.detail || t.error);
      }
    } catch (err) {
      console.error("Signup error:", err);
      setError("Network error: " + err.message);
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
                aria-hidden="true"
              >
                <path d="M36.7273 44C33.9891 44 31.6043 39.8386 30.3636 33.69C29.123 39.8386 26.7382 44 24 44C21.2618 44 18.877 39.8386 17.6364 33.69C16.3957 39.8386 14.0109 44 11.2727 44C7.25611 44 4 35.0457 4 24C4 12.9543 7.25611 4 11.2727 4C14.0109 4 16.3957 8.16144 17.6364 14.31C18.877 8.16144 21.2618 4 24 4C26.7382 4 29.123 8.16144 30.3636 14.31C31.6043 8.16144 33.9891 4 36.7273 4C40.7439 4 44 12.9543 44 24C44 35.0457 40.7439 44 36.7273 44Z" />
              </svg>
            </div>
            <h1 className="text-2xl sm:text-3xl md:text-4xl font-bold text-gray-900 mb-1 sm:mb-2">
              {t.signup}
            </h1>
            <p className="text-sm sm:text-base text-gray-600">
              {t.signupMessage}
            </p>
          </div>

         
          <div className="bg-white rounded-xl sm:rounded-2xl shadow-lg sm:shadow-xl p-4 sm:p-6 md:p-8 space-y-4 sm:space-y-5">
          
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
                {t.password}{" "}
                <span className="text-gray-500">(min 6 characters)</span>
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

           
            <div>
              <label className="block text-sm sm:text-base font-semibold text-gray-700 mb-2">
                {t.confirmPassword}
              </label>
              <div className="relative">
                <input
                  type={showConfirmPassword ? "text" : "password"}
                  value={confirmPassword}
                  onChange={(e) => setConfirmPassword(e.target.value)}
                  className="w-full px-3 py-2.5 sm:px-4 sm:py-3 border-2 border-green-200 rounded-lg focus:outline-none focus:border-green-600 focus:ring-2 focus:ring-green-100 transition-colors text-sm sm:text-base pr-12"
                  placeholder="••••••••"
                  required
                />
                <button
                  type="button"
                  onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                  className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-500 hover:text-gray-700"
                >
                  {showConfirmPassword ? (
                    <EyeOff size={20} />
                  ) : (
                    <Eye size={20} />
                  )}
                </button>
              </div>
            </div>

        
            <button
              onClick={handleSubmit}
              disabled={loading}
              className="w-full bg-green-600 text-white py-2.5 sm:py-3 md:py-3.5 rounded-lg font-semibold hover:bg-green-700 transition-colors disabled:bg-gray-400 disabled:cursor-not-allowed mt-4 sm:mt-6 text-sm sm:text-base"
            >
              {loading ? "Creating..." : t.createAccount}
            </button>
          </div>

         
          <div className="mt-4 sm:mt-6 text-center">
            <p className="text-sm sm:text-base text-gray-600">
              {t.alreadyHaveAccount}{" "}
              <button
                onClick={() => navigate("/login")}
                className="text-green-600 font-semibold hover:underline focus:outline-none focus:ring-2 focus:ring-green-100 rounded"
              >
                {t.login}
              </button>
            </p>
          </div>
        </div>
      </main>

      
      <div className="h-4 sm:h-0"></div>
    </div>
  );
}

export default SignUp;