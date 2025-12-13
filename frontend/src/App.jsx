import { Routes, Route, Navigate } from "react-router-dom";
import LanguageSelection from "./pages/LanguageSelection";
import SignUp from "./pages/SignUp";
import Login from "./pages/Login";
import Chatbot from "./pages/Chatbot";
import ForgotPassword from "./pages/ForgotPassword";

function ProtectedRoute({ children }) {
  const token = localStorage.getItem("token");

  if (!token) {
    return <Navigate to="/" replace />;
  }

  return children;
}

function App() {
  return (
    <Routes>
      <Route path="/" element={<LanguageSelection />} />
      <Route path="/signup" element={<SignUp />} />
      <Route path="/login" element={<Login />} />
      <Route
        path="/chatbot"
        element={
          <ProtectedRoute>
            <Chatbot />
          </ProtectedRoute>
        }
      />
      <Route path="*" element={<Navigate to="/" replace />} />
      <Route path="/forgot-password" element={<ForgotPassword />} />
    </Routes>
  );
}

export default App;
