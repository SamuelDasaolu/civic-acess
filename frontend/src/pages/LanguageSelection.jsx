import { useNavigate } from "react-router-dom";
import Background from "../Background";

function LanguageSelection() {
  const navigate = useNavigate();

  const languages = [
    { name: "English", code: "en" },
    { name: "Hausa", code: "ha" },
    { name: "Igbo", code: "ig" },
    { name: "Yoruba", code: "yo" },
    { name: "Pidgin", code: "pid" },
  ];

  const handleLanguageSelect = (language) => {
    localStorage.setItem("language", language.code);
    localStorage.setItem("languageName", language.name);
    navigate("/login");
  };

  return (
    <div className="flex flex-col min-h-screen items-center justify-center p-4 bg-linear-to-b from-green-50 to-white">
      <Background />
      <div className="w-full max-w-md text-center relative z-10">
        <header className="mb-8">
          <div className="inline-flex items-center justify-center size-16 bg-green-100 rounded-full mb-4">
            <svg
              className="w-8 h-8 text-green-600"
              fill="currentColor"
              viewBox="0 0 48 48"
            >
              <path d="M36.7273 44C33.9891 44 31.6043 39.8386 30.3636 33.69C29.123 39.8386 26.7382 44 24 44C21.2618 44 18.877 39.8386 17.6364 33.69C16.3957 39.8386 14.0109 44 11.2727 44C7.25611 44 4 35.0457 4 24C4 12.9543 7.25611 4 11.2727 4C14.0109 4 16.3957 8.16144 17.6364 14.31C18.877 8.16144 21.2618 4 24 4C26.7382 4 29.123 8.16144 30.3636 14.31C31.6043 8.16144 33.9891 4 36.7273 4C40.7439 4 44 12.9543 44 24C44 35.0457 40.7439 44 36.7273 44Z" />
            </svg>
          </div>
          <h1 className="text-4xl font-bold text-gray-900 mb-2">CivicAccess</h1>
          <p className="text-lg text-gray-600">Your Legal First Responder</p>
          <p className="mt-3 text-gray-500 text-sm">
            Select your preferred language to continue
          </p>
        </header>

        <main className="w-full">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            {languages.map((lang) => (
              <button
                key={lang.code}
                onClick={() => handleLanguageSelect(lang)}
                className={`flex items-center justify-center p-4 h-20 bg-white border-2 border-green-200 rounded-lg shadow-sm hover:border-green-500 hover:bg-green-50 hover:shadow-md transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-green-600 focus:ring-offset-2 ${
                  lang.code === "pid" ? "sm:col-span-2" : ""
                }`}
              >
                <span className="text-lg font-semibold text-gray-800">
                  {lang.name}
                </span>
              </button>
            ))}
          </div>
        </main>

        <div className="mt-12 flex justify-center gap-1 h-1">
          <div className="flex-1 bg-green-600 rounded-full"></div>
          <div className="flex-1 bg-white border border-green-600 rounded-full"></div>
          <div className="flex-1 bg-green-600 rounded-full"></div>
        </div>
      </div>
    </div>
  );
}

export default LanguageSelection;
