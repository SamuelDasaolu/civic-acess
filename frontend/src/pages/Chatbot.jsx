import { useState, useEffect, useRef } from "react";
import {
  Send,
  LogOut,
  Globe,
  ChevronDown,
  Plus,
  CheckCircle,
  Clock,
} from "lucide-react";
import { FaGavel } from "react-icons/fa";
import GradingModal from "../GradingModal";
import Footer from "../Footer";

const translations = {
  en: {
    online: "online",
    selectIssue: "What are you dealing with today?",
    policeEncounter: "Police Encounter",
    landlordTenant: "Landlord/Tenant",
    describeIssue:
      "Please describe your situation, and I'll explain your rights.",
    typeMessage: "Type a message...",
    logout: "Logout",
    civicAccess: "CivicAccess",
    simpleExplanation: "Simple Explanation",
    legalProof: "Legal Proof",
    error: "Sorry I encountered an error. Please try again later.",

    tooManyRequests: "Please wait a moment",
    rateLimitMessage:
      "We're processing many requests. Please wait about 1 minute before checking the grade again.",
    checkingGrade: "Checking grade...",
    gradeAvailable: "Grade available",
    tryAgainSoon: "Try again soon",
    gradeTimeout: "Grade check timed out. Try again later.",
  },
  ha: {
    online: "online",
    selectIssue: "Me kake aiki? Zabar zaɓi.",
    policeEnounter: "Haɗin Jiya Polisi",
    landlordTenant: "Mai Gida/Masu Sulka",
    describeIssue: "Bayyana halin ku, in nadenga hakkinku.",
    typeMessage: "Rubuta saƙo...",
    logout: "Fita",
    civicAccess: "CivicAccess",
    simpleExplanation: "Sauƙaƙƙe Bayani",
    legalProof: "Shaida ta Shari'a",
    error: "Yi hakuri na ci karo da kuskure don Allah a sake gwadawa daga baya",

    tooManyRequests: "Da fatan za a jira",
    rateLimitMessage:
      "Muna sarrafa buƙatun da yawa. Da fatan za a jira minti ɗaya kafin duba maki.",
    checkingGrade: "Ana duba maki...",
    gradeAvailable: "Maki suna samuwa",
    tryAgainSoon: "Gwada kuma nan da nan",
    gradeTimeout: "An dakatar da duba maki. Gwada kuma daga baya.",
  },
  ig: {
    online: "online",
    selectIssue: "Gini ka ruo? Jiri ahụ zaɓi.",
    policeEncounter: "Njikọ Ndị Uwe Ogbako",
    landlordTenant: "Onye Ụlọ/Ndị Na-Aru Ụlọ",
    describeIssue: "Kọwaa ọnọdụ gị, m ga-akọwaa ikike gị.",
    typeMessage: "Dee ozi...",
    logout: "Pụta",
    civicAccess: "CivicAccess",
    simpleExplanation: "Nkọwahụ Dị Mfe",
    legalProof: "Ihe Akaaka Iwu",
    error: "ndo ahụrụ m njehie biko nwaa ọzọ ma emechaa",

    tooManyRequests: "Biko chere",
    rateLimitMessage:
      "Anyị na-ahazi ọtụtụ arịrịọ. Biko chere ihe dịka nkeji tupu ịlele ọkwa ọzọ.",
    checkingGrade: "A na-enyocha ọkwa...",
    gradeAvailable: "Ọkwa dị",
    tryAgainSoon: "Nwaa ọzọ n'oge na-adịghị anya",
    gradeTimeout: "Oge nlele ọkwa agwụla. Nwaa ọzọ ma emechaa.",
  },
  yo: {
    online: "online",
    selectIssue: "Kini o ti lo de? Yan eto kan.",
    policeEncounter: "Isokan Onigun Ijinna",
    landlordTenant: "Onigbowu/Awọn Alabere",
    describeIssue: "Ṣalaye ipo rẹ, emi a ṣalaye ẹtọ rẹ.",
    typeMessage: "Kọ ozi kan...",
    logout: "Jade",
    civicAccess: "CivicAccess",
    simpleExplanation: "Alaye ti o rọrun",
    legalProof: "Ẹri Ofin",
    error: "ma binu mo pade aṣiṣe kan jọwọ gbiyanju lẹẹkansi nigbamii",

    tooManyRequests: "Jọwọ duro",
    rateLimitMessage:
      "A nṣe iṣẹ ọpọlọpọ ibeere. Jọwọ duro iye àádọ́ta ìṣẹ́jú kí o lè ṣe àyẹ̀wò iwọn ọlẹ.",
    checkingGrade: "A nṣe àyẹ̀wò iwọn ọlẹ...",
    gradeAvailable: "Iwọn ọlẹ wa",
    tryAgainSoon: "Gbiyanju lẹẹkansi laipe",
    gradeTimeout: "Àyẹ̀wò iwọn ọlẹ ti pẹ́. Gbiyanju lẹẹkansi laipe.",
  },
  pid: {
    online: "online",
    selectIssue: "Wetin dey sup? Tell me make I help you.",
    policeEncounter: "Police Encounter",
    landlordTenant: "Landlord/Tenant",
    describeIssue: "Tell me your situation, I go explain your right.",
    typeMessage: "Type your message...",
    logout: "Comot",
    civicAccess: "CivicAccess",
    simpleExplanation: "Simple Explanation",
    legalProof: "Legal Proof",
    error: "Omor, error dey here shuu. Abeg try am again!",
    tooManyRequests: "Abeg wait small",
    rateLimitMessage:
      "We dey process plenty request. Abeg wait for about 1 minute before you check your grade again.",
    checkingGrade: "Dey check grade...",
    gradeAvailable: "Grade don ready",
    tryAgainSoon: "Try again soon",
    gradeTimeout: "Grade check don tire. Try again later.",
  },
};

function Chatbot() {
  const langCode = localStorage.getItem("language") || "en";
  const langName = localStorage.getItem("languageName") || "English";
  const t = translations[langCode];

  useEffect(() => {
    setMessages((prev) => {
      if (prev.length === 0) return prev;
      const newFirstMessage = { ...prev[0], text: t.selectIssue };
      return [newFirstMessage, ...prev.slice(1)];
    });
  }, [langCode, t]);

  const [messages, setMessages] = useState(() => {
    return [
      {
        id: 1,
        text: t.selectIssue,
        sender: "bot",
        type: "text",
        judge_score: null,
        judge_reason: null,
        pollingStatus: "complete",
      },
    ];
  });

  const [input, setInput] = useState("");
  const [showLanguageDropdown, setShowLanguageDropdown] = useState(false);
  const [loading, setLoading] = useState(false);
  const [selectedMessage, setSelectedMessage] = useState(null);
  const [isGradingModalOpen, setIsGradingModalOpen] = useState(false);

  const [rateLimitCooldown, setRateLimitCooldown] = useState(false);
  const [rateLimitMessage, setRateLimitMessage] = useState("");

  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleLanguageChange = (code, name) => {
    localStorage.setItem("language", code);
    localStorage.setItem("languageName", name);
    setShowLanguageDropdown(false);
  };

  const handleSendMessage = async () => {
    if (input.trim() === "") return;

    const userMessage = {
      id: messages.length + 1,
      text: input,
      sender: "user",
      type: "text",
      judge_score: null,
      judge_reason: null,
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setLoading(true);
    setRateLimitCooldown(false);
    setRateLimitMessage("");

    try {
      const languageMap = {
        en: "english",
        ha: "hausa",
        ig: "igbo",
        yo: "yoruba",
        pid: "pidgin",
      };
      const BACKEND_URL = import.meta.env.VITE_BACKEND_URL;

      const response = await fetch(`${BACKEND_URL}/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          message: userMessage.text,
          language: languageMap[langCode] || "english",
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      console.log("Backend response:", data);

      const newBotMessageId = messages.length + 2;

      const translatedQuery =
        data.debug_info?.translated_query || userMessage.text;

      const botMessage = {
        id: newBotMessageId,
        text: data.response,
        sender: "bot",
        type: "text",
        judge_score: null,
        judge_reason: null,
        pollingStatus: "checking",
        translatedQuery: translatedQuery,
      };

      setMessages((prev) => [...prev, botMessage]);

      startPollingForGrade(newBotMessageId);
    } catch (error) {
      console.error("Error fetching response:", error);
      const errorMessage = {
        id: messages.length + 2,
        text: t.error,
        sender: "bot",
        type: "text",
        judge_score: null,
        judge_reason: null,
        pollingStatus: "error",
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  const startPollingForGrade = (messageId) => {
    const maxAttempts = 10;
    let attempts = 0;
    const BACKEND_URL = import.meta.env.VITE_BACKEND_URL;
    let cooldownInterval = null;

    const poll = async () => {
      if (rateLimitCooldown) {
        console.log("Rate limit cooldown active, skipping poll attempt.");
        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === messageId
              ? { ...msg, pollingStatus: "rate_limited" }
              : msg
          )
        );
        return;
      }

      try {
        attempts++;
        console.log(
          `Polling attempt ${attempts}/${maxAttempts} for message ${messageId}`
        );

        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === messageId
              ? { ...msg, pollingStatus: `checking_${attempts}` }
              : msg
          )
        );

        const logsResponse = await fetch(`${BACKEND_URL}/logs`);

        if (!logsResponse.ok) {
          if (logsResponse.status === 429) {
            console.log("Rate limited (429). Too many requests.");

            setMessages((prev) =>
              prev.map((msg) =>
                msg.id === messageId
                  ? { ...msg, pollingStatus: "rate_limited" }
                  : msg
              )
            );

            setRateLimitCooldown(true);
            setRateLimitMessage(t.rateLimitMessage);

            let cooldownTime = 60;
            cooldownInterval = setInterval(() => {
              cooldownTime--;
              if (cooldownTime <= 0) {
                clearInterval(cooldownInterval);
                setRateLimitCooldown(false);
                setRateLimitMessage("");

                setMessages((prev) =>
                  prev.map((msg) =>
                    msg.id === messageId
                      ? { ...msg, pollingStatus: "retry_available" }
                      : msg
                  )
                );
              } else {
                setRateLimitMessage(
                  `${
                    t.rateLimitMessage.split(".")[0]
                  }. (${cooldownTime}s remaining)`
                );
              }
            }, 1000);

            return;
          }
          throw new Error(`Failed to fetch logs: ${logsResponse.status}`);
        }

        const logsData = await logsResponse.json();
        console.log("Logs response:", logsData);

        if (logsData.logs && logsData.logs.length > 0) {
          const latestLog = logsData.logs[0];

          if (latestLog.status === "graded") {
            const judge_score = latestLog.judge_score || null;
            const judge_reason = latestLog.judge_reason || null;

            setMessages((prev) =>
              prev.map((msg) =>
                msg.id === messageId
                  ? {
                      ...msg,
                      judge_score,
                      judge_reason,
                      pollingStatus: "complete",
                    }
                  : msg
              )
            );
            console.log("Grading complete! Score:", judge_score);
            return;
          }
        }

        if (attempts < maxAttempts) {
          setTimeout(poll, 5000);
        } else {
          console.log("Max polling attempts reached (Timeout)");
          setMessages((prev) =>
            prev.map((msg) =>
              msg.id === messageId ? { ...msg, pollingStatus: "timeout" } : msg
            )
          );
        }
      } catch (error) {
        console.error("Error fetching logs:", error);
        if (attempts < maxAttempts) {
          setTimeout(poll, 5000);
        }
      }
    };

    setTimeout(poll, 3000);
  };

  const openGradingModal = (message) => {
    if (message.pollingStatus !== "complete" || message.judge_score === null) {
      if (message.pollingStatus === "rate_limited" || rateLimitCooldown) {
        alert(t.rateLimitMessage);
      } else if (message.pollingStatus.startsWith("checking")) {
        alert(t.checkingGrade);
      }
      return;
    }

    setSelectedMessage(message);
    setIsGradingModalOpen(true);
  };

  const retryGradeCheck = (messageId) => {
    if (rateLimitCooldown) {
      alert(t.rateLimitMessage);
      return;
    }

    setMessages((prev) =>
      prev.map((msg) =>
        msg.id === messageId ? { ...msg, pollingStatus: "checking" } : msg
      )
    );
    startPollingForGrade(messageId);
  };

  const closeGradingModal = () => {
    setIsGradingModalOpen(false);
    setTimeout(() => setSelectedMessage(null), 300);
  };

  const handleLogout = () => {
    localStorage.removeItem("token");
    localStorage.removeItem("user");
    window.location.href = "/";
  };

  const renderGradeStatus = (message) => {
    if (message.pollingStatus === "complete" && message.judge_score !== null) {
      return (
        <button
          onClick={() => openGradingModal(message)}
          className="inline-flex items-center gap-1.5 bg-white text-gray-900 text-xs font-semibold py-2 px-3 rounded border-2 border-green-500 hover:bg-gray-50 transition-colors shadow-sm"
        >
          {t.gradeAvailable}
          <ChevronDown size={14} className="text-gray-600" />
        </button>
      );
    }

    if (message.pollingStatus === "rate_limited") {
      return (
        <div className="inline-flex items-center gap-2 bg-amber-600 text-white text-xs font-semibold py-2 px-3 rounded">
          <Clock size={14} className="animate-pulse" />
          <div className="flex flex-col">
            <span className="font-bold">{t.tooManyRequests}</span>
            <span className="text-xs opacity-90">{t.tryAgainSoon}</span>
          </div>
        </div>
      );
    }

    if (message.pollingStatus && message.pollingStatus.startsWith("checking")) {
      return (
        <div className="inline-flex items-center gap-2 bg-blue-500 text-white text-xs font-semibold py-2 px-3 rounded">
          <div className="flex gap-1 items-center">
            <div className="w-2 h-2 bg-white rounded-full animate-bounce"></div>
            <div
              className="w-2 h-2 bg-white rounded-full animate-bounce"
              style={{ animationDelay: "0.1s" }}
            ></div>
            <div
              className="w-2 h-2 bg-white rounded-full animate-bounce"
              style={{ animationDelay: "0.2s" }}
            ></div>
          </div>
          {t.checkingGrade}
        </div>
      );
    }

    if (
      message.pollingStatus === "retry_available" ||
      message.pollingStatus === "timeout"
    ) {
      const text =
        message.pollingStatus === "timeout" ? t.gradeTimeout : t.tryAgainSoon;
      const color =
        message.pollingStatus === "timeout"
          ? "bg-red-500 hover:bg-red-600"
          : "bg-green-600 hover:bg-green-700";

      return (
        <button
          onClick={() => retryGradeCheck(message.id)}
          className={`inline-flex items-center gap-2 ${color} text-white text-xs font-semibold py-2 px-3 rounded transition-colors`}
        >
          <Clock size={14} />
          {text}
        </button>
      );
    }

    return null;
  };

  return (
    <div className="flex flex-col h-screen bg-gray-100">
      <style>{`
        @keyframes scroll {
          0% { transform: translateY(100%); }
          100% { transform: translateY(-100%); }
        }
        .pillar-text {
          animation: scroll 20s linear infinite;
          writing-mode: vertical-rl;
          text-orientation: mixed;
          white-space: nowrap;
        }
        .chat-bg {
          background-image: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23e5ddd5' fill-opacity='0.3'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
          background-color: #eae6df;
        }
      `}</style>

      <header className="shrink-0 bg-linear-to-r from-green-600 to-green-700 text-white shadow-lg">
        <div className="px-4 sm:px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-white/20 rounded-full flex items-center justify-center">
                <svg
                  className="w-6 h-6 text-white"
                  fill="currentColor"
                  viewBox="0 0 48 48"
                >
                  <path d="M36.7273 44C33.9891 44 31.6043 39.8386 30.3636 33.69C29.123 39.8386 26.7382 44 24 44C21.2618 44 18.877 39.8386 17.6364 33.69C16.3957 39.8386 14.0109 44 11.2727 44C7.25611 44 4 35.0457 4 24C4 12.9543 7.25611 4 11.2727 4C14.0109 4 16.3957 8.16144 17.6364 14.31C18.877 8.16144 21.2618 4 24 4C26.7382 4 29.123 8.16144 30.3636 14.31C31.6043 8.16144 33.9891 4 36.7273 4C40.7439 4 44 12.9543 44 24C44 35.0457 40.7439 44 36.7273 44Z" />
                </svg>
              </div>
              <div>
                <h1 className="text-base font-bold">{t.civicAccess}</h1>
                <p className="text-xs text-green-100">{t.online}</p>
              </div>
            </div>

            <div className="flex items-center gap-2">
              {rateLimitCooldown && (
                <div className="mr-2 px-3 py-1 bg-amber-500 text-white text-xs rounded-full flex items-center gap-1">
                  <Clock size={12} />
                  {t.tooManyRequests}
                </div>
              )}

              <div className="relative">
                <button
                  onClick={() => setShowLanguageDropdown(!showLanguageDropdown)}
                  className="flex items-center gap-1 px-3 py-2 rounded-lg hover:bg-white/10 transition-colors text-xs sm:text-sm"
                >
                  <Globe size={16} />
                  <span>{langName}</span>
                  <ChevronDown size={14} />
                </button>

                {showLanguageDropdown && (
                  <div className="absolute right-0 top-full mt-2 w-40 bg-white text-gray-900 rounded-lg shadow-xl py-1 z-50">
                    {[
                      { name: "English", code: "en" },
                      { name: "Hausa", code: "ha" },
                      { name: "Igbo", code: "ig" },
                      { name: "Yoruba", code: "yo" },
                      { name: "Pidgin", code: "pid" },
                    ].map((lang) => (
                      <button
                        key={lang.code}
                        onClick={() =>
                          handleLanguageChange(lang.code, lang.name)
                        }
                        className="block w-full text-left px-4 py-2 hover:bg-green-50 transition-colors text-sm"
                      >
                        {lang.name}
                      </button>
                    ))}
                  </div>
                )}
              </div>

              <button
                onClick={handleLogout}
                className="flex items-center gap-1 px-3 py-2 bg-red-600 hover:bg-red-700 rounded-lg transition-colors text-xs sm:text-sm"
              >
                <LogOut size={14} />
                {t.logout}
              </button>
            </div>
          </div>
        </div>
      </header>

      <div className="flex-1 flex overflow-hidden">
        <div className="hidden lg:flex w-32 bg-linear-to-r from-green-600 to-green-700 items-center justify-center overflow-hidden">
          <div className="pillar-text text-white/40 font-bold text-md">
            Awarri Hackathon by Team Sabilaw • CivicAccess • Awarri Hackathon by
            Team Sabilaw • CivicAccess •
          </div>
        </div>

        <div className="flex-1 flex flex-col overflow-hidden">
          <div className="flex-1 overflow-y-auto p-4 sm:p-6 space-y-3 chat-bg">
            {messages.map((msg) => (
              <div key={msg.id}>
                <div
                  className={`flex gap-2 ${
                    msg.sender === "user" ? "justify-end" : "justify-start"
                  }`}
                >
                  {msg.sender === "bot" && (
                    <div className="w-8 h-8 rounded-full bg-gray-400 shrink-0 flex items-center justify-center text-white font-bold text-sm">
                      <FaGavel />
                    </div>
                  )}
                  <div
                    className={`max-w-xs lg:max-w-md px-4 py-2 rounded-xl shadow-sm ${
                      msg.sender === "user"
                        ? "bg-green-500 text-white rounded-tr-none"
                        : "bg-white text-gray-900 rounded-tl-none border border-gray-200"
                    }`}
                  >
                    {msg.type === "text" && (
                      <p className="text-sm leading-relaxed">{msg.text}</p>
                    )}
                  </div>
                </div>

                {msg.sender === "bot" && (
                  <div className="flex justify-start mt-1 pl-10">
                    {renderGradeStatus(msg)}
                  </div>
                )}
              </div>
            ))}
            {loading && (
              <div className="flex gap-2 justify-start">
                <div className="w-8 h-8 rounded-full bg-gray-400 shrink-0 flex items-center justify-center text-white font-bold text-sm">
                  <FaGavel />
                </div>
                <div className="bg-white text-gray-900 rounded-xl rounded-tl-none border border-gray-200 px-4 py-2">
                  <div className="flex gap-1">
                    <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce"></div>
                    <div
                      className="w-2 h-2 bg-gray-500 rounded-full animate-bounce"
                      style={{ animationDelay: "0.1s" }}
                    ></div>
                    <div
                      className="w-2 h-2 bg-gray-500 rounded-full animate-bounce"
                      style={{ animationDelay: "0.2s" }}
                    ></div>
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          <div className="shrink-0 bg-white border-t border-gray-300 p-3 sm:p-4">
            {rateLimitMessage && (
              <div className="mb-3 px-4 py-2 bg-amber-50 border border-amber-200 rounded-lg text-amber-800 text-sm">
                <div className="flex items-start gap-2">
                  <Clock size={16} className="mt-0.5 shrink-0" />
                  <span>{rateLimitMessage}</span>
                </div>
              </div>
            )}

            <div className="flex items-center gap-2">
              <button className="flex items-center justify-center w-10 h-10 rounded-full text-gray-600 hover:bg-gray-100">
                <span className="text-2xl">
                  <Plus />
                </span>
              </button>
              <input
                type="text"
                placeholder={t.typeMessage}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) =>
                  e.key === "Enter" &&
                  !loading &&
                  !rateLimitCooldown &&
                  handleSendMessage()
                }
                disabled={loading || rateLimitCooldown}
                className="flex-1 px-4 py-2.5 border-2 border-gray-200 rounded-full focus:outline-none focus:border-green-500 focus:ring-2 focus:ring-green-200 transition-colors text-sm disabled:bg-gray-100 disabled:cursor-not-allowed"
              />
              <button
                onClick={handleSendMessage}
                disabled={loading || rateLimitCooldown}
                className="flex items-center justify-center w-10 h-10 bg-green-600 hover:bg-green-700 text-white rounded-full transition-colors disabled:bg-gray-400 disabled:cursor-not-allowed"
              >
                <Send size={20} />
              </button>
            </div>
          </div>
        </div>

        <div className="hidden lg:flex w-32 bg-linear-to-l from-green-600 to-green-700 items-center justify-center overflow-hidden">
          <div className="pillar-text text-white/40 font-bold text-md">
            Awarri Hackathon by Team Sabilaw • CivicAccess • Awarri Hackathon by
            Team Sabilaw • CivicAccess •
          </div>
        </div>
      </div>

      <GradingModal
        message={selectedMessage}
        isOpen={isGradingModalOpen}
        onClose={closeGradingModal}
      />
      <Footer />
    </div>
  );
}

export default Chatbot;
