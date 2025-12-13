import { useState } from "react";
import { Scale } from "lucide-react";

const Footer = () => {
  const currentYear = new Date().getFullYear();
  const [showDetails, setShowDetails] = useState(false);

  return (
    <footer className="bg-green-50 text-gray-800 border-t border-gray-200">
      <div className="mx-auto px-4 py-3 sm:px-6 lg:px-8">
        <div className="flex justify-center">
          <div
            className="relative cursor-pointer"
            onMouseEnter={() => setShowDetails(true)}
            onMouseLeave={() => setShowDetails(false)}
          >
            <p className="text-[10px] text-gray-500 hover:underline">
              &copy; {currentYear} CivicAccess. Built by{" "}
              <span className="text-green-600 font-medium">Team Sabilaw</span>{" "}
              for Awarri Hackathon.
            </p>
            {showDetails && (
              <div className="absolute left-1/2 transform -translate-x-1/2 bottom-full mb-1 bg-white border border-gray-300 rounded-lg shadow-lg p-4 w-80 z-50">
                <div className="flex items-start gap-3 mb-3">
                  <Scale size={24} className="text-green-600 shrink-0 mt-0.5" />
                  <p className="text-[9px] leading-relaxed text-gray-700">
                    CivicAccess is a multilingual LLM that references the 1999
                    Constitution, the Nigerian Police Act, and the Lagos Tenancy
                    Laws. Bridging the gap between the law and the citizen.
                  </p>
                </div>
                <p className="text-[9px] leading-tight text-red-600 italic">
                  Disclaimer: Every translation is subject to interpretation.
                  This AI provides legal information, not professional legal
                  advice. Terms and conditions apply.
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
