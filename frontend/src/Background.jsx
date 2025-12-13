function Background() {
  return (
    <div className="fixed inset-0 overflow-hidden pointer-events-none">
      <div className="absolute inset-0 bg-linear-to-r from-green-50/5 via-white/3 to-green-50/5"></div>

      <div className="absolute inset-0 flex items-center justify-start overflow-hidden">
        <div className="animate-scroll whitespace-nowrap">
          <span className="text-6xl font-bold text-green-900/10 mr-8">
            Awarri Hackathon by Team Sabilaw •
          </span>
          <span className="text-6xl font-bold text-green-900/10 mr-8">
            Awarri Hackathon by Team Sabilaw •
          </span>
          <span className="text-6xl font-bold text-green-900/10 mr-8">
            Awarri Hackathon by Team Sabilaw •
          </span>
        </div>

        {/* <div className="animate-scroll whitespace-nowrap absolute" style={{ animationDelay: '-10s' }}>
          <span className="text-6xl font-bold text-green-900/10 mr-8">
            Awarri Hackathon by Team Sabilaw •
          </span>
          <span className="text-6xl font-bold text-green-900/10 mr-8">
            Awarri Hackathon by Team Sabilaw •
          </span>
          <span className="text-6xl font-bold text-green-900/10 mr-8">
            Awarri Hackathon by Team Sabilaw •
          </span>
        </div> */}
      </div>

      <style>{`
        @keyframes scroll {
          0% {
            transform: translateX(0);
          }
          100% {
            transform: translateX(calc(-50% - 0px));
          }
        }
        .animate-scroll {
          animation: scroll 20s linear infinite;
        }
      `}</style>
    </div>
  );
}

export default Background;
