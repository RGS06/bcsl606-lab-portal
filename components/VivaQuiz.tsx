"use client";
import { useState } from "react";
import { questions } from "../data/questions";
import { CheckCircle2, XCircle, Trophy, RefreshCw } from "lucide-react";

export default function VivaQuiz({ labId }: { labId: string }) {
  const labQuestions = questions[labId] || [];
  const [currentQ, setCurrentQ] = useState(0);
  const [score, setScore] = useState(0);
  const [showScore, setShowScore] = useState(false);
  const [selectedOption, setSelectedOption] = useState<number | null>(null);

  const handleAnswer = (index: number) => {
    setSelectedOption(index);
    if (index === labQuestions[currentQ].answer) {
      setScore(score + 1);
    }
    
    // Auto advance after short delay
    setTimeout(() => {
      if (currentQ < labQuestions.length - 1) {
        setCurrentQ(currentQ + 1);
        setSelectedOption(null);
      } else {
        setShowScore(true);
      }
    }, 800);
  };

  const resetQuiz = () => {
    setCurrentQ(0);
    setScore(0);
    setShowScore(false);
    setSelectedOption(null);
  };

  if (labQuestions.length === 0) return <div className="text-slate-500 text-sm p-4">No Viva questions available for this lab.</div>;

  return (
    <div className="bg-black/20 rounded-xl p-6 border border-white/5">
      {showScore ? (
        <div className="text-center py-8">
          <div className="w-20 h-20 bg-yellow-500/20 text-yellow-400 rounded-full flex items-center justify-center mx-auto mb-4 animate-bounce">
            <Trophy size={40} />
          </div>
          <h3 className="text-2xl font-bold text-white mb-2">Quiz Completed!</h3>
          <p className="text-slate-300 mb-6">You scored <span className="text-blue-400 font-bold text-xl">{score} / {labQuestions.length}</span></p>
          <button 
            onClick={resetQuiz}
            className="flex items-center gap-2 mx-auto px-6 py-2 bg-white/10 hover:bg-white/20 rounded-full text-white transition-all"
          >
            <RefreshCw size={16} /> Retry Quiz
          </button>
        </div>
      ) : (
        <div>
          <div className="flex justify-between items-center mb-6">
            <span className="text-xs font-bold text-slate-400 uppercase tracking-wider">Question {currentQ + 1} of {labQuestions.length}</span>
            <span className="text-xs font-bold bg-blue-500/20 text-blue-300 px-2 py-1 rounded">Score: {score}</span>
          </div>

          <h4 className="text-lg font-medium text-white mb-6 leading-relaxed">
            {labQuestions[currentQ].question}
          </h4>

          <div className="space-y-3">
            {labQuestions[currentQ].options.map((option: string, idx: number) => {
              const isSelected = selectedOption === idx;
              const isCorrect = idx === labQuestions[currentQ].answer;
              
              // Determine card style based on state
              let cardClass = "bg-white/5 border-white/10 hover:bg-white/10 text-slate-300";
              if (selectedOption !== null) {
                if (isSelected && isCorrect) cardClass = "bg-green-500/20 border-green-500/50 text-green-200";
                else if (isSelected && !isCorrect) cardClass = "bg-red-500/20 border-red-500/50 text-red-200";
              }

              return (
                <button
                  key={idx}
                  onClick={() => selectedOption === null && handleAnswer(idx)}
                  className={`w-full p-4 rounded-xl border text-left transition-all duration-200 flex justify-between items-center ${cardClass}`}
                  disabled={selectedOption !== null}
                >
                  {option}
                  {selectedOption !== null && isSelected && (
                    isCorrect ? <CheckCircle2 size={18} className="text-green-400" /> : <XCircle size={18} className="text-red-400" />
                  )}
                </button>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}