"use client";

import { useState, useEffect, useRef } from "react";
// 👇 Imports (4 levels up)
import { db, auth } from "../../../../lib/firebase";
import { doc, getDoc, setDoc } from "firebase/firestore";
import { onAuthStateChanged } from "firebase/auth";
import { useRouter, useSearchParams } from "next/navigation";
import { questionBank } from "../../../../data/questions";
import { AlertTriangle, Timer, ShieldAlert, CheckCircle, Lock, MessageCircle, Trophy, ArrowRight, Maximize } from "lucide-react";

export default function ExamPage({ params }: { params: { labId: string } }) {
  const [user, setUser] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [examState, setExamState] = useState("checking"); 
  const [finalScore, setFinalScore] = useState<number | null>(null);
  const [terminationReason, setTerminationReason] = useState(""); // To tell them WHY they failed
  
  const searchParams = useSearchParams();
  const examType = searchParams.get("type") === "viva" ? "viva" : "cie";
  const labData = questionBank[params.labId as keyof typeof questionBank];
  const rawQuestions = examType === "viva" ? (labData?.viva || []) : (labData?.cie || []);

  const [questions] = useState(() => [...rawQuestions].sort(() => Math.random() - 0.5));

  // Exam State
  const [currentQIndex, setCurrentQIndex] = useState(0); 
  const [answers, setAnswers] = useState<{[key: number]: number}>({});
  const [timeLeft, setTimeLeft] = useState(10); 
  
  const answersRef = useRef(answers); 
  const router = useRouter();

  // Sync Ref
  useEffect(() => { answersRef.current = answers; }, [answers]);

  // 1. Auth & Gatekeeper
  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, async (currentUser) => {
      if (!currentUser) return router.push("/login");
      setUser(currentUser);

      try {
        const scoreDoc = await getDoc(doc(db, "scores", `${params.labId}_${currentUser.uid}`));
        if (scoreDoc.exists()) {
            const data = scoreDoc.data();
            const prevScore = examType === "viva" ? data.vivaScore : data.cieScore;
            if (prevScore !== undefined) {
                setFinalScore(prevScore);
                setExamState("locked_done");
                setLoading(false);
                return; 
            }
        }

        const userDoc = await getDoc(doc(db, "users", currentUser.uid));
        if (userDoc.exists()) {
          const userData = userDoc.data();
          const today = new Date().toISOString().split('T')[0];
          const sessionID = `${params.labId}_${userData.section}_${userData.batch}_${today}`;
          const sessionDoc = await getDoc(doc(db, "attendance", sessionID));
          
          if (sessionDoc.exists() && sessionDoc.data().presentIds.includes(currentUser.uid)) {
            setExamState("start");
          } else {
            setExamState("denied");
          }
        }
      } catch (error) { console.error(error); }
      setLoading(false);
    });
    return () => unsubscribe();
  }, [examType, params.labId]); 

  // 2. START EXAM + ENTER FULL SCREEN 🖥️
  const handleStartExam = async () => {
    try {
        // Force Full Screen
        await document.documentElement.requestFullscreen();
    } catch (err) {
        console.error("Fullscreen blocked:", err);
        // We can optionally block them here if strict mode is required
        alert("⚠️ Please enable Full Screen to proceed!");
    }
    setExamState("active");
  };

  // 3. THE TRIPLE THREAT WATCHDOG 🐶💀
  useEffect(() => {
    if (examState !== "active") return;

    // A. Tab Switch / Minimize
    const onVisibilityChange = () => {
        if (document.hidden) handleSubmit("cheated", "Tab Switched / Minimized");
    };

    // B. Focus Lost (Clicking another window/browser)
    const onBlur = () => {
        handleSubmit("cheated", "Window Lost Focus (Clicked outside)");
    };

    // C. Exiting Full Screen (Pressing ESC)
    const onFullscreenChange = () => {
        if (!document.fullscreenElement) {
            handleSubmit("cheated", "Exited Full Screen Mode");
        }
    };

    document.addEventListener("visibilitychange", onVisibilityChange);
    window.addEventListener("blur", onBlur);
    document.addEventListener("fullscreenchange", onFullscreenChange);

    return () => {
        document.removeEventListener("visibilitychange", onVisibilityChange);
        window.removeEventListener("blur", onBlur);
        document.removeEventListener("fullscreenchange", onFullscreenChange);
    };
  }, [examState]);

  // 4. Timer
  useEffect(() => {
    if (examState !== "active") return;
    const timer = setInterval(() => {
      setTimeLeft((prev) => {
        if (prev <= 1) {
          handleNextQuestion();
          return 10; 
        }
        return prev - 1;
      });
    }, 1000);
    return () => clearInterval(timer);
  }, [examState, currentQIndex]);

  // 5. Navigation
  const handleNextQuestion = () => {
    if (currentQIndex < questions.length - 1) {
        setCurrentQIndex(prev => prev + 1);
        setTimeLeft(10); 
    } else {
        handleSubmit("submitted"); 
    }
  };

  const handleSelectAnswer = (qId: number, optionIndex: number) => {
    setAnswers(prev => ({ ...prev, [qId]: optionIndex }));
  };

  // 6. Submit Logic
  const handleSubmit = async (status = "submitted", reason = "") => {
    if (!user) return;
    
    // Save termination reason to show to user
    if (reason) setTerminationReason(reason);

    const currentAnswers = answersRef.current;
    
    let score = 0;
    questions.forEach((q) => {
      if (currentAnswers[q.id] === q.correct) score += 1;
    });

    const finalStatus = status === "cheated" ? `Flagged: ${reason}` : "Completed";
    const scoreField = examType === "viva" ? "vivaScore" : "cieScore";

    await setDoc(doc(db, "scores", `${params.labId}_${user.uid}`), {
      labId: params.labId,
      studentId: user.uid,
      studentName: user.displayName || "Unknown",
      [scoreField]: score,
      [`${examType}Status`]: finalStatus,
      [`${examType}Timestamp`]: new Date().toISOString()
    }, { merge: true });

    setFinalScore(score);
    setExamState(status === "cheated" ? "cheated" : "submitted");
    
    // Exit fullscreen cleanly if it was active
    if (document.fullscreenElement) {
        document.exitFullscreen().catch(() => {});
    }
  };

  if (loading) return <div className="min-h-screen bg-[#0f172a] text-white flex items-center justify-center">Loading Assessment...</div>;

  // --- UI STATES ---

  if (examState === "locked_done" || examState === "submitted" || examState === "cheated") return (
    <div className="min-h-screen bg-[#0f172a] flex items-center justify-center p-4">
      <div className={`bg-[#1e293b] p-8 rounded-2xl max-w-md text-center border shadow-xl ${examState === "cheated" ? "border-red-500" : "border-slate-700"}`}>
        {examState === "cheated" ? <ShieldAlert className="w-16 h-16 text-red-500 mx-auto mb-4"/> : <Trophy className="w-16 h-16 text-yellow-500 mx-auto mb-4"/>}
        <h1 className="text-2xl font-bold text-white mb-2">{examState === "cheated" ? "Terminated" : "Completed"}</h1>
        
        {/* REASON FOR TERMINATION */}
        {examState === "cheated" && (
            <div className="bg-red-500/10 border border-red-500/20 text-red-300 text-sm p-3 rounded-lg mb-4">
                <b>Violation Detected:</b><br/>{terminationReason || "Suspicious Activity"}
            </div>
        )}
        
        <p className="text-slate-400 mb-6">{examState === "cheated" ? "Score saved up to this point." : "Your score has been recorded."}</p>
        
        <div className="bg-slate-900 p-6 rounded-xl mb-6 border border-slate-700">
            <div className="text-slate-400 text-xs uppercase tracking-widest mb-1">Final Score</div>
            <div className="text-4xl font-bold text-white">{finalScore} <span className="text-lg text-slate-500">/ {questions.length}</span></div>
        </div>

        <button onClick={() => router.push("/dashboard")} className="bg-slate-700 hover:bg-slate-600 text-white px-6 py-3 rounded-xl w-full font-bold">Return to Dashboard</button>
      </div>
    </div>
  );

  if (examState === "denied") return (
    <div className="min-h-screen bg-[#0f172a] flex items-center justify-center p-4">
      <div className="bg-[#1e293b] p-8 rounded-2xl max-w-md text-center border border-red-500/30">
        <Lock className="w-16 h-16 text-red-500 mx-auto mb-4" />
        <h1 className="text-2xl font-bold text-white mb-2">Access Locked</h1>
        <button onClick={() => router.push("/dashboard")} className="bg-slate-700 text-white px-6 py-2 rounded-lg w-full mt-4">Back to Dashboard</button>
      </div>
    </div>
  );

  if (examState === "start") return (
    <div className="min-h-screen bg-[#0f172a] flex items-center justify-center p-4">
      <div className={`bg-[#1e293b] p-8 rounded-2xl max-w-md text-center border shadow-2xl ${examType === "viva" ? "border-purple-500/30" : "border-red-500/30"}`}>
        {examType === "viva" ? <MessageCircle className="w-16 h-16 text-purple-500 mx-auto mb-4"/> : <Timer className="w-16 h-16 text-red-500 mx-auto mb-4"/>}
        
        <h1 className="text-2xl font-bold text-white mb-2">{examType.toUpperCase()} Assessment</h1>
        <p className="text-slate-400 text-sm mb-6">Topic: {params.labId.toUpperCase()}</p>
        
        <div className="text-left bg-slate-900/50 p-4 rounded-xl space-y-3 mb-6 border border-slate-700">
          <div className="flex items-center gap-3 text-sm text-slate-300">
             <Maximize size={16} className="text-blue-400"/> <span><b>Full Screen</b> Required.</span>
          </div>
          <div className="flex items-center gap-3 text-sm text-slate-300">
             <Timer size={16} className="text-blue-400"/> <span>Time: <b>10 Seconds</b> per Question!</span>
          </div>
          <div className="flex items-center gap-3 text-sm text-red-300 bg-red-500/10 p-2 rounded border border-red-500/20">
             <ShieldAlert size={16} className="text-red-500 flex-shrink-0"/> 
             <span><b>STRICT MODE:</b> Tab switch, clicking outside, or exiting Full Screen = <b>FAIL</b>.</span>
          </div>
        </div>

        <button 
          onClick={handleStartExam} 
          className={`px-8 py-3 rounded-xl font-bold text-lg w-full transition-all shadow-lg text-white ${examType === "viva" ? "bg-purple-600 hover:bg-purple-500 shadow-purple-900/20" : "bg-red-600 hover:bg-red-500 shadow-red-900/20"}`}
        >
          Enter Full Screen & Start
        </button>
      </div>
    </div>
  );

  // ACTIVE EXAM (BLITZ UI)
  const q = questions[currentQIndex];

  return (
    <div className="min-h-screen bg-[#0f172a] text-white font-sans selection:bg-blue-500/30 flex flex-col">
      {/* Top Bar */}
      <div className="fixed top-0 w-full bg-[#1e293b]/90 backdrop-blur border-b border-slate-700 p-4 flex justify-between items-center z-10">
         <div className="text-sm font-bold text-slate-400">Q{currentQIndex + 1} of {questions.length}</div>
         <div className="absolute left-0 bottom-0 h-1 bg-blue-600 transition-all duration-300" style={{ width: `${((currentQIndex + 1) / questions.length) * 100}%` }} />
         <div className={`font-mono text-xl font-bold ${timeLeft <= 5 ? "text-red-500 scale-110 transition-transform" : "text-blue-400"}`}>
            00:{timeLeft.toString().padStart(2, '0')}
         </div>
      </div>

      <div className="flex-1 flex flex-col items-center justify-center p-4 pt-20">
        <div className="w-full max-w-2xl">
            <h2 className="text-2xl font-bold mb-8 leading-relaxed text-center">{q.question}</h2>
            <div className="grid gap-4">
                {q.options?.map((opt: string, i: number) => (
                    <button
                        key={i}
                        onClick={() => handleSelectAnswer(q.id, i)}
                        className={`w-full text-left p-5 rounded-xl border transition-all duration-200 flex items-center justify-between group ${answers[q.id] === i ? "bg-blue-600 border-blue-500 text-white shadow-lg scale-[1.02]" : "bg-[#1e293b] border-slate-700 text-slate-300 hover:bg-slate-800"}`}
                    >
                        <span className="font-medium text-lg">{opt}</span>
                        {answers[q.id] === i && <CheckCircle size={20} className="text-white"/>}
                    </button>
                ))}
            </div>
            <button onClick={handleNextQuestion} className="mt-8 w-full bg-slate-700 hover:bg-slate-600 text-white font-bold py-4 rounded-xl flex items-center justify-center gap-2 transition-all active:scale-[0.98]">
                {currentQIndex === questions.length - 1 ? "Finish Exam" : "Next Question"} <ArrowRight size={20}/>
            </button>
        </div>
      </div>
    </div>
  );
}