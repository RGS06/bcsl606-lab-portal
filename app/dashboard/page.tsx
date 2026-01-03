"use client";

import { useEffect, useState } from "react";
import { auth, db } from "../../lib/firebase";
import { onAuthStateChanged, signOut } from "firebase/auth";
import { useRouter } from "next/navigation";
import { doc, getDoc } from "firebase/firestore";
import { labs } from "../../data/syllabus";
import Link from "next/link";
import { BookOpen, Terminal, LogOut, Beaker, Timer, MessageCircle, UserCircle } from "lucide-react";

export default function StudentDashboard() {
  const [user, setUser] = useState<any>(null);
  const [studentData, setStudentData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const router = useRouter();

  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, async (currentUser) => {
      if (!currentUser) {
        router.push("/login");
      } else {
        setUser(currentUser);
        // Fetch extra student details (Name, USN)
        const userDoc = await getDoc(doc(db, "users", currentUser.uid));
        if (userDoc.exists()) {
          setStudentData(userDoc.data());
        }
        setLoading(false);
      }
    });
    return () => unsubscribe();
  }, []);

  const handleLogout = async () => {
    await signOut(auth);
    router.push("/login");
  };

  if (loading) return <div className="min-h-screen bg-[#0f172a] text-white flex items-center justify-center">Loading Dashboard...</div>;

  return (
    <div className="min-h-screen bg-[#0f172a] text-white font-sans selection:bg-purple-500/30">
      
      {/* Header */}
      <header className="max-w-7xl mx-auto p-6 flex flex-col md:flex-row justify-between items-center border-b border-slate-800 mb-8 gap-4">
        <div>
          <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
            Welcome, {studentData?.name || "Student"}
          </h1>
          <p className="text-slate-400 text-sm mt-1 flex items-center gap-2">
             <span className="bg-slate-800 px-2 py-0.5 rounded text-xs font-mono border border-slate-700">{studentData?.usn}</span>
             <span>• Batch {studentData?.batch}</span>
          </p>
        </div>
        <button 
          onClick={handleLogout}
          className="flex items-center gap-2 px-4 py-2 rounded-xl bg-slate-800 hover:bg-red-500/10 hover:text-red-400 border border-slate-700 transition-all text-sm font-bold"
        >
          <LogOut size={16} /> Sign Out
        </button>
      </header>

      <main className="max-w-7xl mx-auto p-6 grid grid-cols-1 lg:grid-cols-3 gap-8">
        
        {/* Left Column: Playground & Stats */}
        <div className="lg:col-span-1 space-y-6">
          
          {/* Playground Card */}
          <Link href="/dashboard/playground">
            <div className="bg-[#1e293b] p-6 rounded-2xl border border-slate-700 shadow-xl hover:border-purple-500/50 transition-all group cursor-pointer relative overflow-hidden">
              <div className="absolute top-0 right-0 w-32 h-32 bg-purple-500/10 rounded-full blur-3xl -mr-16 -mt-16 pointer-events-none"></div>
              
              <div className="w-12 h-12 bg-purple-500/20 rounded-xl flex items-center justify-center mb-4 group-hover:scale-110 transition-transform duration-300">
                <Terminal className="text-purple-400" size={24} />
              </div>
              <h3 className="text-xl font-bold mb-2 group-hover:text-purple-400 transition-colors">Code Playground</h3>
              <p className="text-slate-400 text-sm leading-relaxed">
                Run Python code instantly. Test your logic with NumPy & Pandas.
              </p>
            </div>
          </Link>

          {/* Profile Card */}
          <div className="bg-[#1e293b] p-6 rounded-2xl border border-slate-700 shadow-xl">
            <div className="flex items-center gap-4 mb-6">
                <div className="w-12 h-12 rounded-full bg-blue-500/20 flex items-center justify-center text-blue-400">
                    <UserCircle size={28} />
                </div>
                <div>
                    <div className="font-bold">Your Profile</div>
                    <div className="text-xs text-slate-400">Section {studentData?.section}</div>
                </div>
            </div>
            <div className="space-y-3">
                <div className="flex justify-between text-sm">
                    <span className="text-slate-400">Attendance</span>
                    <span className="text-green-400 font-bold">Good</span>
                </div>
                <div className="w-full bg-slate-700 h-1.5 rounded-full overflow-hidden">
                    <div className="bg-green-500 w-[85%] h-full rounded-full" />
                </div>
            </div>
          </div>

        </div>

        {/* Right Column: Labs List */}
        <div className="lg:col-span-2 space-y-6">
          <h2 className="text-xl font-bold flex items-center gap-2">
            <Beaker className="text-blue-400" size={20}/> Lab Experiments
          </h2>

          <div className="grid gap-4">
            {labs.map((lab, index) => (
              <div key={lab.id} className="bg-[#1e293b] rounded-2xl border border-slate-700 overflow-hidden hover:border-slate-600 transition-all">
                 <div className="p-6">
                    <div className="flex justify-between items-start mb-2">
                        <div className="text-xs font-bold text-blue-400 mb-1 uppercase tracking-wider">Experiment {index + 1}</div>
                        {/* Status Badge (Placeholder for now) */}
                        <div className="px-2 py-0.5 rounded text-[10px] bg-slate-700 text-slate-300 border border-slate-600">Pending</div>
                    </div>
                    <h3 className="text-lg font-bold text-white mb-2">{lab.title}</h3>
                    <p className="text-sm text-slate-400 line-clamp-2">{lab.aim}</p>
                 </div>

                 {/* ACTION BUTTONS */}
                 <div className="bg-slate-800/50 p-4 flex gap-3 border-t border-slate-700/50">
                    
                    {/* 1. Manual */}
                    <Link 
                        href={`/dashboard/manual/${lab.id}`}
                        className="flex-1 bg-slate-700 hover:bg-slate-600 text-white py-2 rounded-lg text-xs font-bold flex items-center justify-center gap-2 transition-all border border-slate-600"
                    >
                        <BookOpen size={14} /> Manual
                    </Link>

                    {/* 2. CIE Assessment */}
                    <Link 
                        href={`/dashboard/exam/${lab.id}?type=cie`}
                        className="flex-1 bg-red-500/10 hover:bg-red-500/20 text-red-400 hover:text-red-300 py-2 rounded-lg text-xs font-bold flex items-center justify-center gap-2 transition-all border border-red-500/20 group"
                    >
                        <Timer size={14} className="group-hover:animate-pulse"/> CIE
                    </Link>

                    {/* 3. Viva Quiz */}
                    <Link 
                        href={`/dashboard/exam/${lab.id}?type=viva`}
                        className="flex-1 bg-purple-500/10 hover:bg-purple-500/20 text-purple-400 hover:text-purple-300 py-2 rounded-lg text-xs font-bold flex items-center justify-center gap-2 transition-all border border-purple-500/20"
                    >
                        <MessageCircle size={14} /> Viva
                    </Link>

                 </div>
              </div>
            ))}
          </div>
        </div>

      </main>
    </div>
  );
}