"use client";

import { useState } from "react";
import { signInWithEmailAndPassword } from "firebase/auth";
import { auth } from "../../../lib/firebase"; 
import { useRouter } from "next/navigation";
import { Mail, Lock, ArrowRight, GraduationCap, CheckCircle2 } from "lucide-react";

export default function LoginPage() {
  const [isFaculty, setIsFaculty] = useState(false);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const router = useRouter();

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError("");
    try {
      await signInWithEmailAndPassword(auth, email, password);
      router.push(isFaculty ? "/admin" : "/");
    } catch (err) {
      setError("Invalid credentials. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex min-h-screen w-full font-sans bg-white">
      
      {/* LEFT SIDE: VISUAL BRANDING (Dark & Cinematic) */}
      <div className="hidden lg:flex w-1/2 bg-slate-900 relative items-center justify-center overflow-hidden">
        {/* Abstract Background Effects */}
        <div className="absolute top-[-10%] left-[-10%] w-[600px] h-[600px] bg-blue-600/30 rounded-full blur-[120px]"></div>
        <div className="absolute bottom-[-10%] right-[-10%] w-[600px] h-[600px] bg-purple-600/20 rounded-full blur-[120px]"></div>
        
        <div className="relative z-10 p-12 max-w-lg text-white">
          <div className="mb-8 inline-flex p-3 bg-white/10 backdrop-blur-md rounded-xl border border-white/10">
             <GraduationCap size={32} className="text-blue-400" />
          </div>
          <h1 className="text-5xl font-extrabold mb-6 leading-tight tracking-tight">
            Master Machine <br/>
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-purple-400">
              Learning Today.
            </span>
          </h1>
          <p className="text-slate-400 text-lg leading-relaxed mb-8">
            Access virtual labs, track experiments, and validate your models with our AI-powered educational platform.
          </p>
          
          <div className="space-y-4">
            <div className="flex items-center gap-3 text-slate-300">
                <CheckCircle2 className="text-blue-500" size={20} />
                <span>Real-time Attendance Tracking</span>
            </div>
            <div className="flex items-center gap-3 text-slate-300">
                <CheckCircle2 className="text-blue-500" size={20} />
                <span>AI-Powered Code Analysis</span>
            </div>
          </div>
        </div>
      </div>

      {/* RIGHT SIDE: CLEAN LOGIN FORM (Light & Clean) */}
      <div className="w-full lg:w-1/2 flex items-center justify-center p-8 bg-white">
        <div className="w-full max-w-md space-y-8">
          
          <div className="text-center">
            <h2 className="text-3xl font-bold text-slate-900 tracking-tight">Welcome Back</h2>
            <p className="mt-2 text-slate-500">Please enter your details to sign in.</p>
          </div>

          {/* Custom Toggle Switch */}
          <div className="flex p-1.5 bg-slate-100 rounded-xl border border-slate-200">
            <button
              onClick={() => setIsFaculty(false)}
              className={`flex-1 py-3 text-sm font-bold rounded-lg transition-all duration-200 ${
                !isFaculty 
                  ? "bg-white text-blue-600 shadow-sm ring-1 ring-slate-200" 
                  : "text-slate-500 hover:text-slate-900"
              }`}
            >
              Student Portal
            </button>
            <button
              onClick={() => setIsFaculty(true)}
              className={`flex-1 py-3 text-sm font-bold rounded-lg transition-all duration-200 ${
                isFaculty 
                  ? "bg-white text-purple-600 shadow-sm ring-1 ring-slate-200" 
                  : "text-slate-500 hover:text-slate-900"
              }`}
            >
              Faculty Portal
            </button>
          </div>

          <form onSubmit={handleLogin} className="space-y-5">
            <div>
              <label className="block text-xs font-bold uppercase text-slate-500 mb-2 ml-1">Email Address</label>
              <div className="relative group">
                <Mail className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-400 group-focus-within:text-blue-600 transition-colors" size={20} />
                <input 
                  type="email" 
                  required
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  className="w-full bg-slate-50 border border-slate-200 rounded-xl py-4 pl-12 outline-none focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500 transition-all font-medium text-slate-900"
                  placeholder={isFaculty ? "faculty@test.com" : "student@test.com"}
                />
              </div>
            </div>

            <div>
              <label className="block text-xs font-bold uppercase text-slate-500 mb-2 ml-1">Password</label>
              <div className="relative group">
                <Lock className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-400 group-focus-within:text-blue-600 transition-colors" size={20} />
                <input 
                  type="password" 
                  required
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="w-full bg-slate-50 border border-slate-200 rounded-xl py-4 pl-12 outline-none focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500 transition-all font-medium text-slate-900"
                  placeholder="••••••••"
                />
              </div>
            </div>

            {error && (
              <div className="p-3 rounded-lg bg-red-50 text-red-600 text-sm text-center font-medium border border-red-100">
                {error}
              </div>
            )}

            <button 
              disabled={loading}
              className={`w-full py-4 rounded-xl font-bold text-white text-lg shadow-lg hover:-translate-y-0.5 active:translate-y-0 transition-all flex items-center justify-center gap-2 ${
                isFaculty 
                  ? "bg-purple-600 hover:bg-purple-700 shadow-purple-200" 
                  : "bg-blue-600 hover:bg-blue-700 shadow-blue-200"
              }`}
            >
              {loading ? "Signing in..." : "Login to Dashboard"} 
              {!loading && <ArrowRight size={20}/>}
            </button>
          </form>

          <p className="text-center text-slate-400 text-sm">
            Having trouble accessing your account? <br/>
            <a href="#" className="text-blue-600 font-bold hover:underline">Contact System Admin</a>
          </p>
        </div>
      </div>
    </div>
  );
}