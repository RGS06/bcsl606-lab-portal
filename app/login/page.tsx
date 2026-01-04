"use client";

import { useState } from "react";
import { signInWithEmailAndPassword } from "firebase/auth";
import { auth, db } from "../../lib/firebase";
import { doc, getDoc } from "firebase/firestore";
import { useRouter } from "next/navigation";
import { Lock, Mail, Loader2, ArrowLeft } from "lucide-react";
import Link from "next/link";

export default function LoginPage() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const router = useRouter();

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);

    try {
      const userCredential = await signInWithEmailAndPassword(auth, email, password);
      const uid = userCredential.user.uid;
      const userDoc = await getDoc(doc(db, "users", uid));
      
      if (userDoc.exists()) {
        const userData = userDoc.data();
        if (userData.role === "faculty") {
          router.push("/admin");
        } else {
          router.push("/dashboard");
        }
      } else {
        alert("User data not found in database!");
      }

    } catch (error: any) {
      console.error(error);
      alert("Login Failed: " + error.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#0f172a] flex flex-col items-center justify-center p-4 relative overflow-hidden">
      
      {/* Background Ambient Glow (Optional visual polish) */}
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[500px] h-[500px] bg-blue-600/10 rounded-full blur-[100px] pointer-events-none" />

      {/* LOGIN CARD */}
      <div className="bg-[#1e293b]/80 backdrop-blur-sm p-8 rounded-2xl shadow-2xl w-full max-w-md border border-slate-700 relative z-10 ring-1 ring-white/5">
        <h1 className="text-3xl font-bold text-white mb-2 text-center tracking-tight">Lab Portal</h1>
        <p className="text-slate-400 text-center mb-8 text-sm">Sign in to access your records</p>
        
        <form onSubmit={handleLogin} className="space-y-5">
          <div>
            <label className="text-slate-300 text-xs font-bold mb-1.5 block uppercase tracking-wider">Email</label>
            <div className="relative group">
              <Mail className="absolute left-3 top-3.5 text-slate-500 group-focus-within:text-blue-400 transition-colors" size={18} />
              <input 
                type="email" 
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="usn@test.com"
                className="w-full bg-[#0f172a] border border-slate-700 rounded-xl py-3 pl-10 text-white focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500 transition-all placeholder:text-slate-600"
                required 
              />
            </div>
          </div>

          <div>
            <label className="text-slate-300 text-xs font-bold mb-1.5 block uppercase tracking-wider">Password</label>
            <div className="relative group">
              <Lock className="absolute left-3 top-3.5 text-slate-500 group-focus-within:text-blue-400 transition-colors" size={18} />
              <input 
                type="password" 
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="••••••••"
                className="w-full bg-[#0f172a] border border-slate-700 rounded-xl py-3 pl-10 text-white focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500 transition-all placeholder:text-slate-600"
                required 
              />
            </div>
          </div>

          <button 
            type="submit" 
            disabled={loading}
            className="w-full bg-gradient-to-r from-blue-600 to-blue-500 hover:from-blue-500 hover:to-blue-400 text-white font-bold py-3.5 rounded-xl transition-all duration-300 transform hover:scale-[1.02] shadow-lg shadow-blue-900/20 flex items-center justify-center gap-2 mt-2"
          >
            {loading ? <Loader2 className="animate-spin" /> : "Sign In"}
          </button>
        </form>
      </div>

      {/* 👇 ANIMATED BACK BUTTON */}
      <Link 
        href="/" 
        className="mt-8 group relative px-6 py-2.5 rounded-full bg-slate-800/40 border border-slate-700/50 hover:border-blue-500/50 hover:bg-blue-900/10 transition-all duration-300 overflow-hidden"
      >
        <div className="relative z-10 flex items-center gap-2.5">
          <ArrowLeft size={16} className="text-slate-400 group-hover:text-blue-400 group-hover:-translate-x-1 transition-transform duration-300" />
          <span className="text-sm font-medium text-slate-400 group-hover:text-blue-100 transition-colors">Back to Home</span>
        </div>
        
        {/* Hover Glow Effect */}
        <div className="absolute inset-0 rounded-full opacity-0 group-hover:opacity-100 transition-opacity duration-500 shadow-[0_0_20px_rgba(59,130,246,0.15)]"></div>
      </Link>

    </div>
  );
}