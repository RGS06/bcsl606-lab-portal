import { Code2, GraduationCap, MapPin, Building2, Terminal } from "lucide-react";

export default function Footer() {
  return (
    <footer className="bg-[#020617] border-t border-slate-800 pt-16 pb-8 relative overflow-hidden">
      
      {/* Background Texture (Subtle Grid) */}
      <div className="absolute inset-0 opacity-[0.03] bg-[url('https://www.transparenttextures.com/patterns/graphy.png')] pointer-events-none"></div>
      
      {/* Top Gradient Line */}
      <div className="absolute top-0 left-0 w-full h-px bg-gradient-to-r from-transparent via-blue-500/50 to-transparent"></div>

      <div className="max-w-7xl mx-auto px-6 relative z-10">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-12 items-start mb-12">
          
          {/* LEFT: LAB & INSTITUTION INFO */}
          <div className="space-y-6">
            <div>
                <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-blue-500/10 border border-blue-500/20 text-blue-400 text-xs font-bold font-mono mb-4">
                    <Terminal size={12} /> BCSL606
                </div>
                <h3 className="text-2xl font-bold text-white tracking-tight">Machine Learning Laboratory</h3>
                <p className="text-slate-400 text-sm mt-2 leading-relaxed max-w-md">
                    A comprehensive digital record for ML experiments.
                </p>
            </div>

            <div className="space-y-3">
                <div className="flex items-center gap-3 text-slate-500 text-sm group">
                    <div className="p-2 rounded-lg bg-slate-900 border border-slate-800 group-hover:border-slate-700 transition-colors">
                        <Building2 size={16} className="text-slate-400"/>
                    </div>
                    <span>Department of Computer Science & Engineering</span>
                </div>
                <div className="flex items-center gap-3 text-slate-500 text-sm group">
                    <div className="p-2 rounded-lg bg-slate-900 border border-slate-800 group-hover:border-slate-700 transition-colors">
                        <GraduationCap size={16} className="text-slate-400"/>
                    </div>
                    <span>SMVITM, Bantakal</span>
                </div>
            </div>
          </div>

          {/* RIGHT: DEVELOPER CREDIT (Badge Style) */}
          <div className="flex flex-col md:items-end">
            <div className="bg-[#0f172a] border border-slate-800 rounded-2xl p-6 max-w-sm w-full hover:border-blue-500/30 transition-all duration-300 group shadow-lg">
                <div className="flex items-center gap-2 mb-4">
                    <Code2 size={18} className="text-blue-500" />
                    <span className="text-xs font-bold uppercase tracking-widest text-slate-500">Designed & Developed By</span>
                </div>
                
                <div className="flex items-center gap-4">
                    {/* Initials Avatar */}
                    <div className="w-12 h-12 rounded-full bg-gradient-to-br from-blue-600 to-indigo-600 flex items-center justify-center text-white font-bold text-lg shadow-lg shadow-blue-900/30">
                        RGS
                    </div>
                    
                    <div>
                        <h4 className="text-white font-bold text-lg group-hover:text-blue-400 transition-colors">Raghavendra GS</h4>
                        <p className="text-slate-400 text-xs font-mono uppercase tracking-wide bg-slate-800/50 px-2 py-0.5 rounded inline-block mt-1">
                        Assistant Professor (Sr.)
                        </p>
                    </div>
                </div>

                <div className="mt-4 pt-4 border-t border-slate-800 flex items-center gap-2 text-xs text-slate-500">
                    <MapPin size={12} /> Dept. of CSE, SMVITM
                </div>
            </div>
          </div>

        </div>

{/* BOTTOM COPYRIGHT */}
        <div className="border-t border-slate-800/50 pt-8 flex flex-col md:flex-row justify-between items-center gap-4 text-xs font-mono uppercase tracking-widest transition-colors">
            
            <p className="text-slate-500 font-semibold hover:text-white cursor-default transition-colors">
                © 2026 - 2027 Lab Portal. All rights reserved.
            </p>
            
            {/* ✨ Brighter & Bolder "Made For" Text */}
            <p className="flex items-center gap-2 text-slate-400 font-bold bg-slate-800/50 px-3 py-1.5 rounded-full border border-slate-700/50 hover:border-blue-500/30 hover:bg-blue-900/10 hover:text-blue-200 transition-all cursor-default shadow-sm">
                <span className="w-1.5 h-1.5 rounded-full bg-blue-500 animate-pulse"></span>
                Made for Students & Faculty
            </p>

        </div>
      </div>
    </footer>
  );
}