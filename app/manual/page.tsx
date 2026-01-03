"use client";

import { useRef, forwardRef, useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { manuals } from "../../data/manual"; 
import { ArrowLeft, Shield, Book, ChevronRight, ChevronLeft, Terminal, FileText } from "lucide-react";
import Link from "next/link";
// @ts-ignore
import HTMLFlipBook from "react-pageflip";

// --- 🌑 DARK MODE PAGE COMPONENT ---
const Page = forwardRef((props: any, ref: any) => {
  return (
    <div className="demoPage bg-[#1e293b] text-slate-300 border-r border-slate-800 shadow-[inset_0_0_20px_rgba(0,0,0,0.5)] h-full overflow-hidden" ref={ref}>
      <div className="h-full p-8 flex flex-col relative">
        <div className="absolute bottom-4 right-4 text-xs text-slate-500 font-mono">{props.number}</div>
        <div className="absolute inset-0 opacity-[0.03] pointer-events-none bg-[url('https://www.transparenttextures.com/patterns/graphy.png')]"></div>
        <div className="flex-1 overflow-y-auto custom-scrollbar pr-2 relative z-10">
           {props.children}
        </div>
      </div>
    </div>
  );
});
Page.displayName = "Page";

// --- 📘 COVER COMPONENT ---
const Cover = forwardRef((props: any, ref: any) => {
  return (
    <div className="demoPage bg-[#0f172a] text-white border-r-4 border-slate-800 h-full shadow-2xl flex flex-col items-center justify-center p-10 text-center relative overflow-hidden" ref={ref}>
      <div className="absolute inset-0 opacity-20 bg-[url('https://www.transparenttextures.com/patterns/leather.png')]"></div>
      <div className="z-10 border-2 border-yellow-500/30 p-10 rounded-xl w-full h-full flex flex-col items-center justify-center bg-[#0f172a]/50 backdrop-blur-sm">
          <Book size={80} className="text-yellow-500 mb-8 drop-shadow-[0_0_15px_rgba(234,179,8,0.5)]"/>
          <h1 className="text-4xl font-bold mb-3 uppercase tracking-[0.2em] text-yellow-500 font-serif">Lab Record</h1>
          <div className="w-24 h-1 bg-yellow-500/50 mb-8 rounded-full shadow-[0_0_10px_rgba(234,179,8,0.5)]"></div>
          <h2 className="text-2xl font-bold text-white mb-4 tracking-wide">Machine Learning</h2>
          <p className="text-slate-400 text-sm italic">2025 - 2026 Academic Year</p>
      </div>
    </div>
  );
});
Cover.displayName = "Cover";

// --- 🚀 MASTER BOOK COMPONENT ---
export default function MasterManualPage() {
  const router = useRouter();
  const bookRef = useRef<any>(null);
  const [mounted, setMounted] = useState(false);
  
  const labList = Object.keys(manuals).map(key => ({ id: key, ...manuals[key] }));

  useEffect(() => { setMounted(true); }, []);

  if (!mounted) return <div className="min-h-screen bg-[#0f172a]" />;

  return (
    <div className="min-h-screen bg-[#0f172a] text-white font-sans flex flex-col h-screen overflow-hidden">
      
      {/* Header */}
      <header className="h-16 bg-[#1e293b]/80 backdrop-blur border-b border-slate-800 flex items-center justify-between px-6 z-20 shadow-lg">
        <div className="flex items-center gap-4">
            <Link href="/" className="p-2 hover:bg-slate-700 rounded-lg transition-colors group">
                <ArrowLeft size={20} className="group-hover:-translate-x-1 transition-transform"/>
            </Link>
            <span className="font-bold text-slate-200 tracking-wide">Master Lab Record</span>
        </div>
        <div className="flex gap-2">
            <button onClick={() => bookRef.current.pageFlip().flipPrev()} className="bg-slate-800 hover:bg-blue-600 text-slate-300 hover:text-white p-2 rounded-lg transition-all border border-slate-700">
                <ChevronLeft size={20}/>
            </button>
            <button onClick={() => bookRef.current.pageFlip().flipNext()} className="bg-slate-800 hover:bg-blue-600 text-slate-300 hover:text-white p-2 rounded-lg transition-all border border-slate-700">
                <ChevronRight size={20}/>
            </button>
        </div>
      </header>

      {/* Book Container */}
      <div className="flex-1 flex items-center justify-center bg-[#0b1120] overflow-hidden p-4 relative perspective-1000">
        <div className="absolute w-[800px] h-[800px] bg-blue-600/10 rounded-full blur-[100px] pointer-events-none" />

        <HTMLFlipBook 
            width={800} height={950} size="stretch" minWidth={400} maxWidth={1000} minHeight={500} maxHeight={1400}
            maxShadowOpacity={0.5} showCover={true} mobileScrollSupport={false} className="shadow-2xl" ref={bookRef}
        >
            <Cover />

            {/* INDEX */}
            <Page number="i">
                <h2 className="text-3xl font-bold text-blue-400 mb-8 font-serif border-b border-slate-700 pb-4 text-center">Table of Contents</h2>
                <div className="space-y-4 pr-4">
                    {labList.map((lab, index) => (
                        <div key={lab.id} className="group flex items-baseline gap-4 border-b border-dashed border-slate-800 pb-3 hover:bg-slate-800/30 transition-colors p-2 rounded-lg cursor-pointer">
                            <span className="text-blue-500 text-xs font-bold font-mono px-2 py-1 bg-blue-500/10 rounded border border-blue-500/20">EXP {String(index + 1).padStart(2, '0')}</span>
                            <span className="text-slate-300 font-bold text-sm flex-1 group-hover:text-white transition-colors">{lab.title}</span>
                        </div>
                    ))}
                </div>
            </Page>

            {/* LAB PAGES */}
            {labList.map((lab, labIndex) => (
                [
                    <Page key={`title-${lab.id}`} number={`${labIndex + 1}.1`}>
                        <div className="h-full flex flex-col justify-center text-center border-4 border-double border-slate-700 p-8 bg-[#0f172a]/40 rounded-xl backdrop-blur-sm">
                            <h3 className="text-blue-500 uppercase tracking-[0.2em] text-xs font-bold mb-6">Experiment {labIndex + 1}</h3>
                            <h1 className="text-4xl font-bold text-white font-serif mb-8 leading-tight">{lab.title}</h1>
                            <div className="w-16 h-1 bg-blue-500 mx-auto mb-8 shadow-[0_0_15px_rgba(59,130,246,0.5)]"></div>
                            <div className="text-left bg-slate-900/50 p-6 rounded-lg border border-slate-800">
                                <h4 className="text-xs font-bold text-blue-400 uppercase mb-3 flex items-center gap-2"><Terminal size={14}/> Aim</h4>
                                <p className="text-slate-300 italic text-sm leading-relaxed">{lab.aim}</p>
                            </div>
                        </div>
                    </Page>,

                    ...lab.steps.map((step: any, stepIndex: number) => (
                        <Page key={`${lab.id}-step-${stepIndex}`} number={`${labIndex + 1}.${stepIndex + 2}`}>
                            <h3 className="text-xl font-bold text-white mb-4 flex items-center gap-3">
                                <span className="bg-blue-600 text-white w-8 h-8 rounded-lg flex items-center justify-center text-sm font-mono shadow-lg shadow-blue-900/50 border border-blue-400">{stepIndex + 1}</span>
                                {step.title}
                            </h3>
                            <p className="text-slate-400 mb-6 text-sm leading-relaxed border-l-2 border-slate-700 pl-4">{step.explanation}</p>
                            
                            {/* 🛡️ CODE BLOCK: LINE NUMBERS ONLY */}
                            <div className="relative bg-[#0b1120] rounded-xl overflow-hidden border border-slate-700 shadow-2xl my-6 group ring-1 ring-white/5 flex flex-col" onContextMenu={(e) => e.preventDefault()}>
                                
                                <div className="bg-[#1e293b] px-4 py-2 flex justify-between items-center border-b border-slate-700 select-none">
                                    <div className="flex gap-2">
                                        <div className="w-3 h-3 rounded-full bg-red-500/80" /><div className="w-3 h-3 rounded-full bg-yellow-500/80" /><div className="w-3 h-3 rounded-full bg-green-500/80" />
                                    </div>
                                    <div className="text-[10px] text-slate-400 font-mono flex items-center gap-1.5 uppercase tracking-wider"><Shield size={12} className="text-blue-500"/> Python 3.x</div>
                                </div>

                                <div className="relative flex overflow-hidden">
                                    {/* Line Numbers (Left Column) */}
                                    <div className="bg-[#0f172a] text-slate-600 text-right pr-3 pl-2 py-4 font-mono text-sm border-r border-slate-700 select-none opacity-60">
                                        {step.code.split('\n').map((_: any, i: number) => <div key={i} className="leading-6">{i + 1}</div>)}
                                    </div>

                                    {/* Code Content (Right Column) */}
                                    <div className="flex-1 relative overflow-x-auto custom-scrollbar bg-[#020617]">
                                        <pre className="p-4 font-mono text-sm text-green-400/90 leading-6 whitespace-pre tab-4" 
                                             style={{ userSelect: 'none', WebkitUserSelect: 'none' }} 
                                             onCopy={(e) => { e.preventDefault(); return false; }}>
                                            {/* Simple Highlighting */}
                                            {step.code.split('\n').map((line: string, i: number) => (
                                                <div key={i}>
                                                    {line.startsWith('#') ? <span className="text-slate-500 italic">{line}</span> : 
                                                     line.split(/(\s+)/).map((part, j) => {
                                                        if (['import', 'from', 'def', 'return', 'if', 'else', 'elif', 'for', 'in', 'print', 'class', 'try', 'except'].includes(part)) return <span key={j} className="text-purple-400">{part}</span>;
                                                        if (['True', 'False', 'None', 'self'].includes(part)) return <span key={j} className="text-blue-400 italic">{part}</span>;
                                                        if (part.startsWith("'") || part.startsWith('"')) return <span key={j} className="text-orange-300">{part}</span>;
                                                        return <span key={j}>{part}</span>;
                                                    })}
                                                </div>
                                            ))}
                                        </pre>
                                    </div>
                                </div>
                            </div>
                        </Page>
                    )),

                    <Page key={`end-${lab.id}`} number={`${labIndex + 1}.End`}>
                         <div className="h-full flex flex-col items-center justify-center text-center">
                            <div className="w-24 h-24 bg-green-500/10 rounded-full flex items-center justify-center mb-6 ring-1 ring-green-500/30">
                                <Terminal size={48} className="text-green-500 drop-shadow-[0_0_8px_rgba(34,197,94,0.5)]" />
                            </div>
                            <h2 className="text-2xl font-bold text-white mb-2">Module Completed</h2>
                            <p className="text-slate-400 text-sm mb-8">You have successfully reviewed {lab.title}.</p>
                            {labList[labIndex + 1] ? (
                                <div className="text-white font-serif text-lg">Next: {labList[labIndex + 1].title} <ChevronRight size={16} className="inline text-blue-500"/></div>
                            ) : (
                                <div className="text-green-400 font-bold border border-green-500/30 px-4 py-2 rounded-lg bg-green-500/10">End of Syllabus</div>
                            )}
                         </div>
                    </Page>
                ]
            ))}
            
            <div className="demoPage bg-[#0f172a] text-slate-500 border-l-4 border-slate-800 h-full shadow-2xl flex flex-col items-center justify-center p-10 text-center">
                 <div className="opacity-30"><Book size={64} className="mx-auto mb-6"/><p className="text-sm font-serif tracking-widest uppercase">End of Record</p></div>
            </div>

        </HTMLFlipBook>
      </div>
    </div>
  );
}