"use client";

import { useState, useEffect, forwardRef, useRef } from "react";
import { auth } from "../../../../lib/firebase"; 
import { onAuthStateChanged } from "firebase/auth";
import { useRouter } from "next/navigation";
import { manuals } from "../../../../data/manuals";
import { FileDown, ArrowLeft, Shield, Book, ChevronRight, ChevronLeft, Terminal } from "lucide-react";
import Link from "next/link";
import HTMLFlipBook from "react-pageflip";

// --- 📄 REUSABLE PAGE COMPONENT ---
const Page = forwardRef((props: any, ref: any) => {
  return (
    <div className="demoPage bg-[#f8fafc] text-slate-900 border-r border-slate-300 shadow-inner h-full overflow-hidden" ref={ref}>
      <div className="h-full p-8 flex flex-col relative">
        <div className="absolute bottom-4 right-4 text-xs text-slate-400 font-mono">{props.number}</div>
        <div className="flex-1 overflow-y-auto custom-scrollbar pr-2">
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
    <div className="demoPage bg-[#1e293b] text-white border-r-4 border-slate-700 h-full shadow-2xl flex flex-col items-center justify-center p-10 text-center relative overflow-hidden" ref={ref}>
      <div className="absolute inset-0 opacity-10 bg-[url('https://www.transparenttextures.com/patterns/leather.png')]"></div>
      <div className="z-10 border-2 border-yellow-500/30 p-8 rounded-lg w-full h-full flex flex-col items-center justify-center">
          <Book size={64} className="text-yellow-500 mb-6"/>
          <h1 className="text-3xl font-bold mb-2 uppercase tracking-widest text-yellow-500 font-serif">Lab Record</h1>
          <div className="w-16 h-1 bg-yellow-500/50 mb-6 rounded-full"></div>
          <h2 className="text-xl font-bold text-white mb-4">{props.title}</h2>
          <p className="text-slate-400 text-sm italic">Confidential Training Material</p>
          <p className="mt-8 text-xs font-mono text-slate-500">Exp ID: {props.labId}</p>
      </div>
    </div>
  );
});
Cover.displayName = "Cover";


// --- 🚀 MAIN COMPONENT ---
export default function ManualPage({ params }: { params: { labId: string } }) {
  const [loading, setLoading] = useState(true);
  const router = useRouter();
  const bookRef = useRef<any>(null); 
  
  const manual = manuals[params.labId];

  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, (user) => {
      if (!user) router.push("/login");
      setLoading(false);
    });
    return () => unsubscribe();
  }, []);

  const handleDownload = () => {
    if (!manual?.datasetUrl) {
        alert("This experiment uses a built-in dataset (e.g., from sklearn). No download required.");
        return;
    }
    // Logic for actual file download would go here
  };

  if (loading) return <div className="min-h-screen bg-[#0f172a] text-white flex items-center justify-center">Loading Book...</div>;

  if (!manual) return <div className="text-white p-10">Manual not found.</div>;

  return (
    <div className="min-h-screen bg-[#0f172a] text-white font-sans flex flex-col h-screen overflow-hidden">
      
      {/* Top Bar */}
      <header className="h-16 bg-[#1e293b] border-b border-slate-700 flex items-center justify-between px-6 z-20 shadow-lg">
        <div className="flex items-center gap-4">
            <Link href="/dashboard" className="p-2 hover:bg-slate-700 rounded-lg transition-colors">
                <ArrowLeft size={20} />
            </Link>
            <span className="font-bold text-slate-200">Lab Manual Viewer</span>
        </div>
        
        <div className="flex gap-2">
            <button onClick={() => bookRef.current.pageFlip().flipPrev()} className="bg-slate-700 hover:bg-slate-600 p-2 rounded-lg transition-colors">
                <ChevronLeft size={20}/>
            </button>
            <button onClick={() => bookRef.current.pageFlip().flipNext()} className="bg-slate-700 hover:bg-slate-600 p-2 rounded-lg transition-colors">
                <ChevronRight size={20}/>
            </button>
            {manual.datasetUrl && (
                <>
                <div className="w-px h-8 bg-slate-600 mx-2"></div>
                <button 
                    onClick={handleDownload}
                    className="bg-green-600 hover:bg-green-500 text-white px-4 py-2 rounded-lg text-sm font-bold flex items-center gap-2 shadow-lg"
                >
                    <FileDown size={16} /> Dataset
                </button>
                </>
            )}
        </div>
      </header>

      {/* Book Container */}
      <div className="flex-1 flex items-center justify-center bg-[#0f172a] overflow-hidden p-4 relative perspective-1000">
        <div className="absolute w-[500px] h-[500px] bg-blue-500/10 rounded-full blur-3xl pointer-events-none" />

        {/* @ts-ignore */}
        <HTMLFlipBook 
            width={500} 
            height={700} 
            size="stretch"
            minWidth={300}
            maxWidth={600}
            minHeight={400}
            maxHeight={800}
            maxShadowOpacity={0.5}
            showCover={true}
            mobileScrollSupport={true}
            className="shadow-2xl"
            ref={bookRef}
        >
            
            {/* 1. FRONT COVER */}
            <Cover title={manual.title} labId={params.labId} />

            {/* 2. AIM PAGE */}
            <Page number="1">
                <h2 className="text-2xl font-bold text-slate-800 mb-4 font-serif border-b border-slate-300 pb-2">Experiment Aim</h2>
                <p className="text-slate-700 leading-relaxed text-lg font-serif">
                    {manual.aim}
                </p>
                <div className="mt-8 bg-blue-50 p-4 rounded-lg border border-blue-100">
                    <h3 className="text-blue-800 font-bold mb-2 text-sm uppercase">Prerequisites</h3>
                    <ul className="list-disc list-inside text-blue-700 text-sm space-y-1">
                        <li>Python 3.x Installed</li>
                        <li>Pandas & NumPy Libraries</li>
                        <li>Basic understanding of Dataframes</li>
                    </ul>
                </div>
            </Page>

            {/* 3. CODE PAGES (Dynamic Generation) */}
            {manual.steps.map((step: any, index: number) => (
                <Page number={index + 2} key={index}>
                    <h3 className="text-lg font-bold text-slate-800 mb-2 flex items-center gap-2">
                        <span className="bg-slate-800 text-white w-6 h-6 rounded-full flex items-center justify-center text-xs">{index + 1}</span>
                        {step.title}
                    </h3>
                    <p className="text-slate-600 mb-4 text-sm leading-relaxed">{step.explanation}</p>
                    
                    <div 
                        className="relative bg-[#1e293b] rounded-lg overflow-hidden border border-slate-600 shadow-md my-4"
                        onCopy={(e) => e.preventDefault()}
                        onContextMenu={(e) => e.preventDefault()}
                    >
                        <div className="bg-[#0f172a] px-3 py-1.5 flex justify-between items-center border-b border-slate-700">
                            <div className="flex gap-1.5">
                                <div className="w-2.5 h-2.5 rounded-full bg-red-500" />
                                <div className="w-2.5 h-2.5 rounded-full bg-yellow-500" />
                            </div>
                            <div className="text-[10px] text-slate-500 font-mono flex items-center gap-1">
                                <Shield size={10}/> Protected
                            </div>
                        </div>
                        <div className="p-3 relative">
                           <div className="absolute inset-0 z-10 w-full h-full cursor-not-allowed"></div>
                           <pre className="font-mono text-sm text-blue-300 overflow-x-auto select-none">
                                {step.code}
                           </pre>
                        </div>
                    </div>
                </Page>
            ))}

            {/* 4. CONCLUSION PAGE (Replaces Output) */}
            <Page number={manual.steps.length + 2}>
                <h2 className="text-xl font-bold text-slate-800 mb-4 font-serif border-b border-slate-300 pb-2">Lab Completed</h2>
                
                <div className="p-6 bg-blue-50 rounded-xl border border-blue-100 text-center flex flex-col items-center">
                    <Terminal size={48} className="text-blue-500 mb-4" />
                    <p className="text-slate-700 mb-4 font-serif italic text-lg">
                        "You have successfully reviewed the code logic."
                    </p>
                    <p className="text-slate-600 text-sm mb-6 leading-relaxed">
                        Execute this code in your local Python environment or the provided Playground to verify the results. Ensure you check for outliers as specified in the Aim.
                    </p>
                    
                    <Link href={`/dashboard/exam/${params.labId}?type=cie`}>
                        <div className="inline-block px-8 py-3 bg-slate-800 text-white rounded-full text-sm font-bold tracking-wider shadow-lg cursor-pointer hover:bg-slate-700 transition-colors">
                            Proceed to CIE Assessment
                        </div>
                    </Link>
                </div>

                <div className="mt-auto text-center pb-8">
                     <p className="text-slate-400 text-xs">End of Experiment {params.labId}</p>
                </div>
            </Page>

            {/* 5. BACK COVER */}
            <div className="demoPage bg-[#1e293b] text-slate-500 border-l-4 border-slate-700 h-full shadow-2xl flex flex-col items-center justify-center p-10 text-center">
                 <div className="opacity-20">
                    <Book size={48} className="mx-auto mb-4"/>
                    <p className="text-sm font-serif">Property of Lab Dept.</p>
                 </div>
            </div>

        </HTMLFlipBook>
      </div>
    </div>
  );
}