// app/page.tsx
"use client"; 

import Navbar from "./components/Navbar";
import Footer from "./components/Footer";
import Link from "next/link";
// 👇 Added BookOpen to imports
import { ArrowRight, Brain, Network, Database, Code2, Terminal, Cpu, FileText, BookOpen } from "lucide-react";

export default function HomePage() {
  return (
    <div className="min-h-screen bg-[#0f172a] text-white font-sans selection:bg-blue-500/30">
      <Navbar />

      {/* ================= HERO SECTION ================= */}
      <section className="relative pt-32 pb-20 lg:pt-48 lg:pb-32 overflow-hidden">
        {/* Background Glows */}
        <div className="absolute top-20 left-1/2 -translate-x-1/2 w-[800px] h-[500px] bg-blue-600/20 rounded-full blur-[200px] pointer-events-none" />
        
        <div className="max-w-10xl mx-auto px-6 text-center relative z-10">
    {/* 🏷️ COURSE BADGE: Increased from text-sm to text-lg */}
    <div className="inline-block mb-8 px-8 py-2.5 rounded-full border border-blue-500/30 bg-blue-500/10 text-blue-400 text-lg md:text-xl font-bold tracking-widest uppercase">
        BCSL606 - Machine Learning Lab
    </div>
          
          <h1 className="text-5xl md:text-7xl font-extrabold tracking-tight mb-8 leading-tight">
            The Future is <br />
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-purple-400">
              Data-Driven.
            </span>
          </h1>
          
          <div className="max-w-3xl mx-auto text-center">
  <h2 className="text-4xl md:text-5xl font-extrabold text-white mb-6 tracking-tight">
    Master the <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-cyan-300">Core Algorithms</span>
  </h2>
  
 <p className="max-w-4xl mx-auto text-slate-400 text-xl md:text-2xl leading-relaxed mb-12 border-l-4 md:border-l-0 md:border-b-4 border-blue-500/20 pl-6 md:pl-0 md:pb-6 text-justify md:text-center italic font-light">
  From data visualization and dimensionality reduction to advanced classification and clustering. 
  Master the core implementations of 
  <span className="text-blue-400 font-mono font-medium mx-1">Find-S</span>, 
  <span className="text-blue-400 font-mono font-medium mx-1">Naive Bayes</span>, 
  <span className="text-blue-400 font-mono font-medium mx-1">KNN</span>, and 
  <span className="text-blue-400 font-mono font-medium mx-1">Regression</span> 
  algorithms.
</p>
</div>
          
          <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
            {/* 1. LOGIN BUTTON */}
            <Link 
              href="/login" 
              className="px-8 py-4 bg-blue-600 hover:bg-blue-500 text-white rounded-full font-bold text-lg transition-all shadow-[0_0_30px_rgba(37,99,235,0.3)] hover:scale-105 flex items-center gap-2 w-full sm:w-auto justify-center"
            >
              Enter Laboratory <ArrowRight size={20} />
            </Link>

            {/* 2. NEW MANUAL BUTTON (Golden/Amber Theme)  */}
            <Link 
              href="/manual" 
              className="px-8 py-4 bg-amber-600 hover:bg-amber-500 text-white rounded-full font-bold text-lg transition-all shadow-[0_0_30px_rgba(217,119,6,0.3)] hover:scale-105 flex items-center gap-2 w-full sm:w-auto justify-center"
            >
              <BookOpen size={20} /> Open Manual
            </Link>
            
            {/* 3. SYLLABUS BUTTON */}
            <a 
              href="/syllabus.pdf" 
              target="_blank" 
              rel="noopener noreferrer"
              className="px-8 py-4 bg-[#1e293b] hover:bg-[#334155] text-white border border-slate-700 rounded-full font-bold text-lg transition-all flex items-center gap-2 cursor-pointer group w-full sm:w-auto justify-center"
            >
              View Syllabus <FileText size={20} className="text-slate-400 group-hover:text-white transition-colors"/>
            </a>
          </div>
        </div>
      </section>

      {/* ================= WHAT IS ML SECTION ================= */}
      <section id="about" className="py-24 bg-[#0b1120] relative">
        <div className="max-w-7xl mx-auto px-6">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-bold mb-4">Core Concepts</h2>
            <p className="text-slate-400 max-w-2xl mx-auto">
              Machine Learning is the study of computer algorithms that improve automatically through experience. We focus on three main pillars.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {/* Card 1 */}
            <div className="p-8 bg-[#1e293b] rounded-2xl border border-slate-700 hover:border-blue-500/50 transition-all group">
              <div className="w-14 h-14 bg-blue-900/30 rounded-xl flex items-center justify-center mb-6 group-hover:scale-110 transition-transform">
                <Database className="text-blue-400" size={28} />
              </div>
              <h3 className="text-xl font-bold mb-3 group-hover:text-blue-400 transition-colors">Supervised Learning</h3>
              <p className="text-slate-400 leading-relaxed">
                Training models on labeled data. Algorithms learn to map inputs to outputs based on example input-output pairs.
              </p>
            </div>

            {/* Card 2 */}
            <div className="p-8 bg-[#1e293b] rounded-2xl border border-slate-700 hover:border-purple-500/50 transition-all group">
              <div className="w-14 h-14 bg-purple-900/30 rounded-xl flex items-center justify-center mb-6 group-hover:scale-110 transition-transform">
                <Brain className="text-purple-400" size={28} />
              </div>
              <h3 className="text-xl font-bold mb-3 group-hover:text-purple-400 transition-colors">Unsupervised Learning</h3>
              <p className="text-slate-400 leading-relaxed">
                Finding hidden patterns in unlabeled data. Used for clustering, dimensionality reduction, and association.
              </p>
            </div>

            {/* Card 3 */}
            <div className="p-8 bg-[#1e293b] rounded-2xl border border-slate-700 hover:border-cyan-500/50 transition-all group">
              <div className="w-14 h-14 bg-cyan-900/30 rounded-xl flex items-center justify-center mb-6 group-hover:scale-110 transition-transform">
                <Network className="text-cyan-400" size={28} />
              </div>
              <h3 className="text-xl font-bold mb-3 group-hover:text-cyan-400 transition-colors">Neural Networks</h3>
              <p className="text-slate-400 leading-relaxed">
                Algorithms modeled after the human brain, designed to recognize patterns and solve complex perceptual problems.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* ================= SYLLABUS PREVIEW ================= */}
      <section id="syllabus-preview" className="py-24 bg-[#0f172a]">
        <div className="max-w-7xl mx-auto px-6 flex flex-col md:flex-row items-center gap-16">
          
          <div className="w-full md:w-1/2">
            <h2 className="text-3xl md:text-4xl font-bold mb-6">What You Will Learn</h2>
            <div className="space-y-6">
              <div className="flex gap-4">
                <div className="mt-1"><Terminal className="text-blue-500" /></div>
                <div>
                  <h4 className="font-bold text-lg">Python & Libraries</h4>
                  <p className="text-slate-400">Master NumPy, Pandas, and Matplotlib for data manipulation.</p>
                </div>
              </div>
              <div className="flex gap-4">
                <div className="mt-1"><Cpu className="text-purple-500" /></div>
                <div>
                  <h4 className="font-bold text-lg">Model Building</h4>
                  <p className="text-slate-400">Implement Find-S, Candidate Elimination, and ID3 algorithms from scratch.</p>
                </div>
              </div>
              <div className="flex gap-4">
                <div className="mt-1"><Network className="text-pink-500" /></div>
                <div>
                  <h4 className="font-bold text-lg">Deep Learning Basics</h4>
                  <p className="text-slate-400">Introduction to Backpropagation and Artificial Neural Networks (ANN).</p>
                </div>
              </div>
            </div>
          </div>

          <div className="w-full md:w-1/2">
             <div className="relative bg-gradient-to-br from-blue-600 to-purple-600 rounded-2xl p-1">
                <div className="bg-[#1a1a2e] rounded-xl p-8 overflow-hidden relative">
                   {/* Fake Code Snippet Visual */}
                   <div className="flex gap-2 mb-4">
                     <div className="w-3 h-3 rounded-full bg-red-500" />
                     <div className="w-3 h-3 rounded-full bg-yellow-500" />
                     <div className="w-3 h-3 rounded-full bg-green-500" />
                   </div>
                   <div className="font-mono text-sm text-slate-300 space-y-2">
                      <p><span className="text-purple-400">import</span> pandas <span className="text-purple-400">as</span> pd</p>
                      <p><span className="text-purple-400">from</span> sklearn.model_selection <span className="text-purple-400">import</span> train_test_split</p>
                      <p>&nbsp;</p>
                      <p><span className="text-slate-500"># Load Dataset</span></p>
                      <p>data = pd.read_csv(<span className="text-green-400">'iris.csv'</span>)</p>
                      <p>X = data.drop(<span className="text-green-400">'species'</span>, axis=1)</p>
                      <p>y = data[<span className="text-green-400">'species'</span>]</p>
                      <p>&nbsp;</p>
                      <p><span className="text-slate-500"># Train Model</span></p>
                      <p>model = DecisionTreeClassifier()</p>
                      <p>model.fit(X_train, y_train)</p>
                      <p><span className="text-blue-400 animate-pulse">_</span></p>
                   </div>
                </div>
             </div>
          </div>

        </div>
      </section>

      <Footer />
    </div>
  );
}