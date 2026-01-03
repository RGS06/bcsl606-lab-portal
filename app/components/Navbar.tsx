// app/components/Navbar.tsx
"use client";

import Link from "next/link";
import { BrainCircuit, ChevronRight } from "lucide-react";

export default function Navbar() {
  return (
    <nav className="fixed top-0 left-0 w-full z-50 border-b border-white/5 bg-[#0f172a]/80 backdrop-blur-md">
      <div className="max-w-7xl mx-auto px-6 h-20 flex items-center justify-between">
        
        {/* Logo */}
        <Link href="/" className="flex items-center gap-2 group">
          <div className="p-2 bg-blue-600/20 rounded-lg text-blue-400 group-hover:bg-blue-600 group-hover:text-white transition-all">
            <BrainCircuit size={24} />
          </div>
          <span className="font-bold text-lg text-white tracking-tight">Machine Learning<span className="text-blue-500"> Lab</span></span>
        </Link>
      </div>
    </nav>
  );
}