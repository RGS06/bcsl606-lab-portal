"use client";
import { useState } from "react";
import { GoogleGenerativeAI } from "@google/generative-ai";
import ReactMarkdown from "react-markdown";
import { Bot, X, Send, Sparkles } from "lucide-react";

export default function ChatBot() {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState<{role: string, content: string}[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  const handleSend = async () => {
    if (!input) return;
    const newMessages = [...messages, { role: "user", content: input }];
    setMessages(newMessages);
    setInput("");
    setLoading(true);

    try {
      const genAI = new GoogleGenerativeAI(process.env.NEXT_PUBLIC_GEMINI_API_KEY!);
      const model = genAI.getGenerativeModel({ model: "gemini-pro" });
      const result = await model.generateContent(input);
      const response = await result.response;
      setMessages([...newMessages, { role: "bot", content: response.text() }]);
    } catch (error) {
      setMessages([...newMessages, { role: "bot", content: "Error: Could not connect to AI." }]);
    }
    setLoading(false);
  };

  return (
    <div className="fixed bottom-6 right-6 z-50 flex flex-col items-end">
      {/* CHAT WINDOW */}
      {isOpen && (
        <div className="mb-4 w-96 h-[500px] bg-[#0f172a]/90 backdrop-blur-xl border border-white/10 rounded-2xl shadow-2xl flex flex-col overflow-hidden animate-in slide-in-from-bottom-10 fade-in duration-300">
          
          {/* Header */}
          <div className="p-4 bg-gradient-to-r from-blue-600/80 to-purple-600/80 flex justify-between items-center">
            <div className="flex items-center gap-2 text-white font-bold">
              <Sparkles size={18} /> AI Assistant
            </div>
            <button onClick={() => setIsOpen(false)} className="text-white/70 hover:text-white">
              <X size={20} />
            </button>
          </div>

          {/* Messages Area */}
          <div className="flex-1 overflow-y-auto p-4 space-y-4 custom-scrollbar">
            {messages.length === 0 && (
              <div className="text-center text-slate-400 mt-10 text-sm px-6">
                <Bot size={40} className="mx-auto mb-3 opacity-50" />
                <p>Hello! I am your Machine Learning AI tutor. Ask me anything about this lab!</p>
              </div>
            )}
            {messages.map((m, index) => (
              <div key={index} className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}>
                <div className={`max-w-[85%] p-3 rounded-2xl text-sm ${
                  m.role === "user" 
                    ? "bg-blue-600 text-white rounded-br-none" 
                    : "bg-white/10 text-slate-200 rounded-bl-none border border-white/5"
                }`}>
                  <ReactMarkdown>{m.content}</ReactMarkdown>
                </div>
              </div>
            ))}
            {loading && <div className="text-slate-400 text-xs animate-pulse">Thinking...</div>}
          </div>

          {/* Input Area */}
          <div className="p-4 border-t border-white/10 bg-black/20">
            <div className="flex gap-2">
              <input 
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleSend()}
                placeholder="Ask a question..."
                className="flex-1 bg-white/5 border border-white/10 rounded-xl px-4 text-sm text-white focus:outline-none focus:border-blue-500/50"
              />
              <button 
                onClick={handleSend}
                className="p-2 bg-blue-600 rounded-xl text-white hover:bg-blue-500 transition-colors"
              >
                <Send size={18} />
              </button>
            </div>
          </div>
        </div>
      )}

      {/* TOGGLE BUTTON */}
      {!isOpen && (
        <button 
          onClick={() => setIsOpen(true)}
          className="group flex items-center gap-3 bg-blue-600 hover:bg-blue-500 text-white px-5 py-4 rounded-full shadow-lg shadow-blue-600/30 transition-all hover:scale-105"
        >
          <span className="font-bold pr-1">Ask AI</span>
          <Bot size={24} />
        </button>
      )}
    </div>
  );
}