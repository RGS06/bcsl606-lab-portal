"use client";

import { useEffect, useState } from "react";
// 👇 Added deleteField and updateDoc
import { db, auth } from "../../lib/firebase";
import { collection, getDocs, doc, setDoc, getDoc, updateDoc, deleteField, arrayUnion, arrayRemove, query, where } from "firebase/firestore";
import { onAuthStateChanged, signOut } from "firebase/auth";
import { useRouter } from "next/navigation";
import { labs } from "../../data/syllabus";
import { CheckCircle, Users, LogOut, LayoutDashboard, Search, CalendarDays, RotateCcw, Save, Loader2, Trash2 } from "lucide-react";

type Student = {
  id: string;
  name: string;
  email: string;
  usn: string;
  section: string;
  batch: string;
};

export default function AdminDashboard() {
  const [students, setStudents] = useState<Student[]>([]);
  const [selectedLab, setSelectedLab] = useState(labs[0].id);
  
  // Controls
  const [selectedBatch, setSelectedBatch] = useState("1"); 
  const [selectedSection, setSelectedSection] = useState("A");
  const [selectedDate, setSelectedDate] = useState(new Date().toISOString().split('T')[0]); 

  const [presentIDs, setPresentIDs] = useState<string[]>([]);
  const [studentScores, setStudentScores] = useState<any>({});
  
  const [loading, setLoading] = useState(true);
  const [isSaving, setIsSaving] = useState(false);
  const [searchTerm, setSearchTerm] = useState("");
  const router = useRouter();

  // 1. Auth Check
  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, (user) => {
      if (!user) router.push("/login");
      else fetchData();
    });
    return () => unsubscribe();
  }, []);

  // 2. Fetch Data
  const fetchData = async () => {
    setLoading(true);
    try {
      // Fetch Students
      const q = query(collection(db, "users"), where("role", "==", "student"));
      const querySnapshot = await getDocs(q);
      const studentList: Student[] = [];
      querySnapshot.forEach((doc) => {
        studentList.push({ id: doc.id, ...doc.data() } as Student);
      });
      setStudents(studentList);
      
      // Fetch Attendance & Scores
      await fetchAttendance(selectedLab, selectedSection, selectedBatch, selectedDate);
    } catch (error) {
      console.error("Error fetching data:", error);
    }
    setLoading(false);
  };

  // 3. Fetch Attendance & Scores
  const fetchAttendance = async (labId: string, section: string, batch: string, date: string) => {
    const sessionID = `${labId}_${section}_${batch}_${date}`;
    
    // A. Attendance
    const docSnap = await getDoc(doc(db, "attendance", sessionID));
    setPresentIDs(docSnap.exists() ? docSnap.data().presentIds || [] : []);

    // B. Scores (Fetch all scores for this lab)
    const scoresQuery = query(collection(db, "scores"), where("labId", "==", labId));
    const scoreDocs = await getDocs(scoresQuery);
    
    const scoreMap: any = {};
    scoreDocs.forEach(doc => {
        const data = doc.data();
        scoreMap[data.studentId] = { 
            cie: data.cieScore, 
            viva: data.vivaScore,
            // We can add timestamps or status here if needed
        };
    });
    setStudentScores(scoreMap);
  };

  // 4. Mark Attendance (Local State)
  const toggleAttendance = (studentId: string) => {
    if (presentIDs.includes(studentId)) {
      setPresentIDs(prev => prev.filter(id => id !== studentId));
    } else {
      setPresentIDs(prev => [...prev, studentId]);
    }
  };

  // 5. RESET EXAM (Release the Lock) 🔓
  const resetExam = async (studentId: string, type: "cie" | "viva") => {
    // 1. Confirm Intent
    const confirmMessage = `⚠️ RESET ${type.toUpperCase()}?\n\nThis will DELETE the score and unlock the exam for retaking.\nAre you sure?`;
    if (!window.confirm(confirmMessage)) return;

    // 2. Optimistic Update (Remove badge immediately)
    setStudentScores((prev: any) => {
        const newScores = { ...prev };
        if (newScores[studentId]) {
            newScores[studentId] = { 
                ...newScores[studentId], 
                [type]: undefined // Remove the score locally
            }; 
        }
        return newScores;
    });

    // 3. Database Update
    try {
        const docRef = doc(db, "scores", `${selectedLab}_${studentId}`);
        await updateDoc(docRef, {
            [type === "cie" ? "cieScore" : "vivaScore"]: deleteField(),
            [type === "cie" ? "cieStatus" : "vivaStatus"]: deleteField(),
            [type === "cie" ? "cieTimestamp" : "vivaTimestamp"]: deleteField(),
        });
        // alert(`✅ ${type.toUpperCase()} reset successfully.`);
    } catch (error) {
        console.error("Error resetting score:", error);
        alert("❌ Failed to reset score. Check console.");
        fetchData(); // Revert on error
    }
  };

  // 6. Bulk Actions
  const markAllPresent = () => {
    if (filteredStudents.length === 0) return;
    const allIDs = filteredStudents.map(s => s.id);
    setPresentIDs(allIDs); 
  };

  const clearAll = () => {
    setPresentIDs([]);
  };

  // 7. Manual Save
  const saveAttendance = async () => {
    setIsSaving(true);
    const sessionID = `${selectedLab}_${selectedSection}_${selectedBatch}_${selectedDate}`;
    
    try {
      await setDoc(doc(db, "attendance", sessionID), {
        labId: selectedLab,
        section: selectedSection,
        batch: selectedBatch,
        date: selectedDate,
        presentIds: presentIDs,
        updatedAt: new Date().toISOString()
      }, { merge: true });
      
      await new Promise(r => setTimeout(r, 500));
      alert("✅ Attendance Saved Successfully!");
    } catch (error) {
      console.error("Save failed:", error);
      alert("❌ Error saving attendance.");
    } finally {
      setIsSaving(false);
    }
  };

  const handleLogout = async () => {
    await signOut(auth);
    router.push("/login");
  };

  // Filter & Sort
  const filteredStudents = students
    .filter(s => 
      s.section === selectedSection &&
      s.batch === selectedBatch &&
      (s.name.toLowerCase().includes(searchTerm.toLowerCase()) || s.usn.toLowerCase().includes(searchTerm.toLowerCase()))
    )
    .sort((a, b) => a.usn.localeCompare(b.usn));

  if (loading) return <div className="min-h-screen bg-[#0f172a] text-white flex items-center justify-center">Loading Portal...</div>;

  return (
    <div className="min-h-screen bg-[#0f172a] text-white p-4 md:p-8 font-sans">
      
      {/* HEADER */}
      <header className="max-w-7xl mx-auto flex flex-col xl:flex-row justify-between items-center mb-8 gap-6">
        <div>
          <h1 className="text-3xl font-bold flex items-center gap-3">
             <div className="p-2 bg-blue-600/20 rounded-lg text-blue-400"><LayoutDashboard size={28} /></div>
             Faculty Dashboard
          </h1>
          <p className="text-slate-400 text-sm mt-1 ml-1">Attendance Management System</p>
        </div>
        
        {/* CONTROL BAR */}
        <div className="flex flex-wrap justify-center items-center gap-4 bg-[#1e293b] p-3 rounded-2xl border border-slate-700 shadow-xl">
           
           {/* Date Picker */}
           <div className="relative group">
              <CalendarDays className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400 group-focus-within:text-blue-500" size={16}/>
              <input 
                type="date" 
                value={selectedDate}
                onChange={(e) => { 
                  setSelectedDate(e.target.value); 
                  fetchAttendance(selectedLab, selectedSection, selectedBatch, e.target.value); 
                }}
                className="bg-slate-900 border border-slate-700 text-white rounded-lg pl-10 pr-3 py-2 text-sm outline-none focus:border-blue-500 font-sans"
              />
           </div>

           <div className="w-px h-8 bg-slate-700 hidden md:block"></div>

           {/* Section Selector */}
           <div className="flex items-center gap-2">
              <span className="text-xs font-bold text-slate-500 uppercase">SEC</span>
              <select 
                value={selectedSection}
                onChange={(e) => { 
                  setSelectedSection(e.target.value); 
                  fetchAttendance(selectedLab, e.target.value, selectedBatch, selectedDate); 
                }}
                className="bg-slate-900 border border-slate-700 text-white rounded-lg px-3 py-2 text-sm outline-none focus:border-blue-500 font-bold"
              >
                <option value="A">A</option>
                <option value="B">B</option>
                <option value="C">C</option>
              </select>
           </div>

           {/* Batch Toggle */}
           <div className="flex bg-slate-900 rounded-lg p-1 border border-slate-700">
              <button 
                onClick={() => { setSelectedBatch("1"); fetchAttendance(selectedLab, selectedSection, "1", selectedDate); }}
                className={`px-4 py-1.5 text-xs font-bold rounded-md transition-all ${selectedBatch === "1" ? "bg-blue-600 text-white shadow-lg" : "text-slate-400 hover:text-white"}`}
              >
                Batch 1
              </button>
              <button 
                 onClick={() => { setSelectedBatch("2"); fetchAttendance(selectedLab, selectedSection, "2", selectedDate); }}
                 className={`px-4 py-1.5 text-xs font-bold rounded-md transition-all ${selectedBatch === "2" ? "bg-purple-600 text-white shadow-lg" : "text-slate-400 hover:text-white"}`}
              >
                Batch 2
              </button>
           </div>

           <div className="w-px h-8 bg-slate-700 hidden md:block"></div>

           <button onClick={handleLogout} className="bg-red-500/10 hover:bg-red-500/20 text-red-400 px-4 py-2 rounded-lg border border-red-500/20 text-sm font-bold flex items-center gap-2 transition-all">
            <LogOut size={16} /> Logout
           </button>
        </div>
      </header>

      <main className="max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-4 gap-8">
        
        {/* SIDEBAR: LABS */}
        <div className="lg:col-span-1 space-y-4">
          <div className="bg-[#1e293b] p-5 rounded-2xl border border-slate-700 shadow-xl">
            <h3 className="font-bold text-slate-200 mb-4 flex items-center gap-2">
              <Users size={18} className="text-blue-400"/> Select Experiment
            </h3>
            <div className="space-y-2 max-h-[65vh] overflow-y-auto pr-2 custom-scrollbar">
              {labs.map((lab) => (
                <button
                  key={lab.id}
                  onClick={() => { 
                    setSelectedLab(lab.id); 
                    fetchAttendance(lab.id, selectedSection, selectedBatch, selectedDate); 
                  }}
                  className={`w-full text-left px-4 py-3 rounded-xl text-sm font-medium transition-all duration-200 border ${
                    selectedLab === lab.id 
                      ? "bg-blue-600 border-blue-500 text-white shadow-lg shadow-blue-900/30" 
                      : "bg-slate-800/50 border-slate-700/50 text-slate-400 hover:bg-slate-800 hover:text-white"
                  }`}
                >
                  <span className="opacity-50 text-xs uppercase tracking-wider block mb-0.5">{lab.id}</span>
                  {lab.title}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* MAIN: STUDENT LIST */}
        <div className="lg:col-span-3">
          <div className="bg-[#1e293b] rounded-2xl border border-slate-700 shadow-xl overflow-hidden min-h-[600px] flex flex-col">
            
            {/* Toolbar */}
            <div className="p-6 border-b border-slate-700 flex flex-col sm:flex-row justify-between items-center gap-4 bg-slate-800/50">
               <div className="flex items-center gap-3">
                  <h2 className="text-xl font-bold text-white">Attendance Sheet</h2>
                  <div className="flex gap-2">
                    <span className="text-[10px] font-bold bg-blue-500/20 text-blue-400 px-2 py-1 rounded border border-blue-500/30">SEC {selectedSection}</span>
                    <span className="text-[10px] font-bold bg-purple-500/20 text-purple-400 px-2 py-1 rounded border border-purple-500/30">BATCH {selectedBatch}</span>
                  </div>
               </div>
               
               <div className="flex items-center gap-4 w-full sm:w-auto">
                  {/* Search */}
                  <div className="relative group w-full sm:w-64">
                    <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500 group-focus-within:text-blue-400 transition-colors" size={18} />
                    <input 
                      type="text" 
                      placeholder="Search student..." 
                      value={searchTerm}
                      onChange={(e) => setSearchTerm(e.target.value)}
                      className="w-full bg-slate-900 border border-slate-700 rounded-lg py-2 pl-10 text-sm text-white focus:outline-none focus:border-blue-500 transition-all placeholder:text-slate-600"
                    />
                  </div>

                  <div className="flex items-center gap-2">
                    <button onClick={clearAll} className="bg-red-500/10 hover:bg-red-500/20 text-red-400 px-3 py-2 rounded-lg transition-colors border border-red-500/20">
                      <RotateCcw size={14}/>
                    </button>
                    <button onClick={markAllPresent} className="bg-slate-700 hover:bg-slate-600 text-white text-xs font-bold px-3 py-2 rounded-lg transition-colors">
                      Mark All
                    </button>
                    <button onClick={saveAttendance} disabled={isSaving} className="bg-blue-600 hover:bg-blue-500 text-white text-xs font-bold px-4 py-2 rounded-lg transition-all shadow-lg flex items-center gap-2">
                      {isSaving ? <Loader2 size={14} className="animate-spin"/> : <Save size={14}/>}
                      {isSaving ? "Saving..." : "Save"}
                    </button>
                  </div>
               </div>
            </div>

            {/* List */}
            <div className="p-6 flex-1 bg-[#1e293b]">
                {filteredStudents.length === 0 ? (
                  <div className="h-full flex flex-col items-center justify-center text-slate-500 opacity-60">
                    <Users size={64} className="mb-4 text-slate-700" />
                    <p className="text-lg font-medium">No students found.</p>
                  </div>
                ) : (
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {filteredStudents.map((student) => {
                      const isPresent = presentIDs.includes(student.id);
                      const scores = studentScores[student.id];

                      return (
                        <div 
                          key={student.id}
                          className={`
                            flex items-center justify-between p-4 rounded-xl border transition-all duration-200 group relative overflow-hidden mb-3
                            ${isPresent 
                              ? "bg-slate-800 border-green-500/30" 
                              : "bg-slate-800/50 border-slate-700/50 hover:bg-slate-800"
                            }
                          `}
                        >
                          {/* Student Info */}
                          <div className="flex items-center gap-4 cursor-pointer" onClick={() => toggleAttendance(student.id)}>
                            <div className={`w-10 h-10 rounded-full flex items-center justify-center font-bold text-xs shadow-inner ${isPresent ? "bg-green-500 text-white" : "bg-slate-700 text-slate-400"}`}>
                              {student.section}{student.batch}
                            </div>
                            <div>
                              <p className={`font-semibold text-sm ${isPresent ? "text-green-400" : "text-slate-200 group-hover:text-white"}`}>
                                {student.name}
                              </p>
                              <p className="text-xs text-slate-500 font-mono tracking-wider">{student.usn}</p>
                            </div>
                          </div>
                          
                          {/* Right Side */}
                          <div className="flex items-center gap-4">
                            
                            {/* SCORES (Click to Reset) */}
                            <div className="flex gap-2">
                                {/* CIE Score */}
                                {scores?.cie !== undefined ? (
                                    <button 
                                        onClick={() => resetExam(student.id, "cie")}
                                        title="Click to RESET Exam"
                                        className="px-3 py-1 rounded-md text-xs font-bold border flex flex-col items-center min-w-[50px] bg-red-500/10 text-red-400 border-red-500/20 hover:bg-red-500/30 hover:border-red-500 transition-all cursor-pointer relative group/badge"
                                    >
                                        <span className="text-[9px] uppercase tracking-wider opacity-70 group-hover/badge:hidden">CIE</span>
                                        <span className="group-hover/badge:hidden">{scores.cie}</span>
                                        <Trash2 size={14} className="hidden group-hover/badge:block absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2" />
                                    </button>
                                ) : (
                                    <div className="px-3 py-1 rounded-md text-xs font-bold border flex flex-col items-center min-w-[50px] bg-slate-900 text-slate-600 border-slate-700 opacity-50">
                                        <span className="text-[9px] uppercase tracking-wider opacity-70">CIE</span>
                                        <span>-</span>
                                    </div>
                                )}

                                {/* Viva Score */}
                                {scores?.viva !== undefined ? (
                                    <button 
                                        onClick={() => resetExam(student.id, "viva")}
                                        title="Click to RESET Exam"
                                        className="px-3 py-1 rounded-md text-xs font-bold border flex flex-col items-center min-w-[50px] bg-purple-500/10 text-purple-400 border-purple-500/20 hover:bg-purple-500/30 hover:border-purple-500 transition-all cursor-pointer relative group/badge"
                                    >
                                        <span className="text-[9px] uppercase tracking-wider opacity-70 group-hover/badge:hidden">VIVA</span>
                                        <span className="group-hover/badge:hidden">{scores.viva}</span>
                                        <Trash2 size={14} className="hidden group-hover/badge:block absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2" />
                                    </button>
                                ) : (
                                    <div className="px-3 py-1 rounded-md text-xs font-bold border flex flex-col items-center min-w-[50px] bg-slate-900 text-slate-600 border-slate-700 opacity-50">
                                        <span className="text-[9px] uppercase tracking-wider opacity-70">VIVA</span>
                                        <span>-</span>
                                    </div>
                                )}
                            </div>

                            <div className="w-px h-8 bg-slate-700 mx-2"></div>

                            {/* ATTENDANCE */}
                            <button onClick={() => toggleAttendance(student.id)}>
                                {isPresent ? (
                                <CheckCircle className="text-green-500 drop-shadow-lg scale-110 transition-transform" size={24} />
                                ) : (
                                <div className="w-6 h-6 rounded-full border-2 border-slate-600 hover:border-slate-400 transition-colors" />
                                )}
                            </button>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                )}
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}