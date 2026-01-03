import { initializeApp, getApps, getApp } from "firebase/app";
import { getAuth } from "firebase/auth";
import { getFirestore } from "firebase/firestore";
import { getStorage } from "firebase/storage";

const firebaseConfig = {
  apiKey: "AIzaSyDaXPv8OtsJJ8YkrpsvkFM_qqMIkC1ThGg",
  authDomain: "bcsl606-lab-8ca8f.firebaseapp.com",
  projectId: "bcsl606-lab-8ca8f",
  storageBucket: "bcsl606-lab-8ca8f.firebasestorage.app",
  messagingSenderId: "785512379254",
  appId: "1:785512379254:web:af20a6ff956a69d3254057",
  measurementId: "G-9TF9C4P8Q4"
};

// Initialize Firebase (Singleton pattern to prevent re-initialization)
const app = getApps().length > 0 ? getApp() : initializeApp(firebaseConfig);

const auth = getAuth(app);
const db = getFirestore(app);
const storage = getStorage(app);

export { auth, db, storage };