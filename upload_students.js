const admin = require("firebase-admin");
const fs = require("fs");
const csv = require("csv-parser");
const serviceAccount = require("./serviceAccountKey.json");

// Initialize Firebase
if (!admin.apps.length) {
  admin.initializeApp({
    credential: admin.credential.cert(serviceAccount)
  });
}

const db = admin.firestore();
const auth = admin.auth();

const students = [];

// 👇 UPDATED FILE NAME HERE
const csvFileName = "students.csv"; 

if (!fs.existsSync(csvFileName)) {
    console.error(`❌ Error: File '${csvFileName}' not found. Make sure you renamed your file to 'students.csv'`);
    process.exit(1);
}

// Read and Upload
fs.createReadStream(csvFileName)
  .pipe(csv())
  .on("data", (row) => {
    // Basic validation to ensure row has data
    if (!row["USN"]) return; 
    
    students.push({
      usn: row["USN"].trim(),
      name: row["STUDENT NAME"].trim(),
      section: row["SECTION"].trim(),
      batch: row["Batch"].toString().trim(),
      email: `${row["USN"].trim()}@test.com`, 
      password: row["USN"].trim() 
    });
  })
  .on("end", async () => {
    console.log(`✅ CSV Read. Found ${students.length} students. Starting upload...`);
    await uploadStudents();
  });

async function uploadStudents() {
  let successCount = 0;
  let errorCount = 0;

  for (const student of students) {
    try {
      let uid;
      
      // 1. Create Login
      try {
        const userRecord = await auth.getUserByEmail(student.email);
        uid = userRecord.uid;
      } catch (e) {
        if (e.code === 'auth/user-not-found') {
          const newUser = await auth.createUser({
            email: student.email,
            password: student.password,
            displayName: student.name,
          });
          uid = newUser.uid;
        } else {
          throw e;
        }
      }

      // 2. Create Database Entry (Critical for Student Dashboard)
      await db.collection("users").doc(uid).set({
        uid: uid,
        usn: student.usn,
        name: student.name,
        email: student.email,
        role: "student", // 👈 This redirects them to /dashboard
        section: student.section,
        batch: student.batch,
        createdAt: new Date().toISOString()
      }, { merge: true });

      process.stdout.write("."); // Progress indicator
      successCount++;
    } catch (error) {
      console.error(`\n❌ Error uploading ${student.usn}:`, error.message);
      errorCount++;
    }
  }

  console.log(`\n\n🎉 Job Complete!`);
  console.log(`✅ Successfully Uploaded: ${successCount}`);
  process.exit();
}