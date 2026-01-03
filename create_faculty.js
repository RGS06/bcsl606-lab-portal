const admin = require("firebase-admin");
const serviceAccount = require("./serviceAccountKey.json");

admin.initializeApp({
  credential: admin.credential.cert(serviceAccount)
});

const db = admin.firestore();
const auth = admin.auth();

async function fixFaculty() {
  const email = "faculty@test.com";
  
  try {
    let uid;
    
    // 1. Check if user exists in Auth
    try {
      const userRecord = await auth.getUserByEmail(email);
      uid = userRecord.uid;
      console.log(`✅ Found existing user: ${email}`);
    } catch (error) {
      if (error.code === 'auth/user-not-found') {
        // Create if missing
        const newUser = await auth.createUser({
          email: email,
          password: "password123",
          displayName: "Professor Admin",
        });
        uid = newUser.uid;
        console.log(`✅ Created new user: ${email}`);
      } else {
        throw error;
      }
    }

    // 2. FORCE update the Firestore Database
    await db.collection("users").doc(uid).set({
      uid: uid,
      name: "Professor Admin",
      email: email,
      role: "faculty", // 👈 Crucial Step
      updatedAt: new Date().toISOString()
    }, { merge: true });

    console.log(`🎉 SUCCESS! Faculty role assigned to ${email}`);
    console.log(`👉 Login with Password: password123`);
    
  } catch (error) {
    console.error("❌ Error:", error.message);
  }
}

fixFaculty();