// Firebase configuration
// TODO: Replace with your actual Firebase config from Firebase Console
const firebaseConfig = {
  apiKey: "AIzaSyBWEt_gg5vnW5Khd-0c9h1fN11XKoT7FUM",
  authDomain: "scivisagentbench-human-eval.firebaseapp.com",
  databaseURL: "https://scivisagentbench-human-eval-default-rtdb.firebaseio.com",
  projectId: "scivisagentbench-human-eval",
  storageBucket: "scivisagentbench-human-eval.firebasestorage.app",
  messagingSenderId: "490426879123",
  appId: "1:490426879123:web:ff36756e1df998e95b5166"
};

// Initialize Firebase
firebase.initializeApp(firebaseConfig);

// Get references to Firebase services
const database = firebase.database();
const storage = firebase.storage();
const auth = firebase.auth();

console.log('Firebase initialized successfully');
