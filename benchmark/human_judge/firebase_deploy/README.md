# 🚀 Quick Start Guide - Deploy in 5 Minutes

Follow these steps to get your human evaluation UI live on Firebase using the **static HTML generator**.

## Prerequisites

- [ ] Google/Gmail account
- [ ] Python 3.x installed
- [ ] Node.js installed (for Firebase CLI)

## Overview: Two-Step Process

1. **Generate static site** from your evaluation cases (using Python)
2. **Deploy to Firebase** (using Firebase CLI)

## Step-by-Step Deployment

### 1️⃣ Create Firebase Project (2 minutes)

1. Go to https://console.firebase.google.com/
2. Click **"Add project"**
3. Name it: `scivisagent-human-eval`
4. Disable Google Analytics (optional)
5. Click **"Create project"**

### 2️⃣ Enable Firebase Services (2 minutes)

**Realtime Database:**
1. Left menu → Build → **Realtime Database**
2. Click **"Create Database"**
3. Choose location (e.g., us-central1)
4. Start in **test mode** → Enable

**Storage:**
1. Left menu → Build → **Storage**
2. Click **"Get started"**
3. Start in **test mode** → Next → Done

**Hosting:**
1. Left menu → Build → **Hosting**
2. Click **"Get started"** → Next → Next → Finish

### 3️⃣ Get Firebase Configuration (1 minute)

1. Project Settings (gear icon) → Scroll down to "Your apps"
2. Click the **web icon** (`</>`)
3. App nickname: `Human Eval UI`
4. Don't check "Firebase Hosting" box yet
5. Click **"Register app"**
6. **Copy the configuration code** (you'll need this in Step 5)

### 4️⃣ Generate Static Site (2 minutes)

**Run the static site generator** to export cases and copy images:

```bash
# From SciVisAgentBench root directory
python -m benchmark.human_judge.generate_static_site \
  --cases-yaml benchmark/eval_cases/selected_15_cases.yaml \
  --output-dir benchmark/human_judge/firebase_deploy
```

**What this does:**
- ✅ Exports all case data to `data/cases.json`
- ✅ Copies all images/videos to `images/` directory
- ✅ Prepares everything for Firebase deployment

Expected output:
```
✓ Loaded 15 cases
✓ Copied 45 images (0 failed)
✓ Generated: data/cases.json
Static Site Generation Complete!
```

**Note:** You can change the content anytime by editing the YAML file and running this command again.

### 5️⃣ Configure Firebase (1 minute)

1. Open `js/firebase-config.js` in your editor
2. Replace the placeholder values with your config from Step 3:

```javascript
const firebaseConfig = {
  apiKey: "AIza....",                          // Copy from Firebase Console
  authDomain: "your-project.firebaseapp.com",  // Copy from Firebase Console
  databaseURL: "https://your-project.firebaseio.com",
  projectId: "your-project-id",
  storageBucket: "your-project.appspot.com",
  messagingSenderId: "123456789",
  appId: "1:123456789:web:abc..."
};
```

### 6️⃣ Handle Images (Choose One)

The static site generator already **copied images to `images/` directory** (Option B below). You can use them as-is, or optionally upload to Firebase Storage for better scalability.

**Option A: Use Local Images** (Already Done! ✅)

The generator already copied all images to `images/` directory. **You can skip to Step 7 and deploy now!**

**Option B: Upload to Firebase Storage** (Optional - for larger deployments)

If you have many images or want faster loading:

```bash
# Get service account key
# Firebase Console → Project Settings → Service Accounts → Generate new private key
# Save as serviceAccountKey.json

# Install firebase-admin
pip install firebase-admin

# Upload images to Firebase Storage
python upload_images_to_storage.py \
  --cases-json data/cases.json \
  --service-account serviceAccountKey.json \
  --workspace-root ../../..

# Replace cases.json with Firebase Storage URLs version
mv data/cases_firebase.json data/cases.json
```

### 7️⃣ Deploy to Firebase (2 minutes)

```bash
# Navigate to deployment directory
cd benchmark/human_judge/firebase_deploy

# Install Firebase CLI (first time only)
npm install -g firebase-tools

# Login to Firebase (first time only)
firebase login

# Initialize Firebase in this directory (first time only)
firebase init

# When prompted, select:
#   - Hosting: Configure files for Firebase Hosting
#   - Database: Configure a security rules file for Realtime Database
#   - Storage: Configure a security rules file for Cloud Storage
#
# Then choose:
#   - Use an existing project: scivisagent-human-eval
#   - Public directory: . (current directory)
#   - Single-page app: Yes
#   - Overwrite index.html: No (keep existing)

# Deploy!
firebase deploy
```

**Alternative (if already initialized):**

```bash
cd benchmark/human_judge/firebase_deploy
firebase deploy
```

### 8️⃣ Access Your Site 🎉

Your site is now live at:
```
https://your-project-id.web.app
```

or

```
https://your-project-id.firebaseapp.com
```

## Testing Your Deployment

1. Open the URL in your browser
2. Enter your name/email
3. Click "Start Session"
4. You should see the first evaluation case
5. Try rating metrics and saving an evaluation

## Sharing with Evaluators

Share the URL with your evaluators:
```
https://your-project-id.web.app
```

They just need to:
1. Open the URL
2. Enter their name/email
3. Start evaluating!

## Viewing Results

### From Browser Console

1. Open your deployed site
2. Press `F12` to open console
3. Run:
   ```javascript
   window.debugHelpers.exportAllEvaluations()
   ```

### From Firebase Console

1. Go to Firebase Console → Realtime Database
2. Browse the `evaluations` node
3. Click the 3-dot menu → Export JSON

### Via Python Script

```python
import requests
import json

url = 'https://your-project-id.firebaseio.com/evaluations.json'
response = requests.get(url)
evaluations = response.json()

with open('evaluations.json', 'w') as f:
    json.dump(evaluations, f, indent=2)
```

## Common Issues

### "Permission denied" when saving

**Solution**: Check Firebase Database rules allow write access
1. Firebase Console → Realtime Database → Rules
2. Ensure rules match `database.rules.json`

### Images not loading

**Solution**:
- If using Firebase Storage: Check Storage rules allow read access
- If bundled locally: Check images are in `images/` directory

### Firebase not initializing

**Solution**: Verify `firebase-config.js` has correct credentials

## Updating Your Deployment

### To Change Content (Add/Remove/Modify Cases)

1. **Edit your YAML file:**

   ```bash
   # Edit benchmark/eval_cases/selected_15_cases.yaml
   ```

2. **Regenerate static site:**

   ```bash
   python -m benchmark.human_judge.generate_static_site \
     --cases-yaml benchmark/eval_cases/selected_15_cases.yaml \
     --output-dir benchmark/human_judge/firebase_deploy
   ```

3. **Redeploy:**

   ```bash
   cd benchmark/human_judge/firebase_deploy
   firebase deploy
   ```

That's it! Your changes are now live.

## Testing Locally First

Before deploying, you can test with the Flask version:

```bash
# Run local Flask server
python -m benchmark.human_judge.run_human_eval \
  --cases-yaml benchmark/eval_cases/selected_15_cases.yaml \
  --port 8081

# Open http://localhost:8081 and verify everything looks good
```

## Next Steps

- [ ] Set up proper security rules (see [DEPLOYMENT_GUIDE.md](../DEPLOYMENT_GUIDE.md))
- [ ] Enable Firebase Authentication for user tracking
- [ ] Set up custom domain (optional)
- [ ] Configure GitHub Actions for auto-deploy (optional)

## Cost

With Firebase Free Tier (Spark Plan):
- ✅ FREE for ~1000 evaluations
- ✅ FREE for ~15 cases with images
- ✅ FREE for ~10 concurrent users

You won't be charged unless you upgrade to Blaze (pay-as-you-go) plan.

## Need Help?

- Check [README.md](README.md) for detailed documentation
- Check [DEPLOYMENT_GUIDE.md](../DEPLOYMENT_GUIDE.md) for architecture details
- Firebase Documentation: https://firebase.google.com/docs

---

**Congratulations! Your human evaluation UI is now live! 🎉**
