# Deploy drift to Vercel — Exact steps

## 1. Go to Vercel

Open [vercel.com](https://vercel.com) and log in (GitHub, GitLab, or email).

## 2. Import project

- Click **"Add New..."** → **"Project"**
- Click **"Import Git Repository"**
- Select your GitHub account and find **`intent2model`** (or your repo name)
- Click **"Import"**

## 3. Configure project

On the "Configure Project" screen:

- **Framework Preset:** Next.js (auto-detected)
- **Root Directory:** Click **"Edit"** → select **`frontend`** (since the Next.js app is in `frontend/`)
- **Build Command:** `npm run build` (default, leave as is)
- **Output Directory:** `.next` (default, leave as is)
- **Install Command:** `npm install` (default, leave as is)

## 4. Environment variables (optional)

You don't need any env vars for the frontend to deploy. The frontend is static; users run the engine locally.

## 5. Deploy

- Click **"Deploy"**
- Wait ~2-3 minutes for the build to complete
- You'll get a URL like `https://intent2model-xyz.vercel.app` or your custom domain

## 6. Done

- Open the URL → you'll see the "drift web coming soon" landing with the Prism background
- Users can click **"Try locally"** or **"Get drift CLI"** to get started
- The frontend is hosted; the engine runs on each user's machine

## 7. Custom domain (optional)

- In Vercel project settings → **Domains** → add your domain (e.g. `drift.ml`)
- Follow Vercel's DNS instructions

---

That's it. The web app is live. Users open it from anywhere; they run the engine locally with an LLM (Gemini CLI, Ollama, etc.).
