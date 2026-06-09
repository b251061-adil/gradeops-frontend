# GradeOps Deployment Guide

## ✅ What Was Fixed

### 1. **Directory Structure**
- ✅ Created `api/` folder structure
- ✅ Added `api/hitl.py` with proper ASGI handler for Vercel
- ✅ Added `api/__init__.py`

### 2. **Vercel Configuration** (`vercel.json`)
- ✅ Updated to reference correct entry point: `api/hitl.py`
- ✅ Set function timeout to 120s (API requests need time)
- ✅ Memory allocation: 1024 MB (fits Hobby plan limit of 2048 MB)
- ✅ Uses lightweight `requirements-api.txt` (excludes ML dependencies)
- ✅ Added proper build command with cache optimization

### 3. **Entry Point Handler**
- ✅ Created `api/hitl.py` which exports the FastAPI app as ASGI entry point
- ✅ Properly imports paths so parent modules are accessible

---

## ⚠️ Remaining Concerns & Recommendations

### Heavy Dependencies Issue
Your `requirements.txt` includes heavy ML packages that may cause issues on Vercel:

```
- torch>=2.1                    (~500MB)
- transformers>=4.40            (~400MB)
- nougat-ocr                    (~200MB)
```

**These exceed Vercel's typical function size limits.**

### Recommended Solutions

#### Option A: Separate Backend Services (Recommended)
Split your application into two deployments:

1. **Vercel (API Only)** - `api/hitl.py` 
   - FastAPI dashboard for TA review
   - Lightweight dependencies only
   
2. **Render/Railway/Hugging Face (Pipeline)** - `orchestrator.py`
   - Heavy ML processing
   - Run on-demand or scheduled

#### Option B: Serverless ML Container
Deploy to cloud platforms designed for ML:
- **Hugging Face Spaces** - Free for open models
- **Modal Labs** - Serverless GPU compute
- **RunPod** - GPU-backed serverless

#### Option C: Local Development → Manual Deployment
Keep `orchestrator.py` local, only deploy API to Vercel.

---

## 🚀 Deployment Steps

### 1. Install Vercel CLI
```bash
npm install -g vercel
```

### 2. Test Locally
```bash
pip install -r requirements.txt
vercel dev
# API should be available at http://localhost:3000/api/dashboard/queue
```

### 3. Deploy
```bash
vercel --prod
```

### 4. Set Environment Variables (if needed)
```bash
vercel env add OPENAI_API_KEY
vercel env add <OTHER_ENV_VAR>
```

---

## 📋 API Endpoints Available After Deployment

- `GET /api/dashboard/queue` - Get submissions pending review
- `GET /api/dashboard/submission/{id}` - Get single submission
- `POST /api/dashboard/review` - Submit TA review
- `GET /api/dashboard/stats` - Get dashboard statistics
- `GET /api/results/final` - Export finalized grades

---

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| Build timeout | Use Option A (separate deployments) |
| Import errors | Ensure `api/hitl.py` is in `api/` folder |
| 413 Payload Too Large | Reduce ML package dependencies |
| Database errors | Use Vercel KV or external DB (MongoDB, etc.) |

---

## Next Steps

1. Choose deployment strategy (Option A/B/C)
2. Create `requirements-api.txt` for lightweight deployment
3. Deploy and test endpoints
4. Monitor logs via `vercel logs`
