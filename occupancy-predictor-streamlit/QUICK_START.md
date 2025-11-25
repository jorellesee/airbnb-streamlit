# Quick Start - Upload to GitHub & Deploy

Your production-ready folder is complete! Here's what to do now:

## Location
```
/Users/jorellesee/Desktop/Code/Big Data Project/Final/occupancy-predictor-streamlit/
```

## 3 Steps to Deploy

### Step 1: Create GitHub Repository
```bash
# Go to github.com and create new repo named "occupancy-predictor"
# Clone it locally
git clone https://github.com/YOUR_USERNAME/occupancy-predictor.git
cd occupancy-predictor
```

### Step 2: Copy Files
Copy everything from `occupancy-predictor-streamlit/` to your cloned repo:
```bash
# Copy all files (keeping folder structure)
# If using command line:
cp -r "/Users/jorellesee/Desktop/Code/Big Data Project/Final/occupancy-predictor-streamlit"/* .

# Or manually in Finder:
# - Copy: occupancy-predictor-streamlit folder contents
# - Paste into: your cloned GitHub repo folder
```

Verify you have:
```
app.py
requirements.txt
packages.txt          ← MUST be at root level!
.gitignore
README.md
DEPLOYMENT_GUIDE.md
.streamlit/config.toml
models/
  ├── occupancy_model_final.rds
  ├── occupancy_model_info.rds
  └── occupancy_optimizer_functions.rds
```

### Step 3: Deploy
```bash
# In your repo directory
git add .
git commit -m "Add occupancy predictor Streamlit app"
git push -u origin main

# Then go to https://share.streamlit.io
# - Click "New app"
# - Select your repository, branch main, file app.py
# - Click Deploy!
```

## That's It! 🚀

Your app will be live in 2-3 minutes at:
```
https://share.streamlit.io/YOUR_USERNAME/occupancy-predictor/main/app.py
```

## Testing Locally (Optional)

Before deploying, test locally:
```bash
cd "/Users/jorellesee/Desktop/Code/Big Data Project/Final/occupancy-predictor-streamlit"
streamlit run app.py
```

Browse to `http://localhost:8501` and test the features.

## Troubleshooting

**"R is not installed" error**
- This is OK - means R will be installed when the app loads
- Once R finishes installing, refresh the page

**"packages.txt not found"**
- Make sure it's at repository **root**, not in a subfolder
- The file should have 2 lines: `r-base` and `r-base-dev`

**Models not loading**
- Verify `models/` folder is in the repo
- All 3 .rds files should be present

## File Summary

| File | Size | What It Does |
|------|------|-------------|
| app.py | 30 KB | Main app - Predictor, Price Optimization, About |
| requirements.txt | 74 B | Python packages (streamlit, pandas, numpy, rpy2) |
| packages.txt | 18 B | R packages (r-base, r-base-dev) |
| models/ | 5.7 MB | ML models for predictions |
| README.md | 5.7 KB | User documentation |
| DEPLOYMENT_GUIDE.md | Detailed deployment instructions |

## Support

- Read `README.md` for detailed usage guide
- Read `DEPLOYMENT_GUIDE.md` for advanced deployment options
- Check Streamlit Cloud logs if deployment fails

---

**You're all set!** Just push to GitHub and deploy. The app works perfectly on Streamlit Community Cloud! 🎉
