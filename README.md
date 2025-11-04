# Fake-Profiles-and-Toxic-Comments
Identifying Fake Profiles and Toxic Comments over the Social Media using ANN.

# Identifying Fake Profiles & Toxic Comments (ANN + ML)

**Overview**  
End-to-end bachelor’s project that detects fake social media profiles (ANN) and classifies toxic comments (TF-IDF + ML). Includes data preprocessing, model training scripts, demo notebooks and a short project report.

**Repo structure**
- `src/` — code for toxic comment pipeline and fake-profile ANN
- `notebooks/` — runnable demo notebooks (small sanitized datasets)
- `data/` — small demo CSVs (no PII)
- `models/` — trained model checkpoints (optional)
- `results/` — charts and confusion matrices
- `report.pdf` — final project report

**Quick start (local)**
1. `git clone <repo-url>`
2. `python -m venv venv && source venv/bin/activate`  *(or `venv\Scripts\activate` on Windows)*
3. `pip install -r requirements.txt`
4. Run demo notebook: `jupyter notebook notebooks/toxic_demo.ipynb`

**Run training (example)**
```bash
python src/fake_profile/pr.py       # trains ANN on sample_fake_profiles.csv
python src/toxic_comments/train_models.py  # trains toxic classifiers on sample_toxic.csv
