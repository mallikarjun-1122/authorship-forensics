# 🔍 Authorship Forensics & AI Detector

Quantitative stylometric fingerprinting, synthetic text detection, and intra-document paragraph drift analysis built with Python, scikit-learn, and Streamlit.

## Features
- **Authorship Verification**: Dual-stage stylometric evaluation combining TF-IDF N-gram cosine similarity (preserving function words) and 9 bounded linguistic feature vectors.
- **Synthetic AI Detection**: Scans for LLM transition clichés and analyzes sentence burstiness (length variance).
- **Intra-Document Drift**: Leave-one-out paragraph consistency testing to identify ghostwritten or spliced sections.
- **Explainable AI (XAI)**: Visualizes top matching authorial phrases and absent author traits.
- **Forensic PDF Export**: Generates official downloadable PDF summary reports.

## Local Setup
```bash
# Clone the repository
git clone <your-repo-url>
cd authorship-forensics

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run app.py
```
