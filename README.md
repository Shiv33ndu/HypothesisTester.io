<!-- ============================================= -->
<!--            HypothesisTester.io README          -->
<!-- ============================================= -->

# 🧪 HypothesisTester.io  

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg?logo=python)](https://www.python.org/)  
[![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)  
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)  
[![Contributions](https://img.shields.io/badge/Contributions-Welcome-brightgreen.svg?logo=github)](CONTRIBUTING.md)  
[![Open Source](https://img.shields.io/badge/Open%20Source-%E2%9D%A4-red.svg)]()  

**HypothesisTester.io** is an interactive, AI-assisted tool built with **Streamlit** that enables data scientists, analysts, and students to perform hypothesis testing seamlessly. It combines statistical rigor with intuitive UI and visual explanations.

---

## 🚀 Why This Project?

- 🧠 **No-code testing** → Upload data & test hypotheses without writing boilerplate code.  
- 👨🏻‍🦱 **User Friendly** → You don't need extensive statistical knowledge to perform various Hypothesis Tests now.
- 📊 **Wide test coverage** → From t-tests to chi-square and ANOVA.  
- ✍️ **Human-readable insights** → Planned integration with LLMs for natural language explanations.  
- 🎯 **Cross-domain utility** → Finance, healthcare, education, marketing, psychology, and more.  

---

## 📂 Project Structure

.
├── components/ # Streamlit UI widgets & reusable blocks
├── modules/ # Statistical test implementations
├── utils/ # Helper utilities (data loading, plotting, etc.)
├── streamlit_app.py # Main app entry
├── requirements.txt # Project dependencies
└── README.md # You’re reading this 🙂

---

## 🔧 Installation & Setup

> Works with **Python 3.10+**

```bash
# 1. Clone repo
git clone https://github.com/Shiv33ndu/HypothesisTester.io.git
cd HypothesisTester.io

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate    # Linux/Mac
.venv\Scripts\activate       # Windows

# 3. Install requirements
pip install -r requirements.txt

# 4. Run app
streamlit run streamlit_app.py

```

👉 App will be available at: http://localhost:8501


---

## 🧩 Usage Workflow

1. Upload your dataset (.csv / .xlsx).
2. Pick variables & choose a hypothesis test.
3. Get results → test statistics, p-values, confidence intervals.
4. Visualize rejection regions & distributions.
5. (Planned) Ask AI for plain-English interpretations & recommendations.

---

## 🔍 Supported Tests

### ✅ Parametric

- One / Two-sample t-test
- Paired t-test
- ANOVA
- One-sample Z-test
- Proportion Z-test
- Linear Regression

### ✅ Nonparametric & Categorical

- Mann-Whitney U
- Wilcoxon signed-rank
- Kruskal–Wallis H
- Chi-square (goodness of fit, independence)
- Fisher’s exact


### ✅ Correlation

- Pearson
- Spearman

---

## 🤝 Contributing

Contributions are welcome!

- Fork repo
- Create feature branch
- Commit changes with clear messages
- Submit PR 🚀
- Bug reports, feature requests, or suggestions → open an Issue.

---

## 📝 License

This project is licensed under the MIT License.

---

## 👤 Author

**Shivendu Kumar**

💼 Data Scientist & ML Engineer | MLOps Enthusiast

---