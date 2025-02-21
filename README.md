# Macroeconomic Regime Analysis & Credit Risk Forecasting

This project combines unsupervised learning and time series forecasting to detect economic regimes using macroeconomic indicators and predict credit risk (e.g., credit spreads). In simple terms, it tells us when the economy is expanding or contracting and forecasts how risky lending might become.

## Overview

- **What It Does:**  
  - Uses six macroeconomic factors (CPI, Fed Funds, Industrial Production, GDP, Unemployment, Consumer Sentiment) to detect hidden economic regimes.
  - Merges these monthly datasets (from January 1996 to August 2022) with credit risk data (e.g., BAMLH0A0HYM2 credit spreads) to forecast credit risk trends.
  
- **Industry Relevance:**  
  - Helps banks and investors adjust risk management strategies.
  - Provides early warnings for increasing credit risk during economic downturns.

## Data Preparation

1. **Data Collection:**  
   - Macro data from FRED for six key indicators.
   - Credit risk data (credit spread series) from FRED.
2. **Data Merging & Normalization:**  
   - All datasets are resampled to monthly frequency (standardized to the end-of-month).
   - The merged dataset covers all six macroeconomic factors plus the credit spread.
3. **Data Cleaning & Preprocessing:**  
   - Missing values are filled using forward fill or linear interpolation.
   - Additional feature engineering includes calculating growth rates, rolling averages, and lag features.

## Modeling Approach

- **Unsupervised Learning:**  
  - Techniques like PCA, K-Means, or Hidden Markov Models (HMM) are used to detect macroeconomic regimes.
- **Time Series Forecasting:**  
  - Models such as ARIMA/SARIMA and regression methods forecast credit risk.
- **Deep Learning:**  
  - LSTM networks are employed to capture non-linear patterns and long-term dependencies.
- **Workflow:**  
  - Data is split chronologically for training and testing.
  - Forecast accuracy is evaluated using metrics like RMSE, MAE, and MAPE.

## Environment Setup with Poetry

This project uses [Poetry](https://python-poetry.org/) for dependency management and virtual environment handling. To set up the environment, run the following commands:

**Initialize Poetry (if not already initialized):**

   ```bash
   poetry init

   Here's a **README.md** file for your repository, providing a short tutorial for your collaborators:  

---

# **Macroeconomic Regime Analysis & Credit Risk Forecasting**  

Welcome to the **Macroeconomic Regime Analysis & Credit Risk Forecasting** project! 🎯  
This README will guide new collaborators on **Git setup, contribution workflow, and commit messages**.  

---

## **🚀 1) Cloning the Repository**  

To start working on the project, clone the repository using the following command:  

```sh
git clone https://github.com/virajparmaj/Macroeconomic-Regime-Analysis-Credit-Risk-Forecasting.git
```

After cloning, navigate into the project folder:  

```sh
cd Macroeconomic-Regime-Analysis-Credit-Risk-Forecasting
```

---

## **🛠️ 2) Initializing the Repository in VS Code**  

To open the project in **VS Code**, follow these steps:  

1. Open **VS Code** and press `Ctrl + Shift + P` (or `Cmd + Shift + P` on Mac).  
2. Search for **"Open Folder"** and select the cloned project folder.  
3. Open a new terminal in **VS Code** (`Ctrl + ~` shortcut).  
4. Check if Git is initialized:  

   ```sh
   git status
   ```

   - If Git **is initialized**, it will show tracked/untracked files.  
   - If Git **is NOT initialized**, run:  

     ```sh
     git init
     ```

---

## **📌 3) Understanding Git Status Symbols ('U', 'M')**  

When you run `git status`, you may see symbols like `U` and `M`. Here's what they mean:  

| Symbol | Meaning |
|--------|---------|
| `U` | **Untracked file** – The file is new and not yet added to Git. |
| `M` | **Modified file** – The file has changes that are not committed yet. |
| `A` | **Added file** – The file is staged for commit. |

### **Checking File Status in Git**  

Run:  

```sh
git status
```

This will show the modified, untracked, or staged files.

---

## **🔄 4) Step-by-Step Guide to Creating a Pull Request (PR)**  

Follow this workflow to contribute changes and create a PR:  

### **1️⃣ Check the current Git status**  

```sh
git status
```

### **2️⃣ Stage the files you want to commit**  

```sh
git add <filename>   # Add a specific file
git add .            # Add all changes
```

### **3️⃣ Create a commit message with detailed changes**  

```sh
git commit -e
```

This opens an editor to write a more **descriptive commit message**.  

Alternatively, use:  

```sh
git commit -m "feat: added data preprocessing pipeline"
```

### **4️⃣ Push changes to GitHub**  

```sh
git push origin main  # If working on main (Not recommended for PRs)
git push origin <branch-name>  # If working on a new branch
```

### **5️⃣ Open a Pull Request on GitHub**  

1. Go to the **repository on GitHub**.  
2. Click on **Pull Requests** → **New Pull Request**.  
3. Compare your branch with `main`.  
4. Add a **title & description** of your changes.  
5. Click **"Create Pull Request"** 🎉  

---

## **📝 5) Writing Good Commit Messages (feat, fix, chore, etc.)**  

Git commit messages should follow a structured **naming convention** for better clarity.  

### **Commit Message Format**

```txt
<type>: <short description>

<optional long description>
```

### **Common Commit Types**

| Type    | When to Use |
|---------|------------|
| `feat`  | Adding a new feature. |
| `fix`   | Fixing a bug. |
| `docs`  | Updating documentation. |
| `style` | Code style changes (formatting, missing semi-colons, etc.) |
| `refactor` | Code restructuring without changing functionality. |
| `test`  | Adding or modifying tests. |
| `chore` | Maintenance tasks (dependency updates, build process changes). |

### **Examples**  

✅ **Adding a feature:**  

```sh
git commit -m "feat: implemented credit risk forecasting model"
```

✅ **Fixing a bug:**  

```sh
git commit -m "fix: resolved incorrect data handling in preprocessing"
```

✅ **Updating documentation:**  

```sh
git commit -m "docs: added explanation for risk score calculation"
```

✅ **Refactoring code:**  

```sh
git commit -m "refactor: optimized feature selection process"
```

✅ **Updating dependencies:**  

```sh
git commit -m "chore: updated pandas to latest version"
```

---

### **🚀 Now You're Ready to Contribute!**

If you face any issues, feel free to reach out via GitHub Issues. Happy coding! 🎉

---

This **README.md** ensures your collaborators have a smooth start with Git and follow best practices while contributing. 🚀 Let me know if you want any refinements!
