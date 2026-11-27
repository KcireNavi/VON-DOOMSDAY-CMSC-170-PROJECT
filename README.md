# VON DOOMSDAY ON DEC 18 — CMSC 170

Predicting fatal landslides using Naive Bayes, KNN, and Decision Trees from scratch.

- **Project Website:** https://kcirenavi.github.io/VON-DOOMSDAY-CMSC-170-PROJECT/
- **Slides:** https://www.canva.com/design/DAHcBMZ-92k/Xcf5yIMWe-n0G7a21oNVYQ/edit

---

## How to Run & Reproduce Results

### 1. Install dependencies

```bash
pip install numpy matplotlib
```

### 2. Run the comparison script

```bash
python Codes/compare.py
```

This runs the 3-fold cross-validation on `datasets/landslides.csv`, prints the performance scores and confusion matrices for all models, and saves the comparison plot to `method_comparison.png`.
