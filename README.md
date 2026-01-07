# Handwritten-Digits-Classifier

From-scratch implementation of a multiclass Logistic Regression classifier on the digits dataset, featuring a custom Gradient Descent algorithm and performance benchmarking against scikit-learn.

# Implémenter from scratch une régression logistique

## Installation

We recommend using `uv` for dependency management. If you prefer `pip`, see the alternative below.

### Using uv (Recommended)

#### Manual Installation
Install uv

**Unix/macOS:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows:**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

#### Sync dependencies
```bash
uv sync
```

#### Run the project
```bash
uv run src/main.py
```

### Using pip (Alternative)

If you don't want to use uv, you can install dependencies with pip:

```bash
pip install -r requirements.txt
```

Then run the project directly:

```bash
python src/main.py
```