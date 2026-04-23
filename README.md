# CAFE (Catastrophic Data Leakage in Vertical Federated Learning) Verification & FedAvg Baseline

This repository implements the baseline algorithm for Horizontal Federated Learning with non-IID label distribution (Dirichlet) to serve as a comparative standard against vertical FL architectures like CAFE.

## Commands to run the application (Windows)

### 1. Creating & activating virtual environment
```bash
python3 -m venv venv
venv\Scripts\activate
```
### 2. Install dependencies
```bash
pip install -r requirements.txt
```
### 3. Run the application
```bash
python run_experiments.py
```
### 4. Plot the results
```bash
python src/plot_results.py
```

## Troubleshooting (If you encounter ImportError: Unable to import module `ray`)
### 1. Install Python 3.11 (If not installed)
```bash
Winget install Python.Python.3.11   
```
### 2. Create & Activate Python 3.11 virtual environment
```bash
py -3.11 -m venv venv311
venv311\Scripts\activate
```
### 3. Reinstall dependencies using Python 3.11
```bash
py -3.11 -m pip install -r requirements.txt
```
### 4. Verify Python 3.11 is installed
```bash
python -c "import ray; print(ray.__version__)"
```