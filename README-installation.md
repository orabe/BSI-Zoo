# Getting Started for the first time

To install BSI-Zoo, you can either use Conda or Pip to create an environment.

## 1. Clone the repository:
```bash
git clone https://github.com/orabe/BSI-Zoo.git
```

## 2. Check out the uncertainty calibration branch (un-ca):
```bash
git checkout un-ca
```

## 3. Create the a new environment

1. Create the environment using conda:
```bash
conda create -n bsi_zoo python=3.8
```

2. Activate the environment:
```bash
conda activate bsi_zoo
```

## Install BSI-ZOO
1. Install the required packages:
```bash
pip install -r requirements.txt
```

2. Install the package:
```bash
pip install --use-pep517 .
```

# Getting Started for Future Changes
Every time you want to make changes to the project, follow these steps:

1. Activate your environment:
```bash
conda activate bsi_zoo
```

2. Navigate to the project directory (if you are not already there):
```bash
cd path/to/your/project
```

3. Check out the desired branch (if you are working on a different branch):
```bash
git checkout un-ca
```

4. Open the project in Visual Studio Code:
```bash
code .
```