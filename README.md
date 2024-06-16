<!-- Add banner here -->

# UniKP

## Description
- Visit [original repository](https://github.com/Luo-SynBioLab/UniKP) for detailed information
- This is simplified version of original repo

## Setup
### 1. Clone git repostiroy
```bash
cd {directory-you-want-to-clone}
git clone https://github.com/jhl8041/UniKP.git
```

### 2. Setup Anaconda environment
```bash
conda create -n Uni_test python=3.7 # can be higher than 3.7
conda activate Uni_test
```

### 3. Install dependencies
```bash
cd {project-directory}
pip install torch torchvision torchaudio
pip install -r requirements.txt
pip install scikit-learn==1.2.2 # if you have error with scikit-learn
```

### 4. Download pretrained model(1)
- Download following files from https://huggingface.co/Rostlab/prot_t5_xl_uniref50/tree/main
```markdown
pytorch_model.bin
config.json
special_tokens_map.json
spiece.model
tokenizer_config.json
```
- Place them in `prot_t5_xl_uniref50` folder

### 4. Download pretrained model(2)
- Download following files from https://huggingface.co/HanselYu/UniKP/tree/main
```markdown
UniKP for Km.pkl
UniKP for kcat.pkl
UniKP for kcat_Km.pkl
```
- Place them in `trained_dataset` folder

### 5. Fill in input data
- Fill sequence and SMILE in `inplut/input.xlsx`
- **Note**: When SMILE is longer than 218, it will be truncated


### 6. Run
```bash
python run.py
```