# synthesized-dataset
synthesized dataset by distribution matching and gradient matching  
from: https://github.com/VICO-UoE/DatasetCondensation  

### ⚠️ Caution ⚠️  
distribution-matching set and cifar100 are not sharing same labels.  

### 📁 File Tree 📁  
- generater
    - DCGAN.py
    - DCGANmodel.py
- gradient-matching (CIFAR10,  )
    - 0  
    - ...  
    - 9  
- distribution-matching (CIFAR100) - used seed0, seed1 at 5 steps  
    - 0  
    - ...  
    - 99  
- leakage
    - label{synthesized data's label(data was picked randomly)}_minlabel{original data's label that is the most similar with synthesized data among the original dataset}.png
- distribution-generated-leakage
    - {cifar100 label}minlabel{distribution label}.png

- tsne.py (https://github.com/2-Chae/PyTorch-tSNE)  
- cond-gen-samples - samples made by one class generater (`apples`, `leopard`, `bicycle`)
    - 100
        - {label}{#}.png
- dm-{`apple` | `leopard` | `bicycle`}