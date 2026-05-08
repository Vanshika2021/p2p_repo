# Pixels to Predictions: DL Vision Challenge

Parameter-efficient fine-tuning of `SmolVLM-500M-Instruct` for science multiple-choice visual question answering.

**Final public-leaderboard score: 0.82092** (3-seed score-averaged ensemble).

---

## Authors

- **Vanshika Agrawal** (`va2652`)
- **Chiranjeev Kumar** (`ck4137`)

NYU Deep Learning, Spring 2026.

---

## Approach in one paragraph

We fine-tune the competition-required `SmolVLM-500M-Instruct` backbone using QLoRA: 4-bit NF4 base weights with bfloat16 compute, plus a rank-8 LoRA adapter targeting all attention and MLP projections (`q,k,v,o,gate,up,down`). The adapter has **4,784,128 trainable parameters** (under the 5M competition cap). We train three independent seeds (42, 7, 17) for 3 epochs each on a single H100 80GB GPU, then ensemble by averaging per-seed log-likelihood scores on the answer-letter token at the first generation position.

| Component | Setting |
|---|---|
| Base model | `HuggingFaceTB/SmolVLM-500M-Instruct` (5M-param cap honored) |
| Quantization | 4-bit NF4, double-quant, bfloat16 compute |
| LoRA rank / α / dropout | 8 / 32 / 0.05 |
| Target modules | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` |
| Trainable params | 4,784,128 (verified from `adapter_model.safetensors`) |
| Optimizer | AdamW, LR 2e-4, cosine schedule, grad-norm clip 1.0 |
| Batch / grad-accum | 2 / 4 (effective batch 8) |
| Epochs | 3 (epoch-3 selected for all seeds by val accuracy) |
| Image preprocessing | 384×384 letterbox, bicubic, white background |
| Hardware | 1× NVIDIA H100 80GB |
| Seeds (ensemble) | {42, 7, 17} |

---

## Results

### Public-leaderboard scores

| Submission | Score |
|---|---|
| Initial baseline (data only) | 0.42857 |
| Seed 42 (re-run) | 0.71629 |
| Seed 17 | 0.79476 |
| Seed 7 | 0.80080 |
| Seed 42 (used in ensemble) | 0.81287 |
| **3-seed score ensemble** | **0.82092** |

### Per-seed validation vs test

| Seed | Val acc | Test acc | Gap |
|---|---|---|---|
| 42 | 0.8025 | 0.81287 | +0.0104 |
| 7 | 0.7767 | 0.80080 | +0.0241 |
| 17 | 0.7853 | 0.79476 | +0.0095 |
| mean | 0.7882 | 0.80281 | +0.0146 |
| ensemble | — | **0.82092** | — |

Test exceeds val for every seed (no overfitting at 3 epochs). Pearson r(val, test) = 0.79.

### Inter-seed agreement on test (1,008 instances)

- All 3 seeds agree on **831 instances (82.4%)**
- 2-of-3 majority on **173 instances (17.2%)** — where ensembling adds value
- All 3 differ on **4 instances (0.4%)**
- Disagreement strongly correlates with `num_choices`: 6.9% on 2-choice, 15.7% on 3-choice, 7.2% on 4-choice, **35.1% on 5-choice**

---

## Repository layout

```
.
├── README.md                       # this file
├── notebook/
│   └── p2p_h100_paper.ipynb        # full pipeline (setup → train → ensemble)
├── adapters/                       # final LoRA adapters per seed (~19 MB each)
│   ├── seed42/
│   │   ├── adapter_config.json
│   │   └── adapter_model.safetensors
│   ├── seed7/
│   └── seed17/
├── submissions/                    # Kaggle-format CSVs
│   ├── submission_seed42.csv
│   ├── submission_seed7.csv
│   ├── submission_seed17.csv
│   └── submission_ensemble.csv     # final 0.82092 submission
├── scores/                         # raw per-choice log-prob arrays (1008 × 5)
│   ├── test_scores_seed42.npy
│   ├── test_scores_seed7.npy
│   └── test_scores_seed17.npy
├── data_logs/                      # validation accuracies per seed
│   ├── val_acc_seed42.txt
│   ├── val_acc_seed7.txt
│   └── val_acc_seed17.txt
└── report/
    ├── p2p_report_final.tex        # ACL-style 4-page report (LaTeX)
    └── p2p_report_final.pdf        # compiled PDF
```

---

## How to reproduce

### 1. Environment

```bash
pip install transformers==4.57.6 peft==0.18.1 bitsandbytes==0.48.1 accelerate==1.10.1
```

PyTorch 2.x with CUDA, bfloat16 + 4-bit NF4 support.

### 2. Data

Download `Pixels_to_Predictions.zip` from the Kaggle competition page and extract to `/content/data/` (or update the path in the notebook). The zip contains `train.csv`, `val.csv`, `test.csv`, and the corresponding image folders.

### 3. Run

Open `notebook/p2p_h100_paper.ipynb` in Colab (H100 recommended; A100 also works). Three cells:

- **Cell 1 (setup):** mounts Drive, unzips data, loads SmolVLM-500M in 4-bit, defines prompt + image helpers. Run once per session.
- **Cell 2 (train + infer):** trains one seed end-to-end with checkpointing every 50 steps. Edit `SEED = 42` and re-run for each of `{42, 7, 17}`. Saves `submission_seed{N}.csv` and `test_scores_seed{N}.npy`.
- **Cell 3 (ensemble):** loads the three score arrays, masks invalid choice positions, takes per-row mean, argmax → `submission_ensemble.csv`.

Total wall-clock on H100: ~3 hours (≈19 min/epoch × 3 epochs × 3 seeds + inference).

### 4. Inference on a saved adapter

```python
from peft import PeftModel
from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig
import torch

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                         bnb_4bit_compute_dtype=torch.bfloat16,
                         bnb_4bit_use_double_quant=True)
base = AutoModelForVision2Seq.from_pretrained(
    'HuggingFaceTB/SmolVLM-500M-Instruct',
    quantization_config=bnb, torch_dtype=torch.bfloat16, device_map='auto')
model = PeftModel.from_pretrained(base, 'adapters/seed42')
processor = AutoProcessor.from_pretrained('HuggingFaceTB/SmolVLM-500M-Instruct')
# ... build prompt, score letter tokens, see notebook Cell 2 for the full inference function
```

---

## Competition rules honored

- ✅ Backbone fixed to `SmolVLM-500M-Instruct` (verified in `adapters/seed42/adapter_config.json`)
- ✅ Trainable parameters ≤ 5M (4,784,128 actual; runtime assertion in notebook)
- ✅ No external data used at training or inference
- ✅ Submission file has only `id,answer` columns with integer answer indices

---

## License

Code in this repository is released under the MIT License for educational use.
The `SmolVLM-500M-Instruct` base model is governed by its own [HuggingFace license](https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct).
The Pixels to Predictions dataset is governed by the competition terms; we do not redistribute it here.
