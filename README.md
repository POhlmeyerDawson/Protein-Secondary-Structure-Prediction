Protein Secondary Structure Prediction with ESM2 + BiLSTM/CNN (5-Model Ensemble)

Overview
Residue-level Q9 secondary-structure prediction using Meta's ESM2 token embeddings, a 2xBiLSTM for long-range context, and a 4-layer 1D-CNN decoder for local patterns. Includes Ray Tune HPO, 5-fold ensemble, validation metrics, training curves, and Codabench-style submission generation.

Result (course leaderboard): F1 ~= 0.72 (macro), 1st/500.
Stack: PyTorch, HF Transformers, CUDA, NumPy, Pandas, scikit-learn, Ray Tune, Matplotlib, BioPython.

Features
- Per-residue classification (Q9): H, B, E, G, I, P, T, S, .
- ESM2 embeddings: facebook/esm2_t6_8M_UR50D; backbone is fine-tunable by default (not frozen).
- BiLSTM + CNN head: 2xBiLSTM (80 dims bi), 4 conv layers for local decoding.
- Hyperparameter tuning: Ray Tune + ASHA + HyperOpt.
- K-Fold ensemble: 5 models trained on stratified folds; logits averaged.
- Evaluation & reports: Accuracy, Precision, Recall, F1; classification report; training curves.
- Submission helper: Generates predictions at the specified path (see note below).
- Repro settings: seeds for PyTorch/NumPy/random; AMP + grad-clip.

Data
Place these files in the project root (or update paths in plm.py):
- sequences.fasta - FASTA with sequence IDs.
- train.tsv - columns: id = <pdb_id>_<...>_<pos>, secondary_structure in Q9.
- test.tsv - columns: id; labels are filled by the script.

Heads-up: plm.py currently points to
FASTA_PATH = "/home/pie_crusher/CS121/Project1/sequences.fasta"
LABEL_PATH = "/home/pie_crusher/CS121/Project1/train.tsv"
Change these or symlink if your paths differ.

Installation
# Python 3.10+ recommended
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.* datasets ray[tune] hyperopt scikit-learn biopython matplotlib pandas numpy iterstrat tensorboard
GPU strongly recommended (the tiny ESM works on CPU but it will crawl).

Quick Start
1) Hyperparameter Tuning (Ray Tune)
Runs multiple trials then saves the best checkpoint + config.
Command:
  python plm.py
Outputs:
  - checkpoints/best_model.pth
  - checkpoints/best_model_config.json
  - TensorBoard logs under runs/

2) Fast Debug Run (no tuning)
Uses a fixed config to sanity-check the pipeline.
Command:
  python plm.py -debug

3) Train the 5-Fold Ensemble
Both paths above end by training 5 folds with the best config.
Saves weights: checkpoints/final_model_fold_0..4.pth
Writes summary: checkpoints/best_result.txt

4) Evaluate the Ensemble
Evaluation on each fold's validation split happens inside plm.py.
To re-run just the eval, call (single line; wrap as needed):
  python -c "from plm import *; best_config={'learning_rate':2e-4,'batch_size':1,'dropout_rate':0.25,'max_length':1024,'weight_decay':0.01,'patience':5,'num_epochs':5,'cnn_sizes':[64,48,32,16],'num_labels':9,'num_folds':5}; tok=load_tokenizer(); for f in range(5): _, vl, _ = build_data_loaders(best_config['batch_size'], FASTA_PATH, LABEL_PATH, tok, ResidueLevelDataset, fold=f, num_folds=5, config=best_config); evaluate_ensemble_on_val_loader(vl)"

5) Generate Submission File
Creates predictions for the test set.
Command:
  python -c "from plm import generate_submission_csv; generate_submission_csv('sequences.fasta','test.tsv','prediction.csv')"

Important format note:
- The current function writes a CSV to the given path and then overwrites the same path with a TSV (id, secondary_structure). The final file at 'prediction.csv' is TSV-formatted. If you need distinct CSV and TSV outputs, adjust the function or rename the output path accordingly. (Quirk required for leaderboard submission)

Hyperparameter Tuning
Edit the search space in plm.py:
  search_space = {
      "learning_rate": tune.loguniform(1e-4, 5e-3),
      "batch_size": tune.choice([1, 2]),
      "dropout_rate": tune.uniform(0.1, 0.3),
      "max_length": tune.choice([512, 768, 1024]),
      "weight_decay": tune.choice([0.0, 0.005, 0.01]),
      "patience": tune.choice([2, 3, 5])
  }
Scheduler: ASHA; Search: HyperOpt; default trials: 15.

Model
- Backbone: ESM2 t6_8M (no pooling, token embeddings).
- Context encoder: 2-layer BiLSTM (hidden 40 per dir -> 80).
- Decoder: 1D-CNN stack ([7,7,5,5] kernels, Tanh), final Conv1d -> num_labels.
- Loss: Cross-Entropy with ignore_index=-100 for special/pad tokens.
- Training: AMP (torch.amp), AdamW, grad clip=1.0, early stopping.

Outputs
- Checkpoints: checkpoints/
- TensorBoard: runs/
- Plots: training_history.png
- Predictions: prediction.csv (TSV content; see note above)
- Metrics summary: checkpoints/best_result.txt

Graphs
- Training history: Loss/Acc/F1 over epochs saved to training_history.png.
  (Auto-created if the loaded checkpoint contains history.)

Repo Structure (typical)
Project1/
├─ plm.py
├─ train.tsv
├─ test.tsv
├─ sequences.fasta
├─ checkpoints/
├─ runs/
└─ prediction.csv

Repro Notes
- Seeds set for torch, numpy, and random.
- ESM downloads on first run (cache in ~/.cache/huggingface).
- If you change label order, keep it consistent across train/eval/submission.

Troubleshooting
- Out of memory: reduce max_length, batch size to 1, or use CPU (slow).
- HuggingFace SSL/proxy errors: set HF_HOME or use TRANSFORMERS_OFFLINE=1 after caching.
- Different file paths: fix FASTA_PATH/LABEL_PATH at the top of plm.py.

Contributors
- Pierce Ohlmeyer-Dawson
