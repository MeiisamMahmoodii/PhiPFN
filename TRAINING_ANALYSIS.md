# PhiPFN Training Analysis & Recommendations

## Current Training Setup

### ✅ What's Working
1. **Architecture**: Hybrid TabPFN + Phi-4 with bridge layer (192→3072)
2. **Multi-task Loss**: LM + ATE regression + Trust calibration
3. **Efficient Training**: 4-bit quantization, LoRA adapters, gradient checkpointing
4. **Data**: 1,500 synthetic causal reasoning examples

### ⚠️ Limitations & Areas for Improvement

## 1. **Dataset Size (CRITICAL)**
- **Current**: 1,500 examples (500 × 3 shards)
- **Issue**: Very small for fine-tuning LLMs
- **Recommendation**: 
  - Minimum: 10,000-50,000 examples
  - Ideal: 100,000+ examples
  - **Action**: Generate more diverse causal scenarios

## 2. **Data Diversity**
- **Current**: Only 3 causal types (confounder, mediator, collider)
- **Missing**:
  - More complex causal graphs (4-5+ variables)
  - Time-series causality
  - Selection bias scenarios
  - Instrumental variables
  - Front-door/back-door adjustment scenarios
- **Recommendation**: Expand to 10+ causal patterns

## 3. **Training Configuration**

### Current Settings:
```python
Batch Size: 4 × 4 = 16 effective
Learning Rate: 5e-5
Epochs: 3
Total Steps: ~93 steps/epoch × 3 = ~280 steps
```

### Recommendations:
```python
# For better convergence
per_device_train_batch_size=8  # if GPU memory allows
gradient_accumulation_steps=8   # effective batch = 64
learning_rate=2e-5              # lower for stability
num_train_epochs=5-10           # more epochs with larger data
warmup_ratio=0.1                # add warmup
weight_decay=0.01               # add regularization
```

## 4. **Validation & Early Stopping**
- **Missing**: No validation set
- **Recommendation**:
  - Split data 80/10/10 (train/val/test)
  - Add `eval_strategy="steps"` with `eval_steps=50`
  - Implement early stopping on validation loss
  - Track validation metrics during training

## 5. **Loss Balancing**
- **Current**: Fixed weights (1.0 LM + 1.0 REG + 0.5 CAL)
- **Issue**: May not be optimal
- **Recommendation**:
  - Use uncertainty weighting or GradNorm
  - Or tune weights based on validation performance
  - Monitor loss ratios and adjust dynamically

## 6. **Model Architecture Considerations**

### Current: LoRA Fine-tuning
```python
lora_r=64
lora_alpha=16
```

### Consider:
- **Increase LoRA rank**: r=128 or r=256 for more capacity
- **Full fine-tuning**: If resources allow (remove LoRA, no quantization)
- **Bridge layer complexity**: Add non-linearity or deeper projection

## 7. **Evaluation & Testing**

### Add:
1. **Comprehensive metrics**:
   - Per-type accuracy (confounder/mediator/collider)
   - Calibration curves for trust scores
   - ATE error distribution analysis

2. **Adversarial testing**:
   - Test on out-of-distribution causal graphs
   - Stress test with contradictory claims
   - Test on real-world datasets (if available)

3. **Ablation studies**:
   - TabPFN embeddings vs. random embeddings
   - Bridge layer vs. direct concatenation
   - Multi-task vs. single-task training

## 8. **Data Quality Improvements**

### Current Issues:
- Synthetic data only
- Simple patterns repeated
- Limited claim variations

### Recommendations:
1. **Claim diversity**: 
   - 10-20 different claim templates per causal type
   - Include ambiguous/nuanced claims
   - Add distractors (irrelevant information)

2. **TabPFN embedding quality**:
   - Verify embeddings are meaningful
   - Check if TabPFN actually learned causal patterns
   - Consider pre-training TabPFN on more causal data

3. **Add noise/realism**:
   - Sample sizes (small vs. large)
   - Measurement error
   - Confounding strength variations

## 9. **Training Stability**

### Monitor:
- Gradient norms (currently logged ✓)
- Loss spikes
- Learning rate schedule effectiveness
- Weight updates magnitude

### Add:
```python
# In TrainingArguments
max_grad_norm=1.0  # gradient clipping
lr_scheduler_type="cosine"  # better than linear
warmup_ratio=0.1
```

## 10. **Reproducibility & Experiment Tracking**

### Missing:
- Random seeds not set
- No experiment tracking (wandb/tensorboard)
- Hyperparameter versions not logged

### Add:
```python
import random
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Add to TrainingArguments
report_to="wandb"  # or "tensorboard"
logging_dir="logs"
```

## Priority Action Items

### 🔴 High Priority (Must Do)
1. **Generate 10-50x more training data** (15K-75K examples)
2. **Add validation set and early stopping**
3. **Test current model** (run `python scripts/test_model.py`)

### 🟡 Medium Priority (Should Do)
4. **Increase LoRA rank** (r=128)
5. **Add warmup and better LR schedule**
6. **Tune loss weights** based on validation
7. **Expand to more causal types**

### 🟢 Low Priority (Nice to Have)
8. **Add experiment tracking** (wandb)
9. **Implement ablation studies**
10. **Test on real-world benchmarks**

## Expected Performance

### With Current Setup (1.5K examples):
- ATE MAE: ~0.3-0.5 (baseline ~0.5-1.0)
- Trust Accuracy: ~60-70%
- **Status**: Proof of concept, likely underfitting

### With Improved Setup (50K+ examples):
- ATE MAE: <0.2
- Trust Accuracy: >85%
- **Status**: Production-ready for research

## Quick Wins

### 1. Test Current Model (5 minutes)
```bash
python scripts/test_model.py
```

### 2. Generate More Data (1-2 hours)
```bash
# Modify scripts/data_factory.py to generate 10x more
python scripts/data_factory.py --n_shards 30 --examples_per_shard 500
```

### 3. Better Training Config (5 minutes)
```python
# In training/train.py
per_device_train_batch_size=8
num_train_epochs=5
warmup_ratio=0.1
eval_strategy="steps"
eval_steps=100
load_best_model_at_end=True
```

## Conclusion

**Current Status**: 🟡 Functional but needs more data

**Next Steps**:
1. Test current model to establish baseline
2. Generate 10-50x more diverse training data
3. Add validation and improve training config
4. Re-train and compare results

The architecture and approach are sound, but the model is severely data-limited. With more data and proper validation, this should work well for causal reasoning tasks.
