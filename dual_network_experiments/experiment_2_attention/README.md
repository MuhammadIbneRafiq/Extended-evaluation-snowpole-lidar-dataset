# Experiment 2: Attention Module Ablation

This experiment evaluates the contribution of attention mechanisms by comparing models with and without attention modules.

## Experiment Design

### Configurations Tested
1. **Baseline**: Full model with all attention modules enabled
2. **No Attention**: Model with attention modules disabled
3. **EMA Only**: Only Efficient Multi-scale Attention
4. **CBAM Only**: Only Convolutional Block Attention Module
5. **Cross-Modal Only**: Only cross-modal attention between branches

## Key Findings (Expected)

Based on the architecture design, attention modules are expected to:
- Improve mAP@50 by 2-3%
- Increase inference time by ~10%
- Enhance feature discrimination in challenging conditions
- Better integrate multi-modal information

## Files

- `train_with_attention.py`: Training with full attention
- `train_without_attention.py`: Training without attention
- `compare_results.py`: Performance comparison script
- `config_attention.yaml`: Attention-specific configuration
- `ablation_study.py`: Systematic ablation study

## Running the Experiment

### Train with Attention
```bash
python train_with_attention.py --config config_attention.yaml --attention_type ema
```

### Train without Attention
```bash
python train_without_attention.py --config config_attention.yaml
```

### Compare Results
```bash
python compare_results.py --exp1 with_attention --exp2 without_attention
```

## Metrics to Compare

- Detection Performance: mAP@50, mAP@50-95, Precision, Recall
- Computational Cost: FLOPs, Parameters, Inference time
- Feature Quality: Attention map visualization, Feature similarity

## Expected Results Table

| Configuration | mAP@50 | mAP@50-95 | GPU (ms) | Parameters |
|--------------|--------|-----------|----------|------------|
| Full Attention | 0.92 | 0.46 | 5.0 | 2.5M |
| No Attention | 0.89 | 0.44 | 4.5 | 2.3M |
| EMA Only | 0.91 | 0.45 | 4.7 | 2.4M |
| CBAM Only | 0.90 | 0.45 | 4.8 | 2.4M |
| Cross-Modal Only | 0.91 | 0.45 | 4.6 | 2.4M |
