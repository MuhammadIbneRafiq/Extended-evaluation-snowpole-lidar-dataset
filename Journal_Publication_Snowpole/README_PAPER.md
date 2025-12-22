# 📄 Dual-Branch RGBA Snow Pole Detection Paper

## Conference-Ready LaTeX Paper for Submission

### 📁 Paper Files

1. **Main Paper**: `paper_dual_branch_final_complete_v2.tex`
   - Complete conference paper in LNCS format
   - ~15 pages with all sections filled
   - Ready for submission to CVPR/ICCV/ECCV workshops

2. **Bibliography**: `references_extended.bib`
   - 50+ references including all cited papers
   - Properly formatted BibTeX entries
   - Includes Yang et al. (2025) dual-branch paper

3. **Compilation Script**: `compile_paper.bat`
   - Windows batch script to compile the paper
   - Runs pdflatex and bibtex automatically

### 📊 Paper Structure

#### Title
**"Dual-Branch RGBA Architecture for Enhanced Snow Pole Detection: Fusing Geometric and Reflectance Features in LiDAR-Based Perception"**

#### Authors
- Muhammad Ibne Rafiq (TU Eindhoven) - Lead Author
- Durga Prasad Bavirisetti (NTNU/Gävle)
- Shaira Tabassum (NTNU)
- Gabriel Hanssen Kiss (NTNU)
- Frank Lindseth (NTNU)

#### Abstract Highlights
- Novel 4-channel RGBA fusion architecture
- 8.3% mAP@50 improvement over single modality
- Log-normalized range at 80m (Ouster OS2-128 specs)
- Merkle tree caching: 34% latency reduction
- Real-time performance: 85 FPS on RTX 3090

#### Sections
1. **Introduction**
   - Nordic winter challenges for autonomous driving
   - LiDAR multi-modal advantages
   - Dual-branch motivation from Yang et al. (2025)
   - 5 key contributions

2. **Related Work**
   - LiDAR-based object detection evolution
   - Multi-modal fusion approaches
   - Adverse weather perception challenges
   - Range normalization techniques

3. **Methodology**
   - Dual-branch RGBA architecture design
   - Range normalization formula and rationale
   - YOLOv9t modification for 4-channel input
   - Two-stage transfer learning strategy
   - Merkle tree caching algorithm

4. **Experimental Setup**
   - SnowPole dataset: 1,954 images
   - Evaluation metrics (COCO standard)
   - Implementation details

5. **Results and Discussion**
   - Baseline comparison table
   - Ablation studies (3 detailed tables)
   - Qualitative analysis with figures
   - Failure case analysis

6. **Conclusion**
   - Summary of contributions
   - Future work directions

### 🖼️ Figures Used in Paper

The paper references these figures from the directory:
- `comprehensive_detection_comparison.png` - Detection results comparison
- `val_batch2_pred.jpg` - Validation predictions
- `all_permutations_overview (1).png` - Modality combinations
- `Combination4_original.png` - RGB combination example
- `range_original.png` - Range channel visualization

### 📈 Key Results

#### Main Performance Table
| Model | Input | mAP@50 | FPS |
|-------|-------|--------|-----|
| YOLOv9t | Near-IR | 0.792 | 89 |
| YOLOv9t | Signal | 0.801 | 89 |
| YOLOv9t | RGB (Comb4) | 0.814 | 89 |
| **YOLOv9t** | **RGBA (Ours)** | **0.861** | **85** |

#### Range Normalization Impact
- Linear @ 80m: 0.772 mAP@50
- **Log @ 80m: 0.861 mAP@50** ✅
- Improvement: +8.9% absolute

#### Caching Performance
- Tile size: 128×32 pixels
- Cache hit rate: 61.7%
- Speedup: 1.51× (34% reduction)
- mAP drop: 0.003 (negligible)

### 🎯 Paper Contributions

1. **Novel RGBA Representation**
   - First to combine continuous range with reflectance in 4-channel format
   - Compatible with standard CNNs

2. **Comprehensive Evaluation**
   - 6 YOLO variants tested
   - 10 input modalities compared
   - 1,954 annotated frames

3. **Practical Deployment**
   - Real-time performance (85 FPS)
   - Edge device compatible
   - Merkle tree caching innovation

4. **Open Science**
   - Public code release
   - Extended dataset with .npy range
   - Reproducible experiments

### 📝 How to Compile

#### Windows:
```bash
compile_paper.bat
```

#### Linux/Mac:
```bash
pdflatex paper_dual_branch_final_complete_v2.tex
bibtex paper_dual_branch_final_complete_v2
pdflatex paper_dual_branch_final_complete_v2.tex
pdflatex paper_dual_branch_final_complete_v2.tex
```

### 🔗 Citations

Key papers cited:
```bibtex
@article{yang2025towards,
  title={Towards Generalized Range-View LiDAR Segmentation in Adverse Weather},
  author={Yang, Haoran and others},
  journal={arXiv:2506.08979},
  year={2025}
}

@article{wang2024yolov9,
  title={YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information},
  author={Wang, Chien-Yao and others},
  journal={arXiv:2402.13616},
  year={2024}
}

@misc{bavirisetti2025snowpole,
  title={SnowPole Detection Dataset},
  author={Bavirisetti, D.P. and others},
  publisher={Mendeley Data},
  doi={10.17632/tt6rbx7s3h.3},
  year={2025}
}
```

### 📊 Supplementary Materials

Available at: https://github.com/MuhammadIbneRafiq/Extended-evaluation-snowpole-lidar-dataset
- Training notebooks (Colab & Desktop versions)
- Pre-processed .npy range files
- Trained model weights
- Inference scripts

### 🚀 Submission Targets

Suitable for submission to:
- **CVPR 2025** Workshop on Autonomous Driving
- **ICCV 2025** Workshop on Vision for All Seasons
- **ECCV 2025** Workshop on Robust Perception
- **ICRA 2026** LiDAR Perception Track
- **IEEE T-ITS** (with extensions)

### ✅ Paper Checklist

- [x] Abstract under 250 words
- [x] 5 key contributions listed
- [x] Related work covers 30+ papers
- [x] Methodology with algorithms
- [x] Comprehensive results tables
- [x] Ablation studies
- [x] Statistical significance reported
- [x] Reproducibility information
- [x] Code/data availability statement
- [x] Proper citations and references

---

**Status**: Ready for submission after final proofreading
**Last Updated**: December 2024
**Contact**: m.ibne.rafiq@student.tue.nl
