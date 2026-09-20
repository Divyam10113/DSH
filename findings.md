# Project Overview & Findings: RSNA Knee Abnormality Detection

**Course:** Data Science for Healthcare (Semester 5)  
**Group:** 4  
**Project Title:** AI-Assisted Detection of Twelve Clinically Important Abnormalities on Knee MRI  
**Group Leader:** Divyam Agarwal (2024A7PS1442G)  
**Team Members:**  
- Krish Taparia (2024A7PS0580G) — M1: Data & DICOM Pipeline  
- Arya Chauhan (2024A7PS1441G) — M2: Report NLP Mining  
- Divyam Agarwal (2024A7PS1442G) — M3: Image Modeling & Architecture  
- Pranav Shreekrishna Lorekar (2024A7PS0567G) — M4: Integration, Training Engine & Submission  

---

## The 12 Target Abnormalities

The model predicts independent per-study probabilities $\in [0, 1]$ for the following 12 conditions evaluated by macro-averaged AUC-ROC:

| # | Abnormality | Key Anatomical Plane / Contrast | Clinical Significance |
|---|---|---|---|
| 1 | **ACL Tear** | Sagittal (PD / T2) | Primary knee stabilizer; common sports injury requiring surgical reconstruction |
| 2 | **MCL Tear** | Coronal (PD-FS / Fluid-sensitive) | Medial stability; graded sprain or complete tear |
| 3 | **Medial Meniscus Tear** | Sagittal & Coronal | Shock absorber tear; high prevalence in degenerative and traumatic knee pain |
| 4 | **Lateral Meniscus Tear** | Sagittal & Coronal | Lateral load transmission; often accompanied by ACL rupture |
| 5 | **Medial Osteoarthritis (OA)** | Coronal & Sagittal | Joint space narrowing, cartilage degradation in medial compartment |
| 6 | **Lateral Osteoarthritis (OA)** | Coronal & Sagittal | Joint space narrowing in lateral compartment |
| 7 | **Patellofemoral (PF) OA** | Axial & Sagittal | Cartilage wear and osteophytes between patella and femoral trochlea |
| 8 | **Joint Effusion** | Sagittal & Axial (Fluid-sensitive) | Fluid accumulation indicating acute trauma or underlying inflammatory disease |
| 9 | **Synovitis** | Axial & Sagittal (Contrast / T2) | Inflammation of synovial membrane; common in inflammatory arthropathies |
| 10 | **Baker's Cyst** | Axial & Sagittal | Popliteal cyst protruding behind the knee joint |
| 11 | **Bone Contusion** | Coronal & Sagittal (STIR / PD-FS) | Subchondral bone marrow edema ("bone bruise") indicating high-energy impact |
| 12 | **Fracture** | All planes (Bone window) | Cortical break; rare in routine MRIs, highly penalized under macro-AUC |

---

## Core Problem Statement & Workflow

1. **Input**: Multi-series MRI studies (Sagittal, Coronal, Axial planes; 20–300 slices per series; variable resolution and slice thickness).
2. **Supervision**: Large unlabeled pool + small labeled subset. M2 mines pseudo-labels from multilingual radiology reports at training time.
3. **Inference**: Strictly image-only at test time (no radiology reports available).
4. **Execution**: Offline Kaggle notebook submission $\le 9$ hours without internet access.

For the detailed timeline and weekly milestones, see [gantt_chart.md](file:///Users/divyansh/Desktop/acads/sem%205/DS%20for%20Healthcare/knee_abnormality/gantt_chart.md).
