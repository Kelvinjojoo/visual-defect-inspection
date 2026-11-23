# PatchCore vs YOLO: Does a Factory Need Labeled Defects?

**Live demo:** [visual-defect-inspection-gumcheck.streamlit.app](https://visual-defect-inspection-gumcheck.streamlit.app) (both models judge the same gum photo, and you can upload your own)

A factory camera takes a photo of every product, here pieces of chewing gum. A model looks at each photo and decides: **PASS** or **REJECT**. This is called visual defect inspection.

There are two ways to teach such a model:

| | PatchCore | YOLO |
| --- | --- | --- |
| Type of learning | unsupervised | supervised |
| Learns from | photos of good products only | photos where people marked each defect |
| Output | a score per photo + a heatmap of unusual areas | a box around each defect, with the defect type |
| Needs labeled defect photos? | no | yes |

Which way is better? That depends on how many labeled defect photos there are. YOLO gets better with every labeled photo, while PatchCore needs none at all. Defects are rare, and marking them by hand takes time. So this project compares the two ways at several amounts of labels:

> **Main question:** on the same test photos, which model catches more defects: PatchCore, trained on good photos only, or YOLO, trained with 5, 10, 20 or 40 labeled defect photos? And if YOLO catches up, how many labeled photos does it take?

The answer tells a factory what to do. If PatchCore is as good, no defect has to be labeled. If YOLO is clearly better, the number of labels shows how much labeling work that costs.

We also compare which defect types each model misses, whether the heatmap or the boxes point at the real defect, and the time per photo. The demo shows both models side by side on the same photo.

Both models are scored on the **same test photos** with the **same rule**, so the comparison is fair.

Every number about the data in this README is printed by code in the notebook named next to it.

## Dataset

[VisA](https://github.com/amazon-science/spot-diff) (Visual Anomaly) by Amazon. License: CC BY 4.0. VisA has 12 objects and 10,821 photos. This project uses one object, **chewing gum** (notebook 1.1).

### What is inside the raw data

```
data/raw/
├── chewinggum/                  <- downloaded from VisA (not in this repo)
│   ├── image_anno.csv           <- the list of all photos: file, label, mask
│   └── Data/
│       ├── Images/
│       │   ├── Normal/          <- 503 photos of good gum (.JPG)
│       │   └── Anomaly/         <- 100 photos of gum with a defect (.JPG)
│       └── Masks/
│           └── Anomaly/         <- 100 masks, one per defect photo (.png)
└── split_csv/                   <- VisA's own splits (in this repo, unchanged copies)
    ├── 1cls.csv
    ├── 2cls_highshot.csv
    └── 2cls_fewshot.csv
```

- All photos are 1342 × 1118 pixels (notebook 1.1).
- A **mask** is a map of the defect. Each pixel holds a number: 0 = no defect, 1 to 6 = the defect class, as listed in VisA's [`utils/id2class.py`](https://github.com/amazon-science/spot-diff/blob/main/utils/id2class.py). VisA's README counts 6 classes for chewing gum, but class 6 (*other*) is never used, so the data has 5 defect types (notebook 1.1).
- Normal photos have no mask, because there is nothing to mark.

| Defect type | Photos | What it looks like |
| --- | --- | --- |
| chunk of gum missing | 47 | a piece of the top surface is gone (a rough hole) |
| corner missing | 36 | a piece is broken off an edge or corner |
| small cracks | 29 | thin cracks in the surface |
| similar colour spot | 25 | a faint stain, close to the gum's own colour |
| scratches | 16 | thin scratch lines on the surface |

43 of the 100 defect photos have more than one type, so the counts add up to 153 (notebook 2.1).

## What the EDA found (notebook 2.1)

- **The defects are tiny.** In 63 of the 100 defect photos, the defects cover less than 1% of the photo. *small cracks* and *scratches* are the smallest, with a median of 419 and 724 pixels out of 1,500,356.
- **The gum is always near the middle** of the photo, but its exact place moves a little.
- **The masks are not always clean.** In one photo (068.JPG), one scratch is drawn as 32 separate pieces. In another (057.JPG), a crack has only 1 pixel left in its mask. One YOLO box per mask piece would give many tiny, useless boxes, so step 3 will join nearby pieces and drop specks.

## Data leakage: how it is prevented

**Data leakage** means that information from the test photos reaches a model while it learns or while we choose its settings. The test score then looks better than it really is. This project prevents it like this:

- **No exact copies.** No photo file is an exact copy of another (notebook 1.1).
- **No gum photographed twice across parts.** 6 gums were photographed twice: the two photos of each pair look the same, with the same small bumps and marks. A careful comparison of the gum surfaces gives these 6 pairs a score of 0.569 to 0.727, while every other pair that was compared scores 0.210 or lower. For each pair we keep the first photo and leave out the second, so these 6 normal photos are never used (notebook 1.2).
- **Every photo is in exactly one part**, and every part has the right size (notebook 1.4).
- **The test photos are only used for the final score.** PatchCore learns from the normal train photos only, YOLO learns from train photos only, and the PASS/REJECT limit is chosen on the validation photos (checked in notebook 1.4 and followed in every later step).
- **Settings are never tuned on test photos.** The EDA looks at all photos, but only to describe the data.
- **Preprocessing settings come from train photos only.** The crop size is measured on the normal train photos of each split. The masks are only used to check the crops and to write YOLO's labels, with the same rule for every photo (notebooks 3.1 and 3.2).
- **YOLO's lists hold the right photos.** Every train list holds only train photos of its own split, and no list holds a left-out photo (checked in notebook 3.3).

## How the data is split (notebooks 1.3 and 1.4)

VisA has three ready-made splits:

- **1-class** (`1cls.csv`): the model learns from one class only, the normal photos. For chewing gum: train 453 normal + 0 defect, test 50 normal + 100 defect.
- **2-class high-shot** (`2cls_highshot.csv`): the model learns from normal and defect photos, with many defect examples ("shot" = example). Train 302 + 60, test 201 + 40.
- **2-class few-shot** (`2cls_fewshot.csv`): the same with only a few defect examples. Train 101 + 20, test 402 + 80.

None of them fits this project:

- 1-class gives YOLO no defect photos to learn from, and the 2-class splits give only two fixed amounts (60 or 20).
- There is no fair shared test set: 180 of the 241 high-shot test photos are train photos in 1-class, so a PatchCore trained with 1-class would be tested on photos it learned from.
- No file has a validation part to choose the PASS/REJECT limit.
- They do not know about the photos of the same gum.

So we make our own split:

| Part | Normal | Defect | Used for |
| --- | --- | --- | --- |
| train | 347 | 40 | PatchCore: the 347 normal photos only. YOLO: the 347 normal photos + 5, 10, 20 or 40 defect photos |
| validation | 50 | 10 | choosing the score limit between PASS and REJECT |
| test | 100 | 50 | the final score of both models |
| excluded | 6 | 0 | nothing: the second photo of a gum that was photographed twice |

- The split is made 3 times (random seeds 1, 2 and 3), so one lucky split cannot decide the result. Two splits share 48 to 51 of their 150 test photos.
- Defect photos are split evenly by type, so every part gets every type.
- The YOLO sets grow step by step: the 10 defect photos include the first 5, the 20 include the first 10, and so on.
- Notebook 1.4 runs 83 checks on the three split files, including the leakage checks above.

Because the split is our own, the scores cannot be compared one-to-one with scores in VisA papers.

## Preprocessing (notebooks 3.1 to 3.3)

**Crop the gum (3.1).** The gum covers only about a fifth of each photo; the rest is mat. So every photo is cut down to a **928 × 608** window around its gum (37.6% of the photo), and its mask is cut with the same window.

- Window size = the biggest gum among the normal train photos + 64 pixels on every side, rounded up to a multiple of 32. All three splits give the same size. The 64 pixels also cover the darker top strip of the gum (at most 37 pixels high), which the brightness rule that finds the gum misses.
- Every photo gets the same window size, so nothing is stretched: a defect keeps its real size in pixels.
- The crops are saved as PNG, which keeps every pixel exactly as in the original photo.
- At the same picture width, the crop shows the gum 1.45 times bigger than the whole photo, so thin cracks keep more pixels.
- Check with the masks: no real defect is cut off. Only a 3-pixel speck on the mat of one photo lies outside its window, most likely a slip when the mask was drawn.

**YOLO labels (3.2).** Every mask becomes boxes, separately for each defect type:

- pieces of fewer than 10 pixels (specks) are dropped,
- pieces of one type at most 20 pixels apart are joined into one box,
- boxes thinner than 16 pixels are widened to 16 pixels, so YOLO's random resizing during training does not throw them away.

This turns the 255 pieces of the 100 masks into 205 boxes. Normal photos get an empty label file. For every split there is a train list for every k, a validation list and a test list. In the train lists every defect photo is copied, so that about one line in three is a defect photo (35 copies each for k = 5, 4 for k = 40). Without copies, 98.6% of the k = 5 training pictures would show no defect. With them, every train list has 507 to 527 lines, so YOLO trains about the same amount for every k.

**Checks (3.3).** 109 checks: crops and masks are pixel-exact copies of their windows, every label file has the right format and matches its mask, every list holds exactly the right photos, and YOLO's own path rules find every picture and label file.

**Not done on purpose:** no resizing (each model chooses its own picture size in step 4), no colour or brightness changes, no augmentation saved to disk (YOLO changes the pictures randomly while it trains), and no rotation (the gum always lies straight).

## Modeling (notebooks 4.1 to 4.3)

**PatchCore (4.1).** Written with PyTorch, following [the PatchCore paper](https://arxiv.org/abs/2106.08265). For each split it learns from the 347 normal train photos only:

- A WideResNet-50-2, pretrained on ImageNet and not trained again, looks at every crop at half size (464 × 304) as a grid of 38 × 58 = 2,204 patches. Every patch becomes a list of 1,536 numbers (its feature), taken from two middle layers of the network and averaged with its 3 × 3 neighbours.
- The 764,788 patch features of the normal train photos form the memory bank. A greedy coreset keeps 7,648 of them (1%), chosen so that every normal patch still has a close neighbour among them.
- A new photo is checked patch by patch: the distance of every patch to its nearest patch in the memory bank. These distances form the heatmap, and the largest one is the photo's anomaly score.
- Notebook 4.1 only looks at the validation scores. The test scores are saved for step 5.

**YOLO (4.2).** YOLO26n from Ultralytics, pretrained on COCO, learns from boxes (supervised). For each split and each k (5, 10, 20, 40), it trains on the 347 normal train photos plus k labeled defect photos, with all defect types as one class "defect". All 12 models use the same settings: 100 epochs, pictures of 928 pixels (the crop width, so nothing is shrunk), batch 16, and Ultralytics' default augmentation without upside-down flips (the light always comes from below). Every model is taken after its last epoch (`last.pt`), never the epoch that scored best on the validation photos, so the validation photos never choose a model. One more model learns the 5 defect types by name. It was meant for the demo, but the demo does not use it (see Demo). Training all 13 models took 3.5 hours on the laptop GPU. Notebook 4.3 then gives every validation and test photo one YOLO score: the highest confidence of all places YOLO proposes in it, the counterpart of PatchCore's score. There is no lower limit on the confidence, because with one, many good photos would get no box at all and would all share the score 0, and then they could not be put in order. Like notebook 4.1, notebook 4.3 only looks at the validation scores; the test scores are saved for step 5.

## How the models are judged

Both models give one score per photo: PatchCore's anomaly score, and for YOLO the confidence of its most confident box. A photo with a score above the model's limit gets REJECT.

Main score: **the catch rate, the share of defect test photos that get REJECT, when at most 5% of the good validation photos get REJECT by mistake.**

1. For every model and split, sort the scores of the 50 normal validation photos from high to low. At most 5% of them, so 2 photos, may get REJECT: the limit is the 3rd-highest score.
2. Use that limit once on the 150 test photos of the split: the catch rate on the 50 defect photos, and the false-reject rate (good photos that get REJECT) on the 100 normal photos.
3. Report the mean of the 3 splits, and the lowest and highest split.

The limit comes from only 50 normal photos, so on the test photos the false-reject rate can end up a bit above or below 5%. That is why it is reported next to the catch rate, together with the **AUROC**: a score that does not need a limit, from 0.5 (guessing) to 1.0 (perfect).

At each k, a model **clearly catches more** than the other only if it catches more defect test photos in all 3 splits; otherwise there is no clear winner. These rules were written down before the test photos were looked at (notebook 5.1), and they are not changed afterwards.

We will also compare the models per defect type, check whether the heatmap or the boxes point at the real defect, and measure the time per photo.

## Results

### Catching defects (notebook 5.1)

The test photos of the 3 splits together: 150 defect photos and 300 good photos per model. The percentages and the AUROC are the means of the 3 splits.

| Model | Labeled defect photos | Defect photos caught | Good photos rejected | AUROC |
| --- | --- | --- | --- | --- |
| PatchCore | 0 | 146 of 150 (97.3%) | 20 of 300 (6.7%) | 0.985 |
| YOLO | 5 | 125 of 150 (83.3%) | 17 of 300 (5.7%) | 0.936 |
| YOLO | 10 | 137 of 150 (91.3%) | 30 of 300 (10.0%) | 0.954 |
| YOLO | 20 | 139 of 150 (92.7%) | 19 of 300 (6.3%) | 0.978 |
| YOLO | 40 | 147 of 150 (98.0%) | 8 of 300 (2.7%) | 0.994 |

- **Answer to the main question:** with 5, 10 or 20 labeled defect photos, YOLO catches fewer defects than PatchCore, which needs no labeled photo at all. By the rule above, PatchCore clearly catches more at k = 5 and k = 20. At k = 10 there is no clear winner, but only because of one split, and YOLO is still 9 photos behind over the 3 splits.
- **YOLO catches up at k = 40,** the largest amount tried here: 147 against 146 defect photos caught (no clear winner), with fewer good photos rejected (8 against 20 of 300) and a higher AUROC.
- **What this means for a factory:** without any labeled defect, PatchCore already catches about 97% of the defect photos. YOLO needs about 40 labeled defect photos to be as good, and then it also throws away less good gum.
- **Hardest defect type:** *corner missing*. Every defect photo that PatchCore misses has a missing corner (4 of 150), and so do the 3 photos that YOLO with k = 40 misses.
- **Together:** at k = 40, every defect test photo is caught by at least one of the two models.
- **The limit:** it is set on only 50 good validation photos, so on the test photos the false-reject rate of a split ended up between 2% and 14% instead of 5%.

### Where the models look (notebook 5.2)

- **Right for the right reason:** when a model catches a defect photo, its most suspicious place (the reddest patch of PatchCore's heatmap, or the centre of YOLO's most confident box) lies on the real defect in 97.9% of these photos for PatchCore, and in 95.2% (k = 5) to 98.6% (k = 40) for YOLO.
- **Different misses:** PatchCore misses 2 photos (one of them in all 3 splits) with only tiny pieces broken off the corners. There the photo shows only a little more mat, and mat looks normal to PatchCore. YOLO with k = 40 misses 3 other photos with bigger missing corners. Each model catches the photos the other one misses.
- **False rejects follow the limit:** in each model, the split with the lowest limit has the most false rejects (PatchCore: 14 of 100 in split 2).
- **A possible unmarked defect:** the good photo that both models reject most strongly (`Normal/353`) has a thin line across the gum, and both models mark that line.

### Time per photo (notebook 5.3)

PatchCore of split 1 and YOLO of split 1 with k = 40, on the development laptop (see How to run), one photo at a time, from reading the crop file to the score. Median of 150 test photos on the GPU and of 20 on the CPU:

| Model | GPU (RTX 5070 Laptop) | CPU (20 threads) |
| --- | --- | --- |
| PatchCore | 49.3 ms (20.3 photos per second) | 153.1 ms (6.5 photos per second) |
| YOLO, k = 40 | 30.5 ms (32.7 photos per second) | 105.8 ms (9.4 photos per second) |

YOLO is the faster model on both devices. The number of labeled photos does not change YOLO's time, because its network is the same size for every k.

## Demo (notebook 6.1 and the app)

The app, **GumCheck**, shows both models on the same photo: PASS or REJECT, the score against the limit, PatchCore's heatmap and YOLO's boxes. It has 3 pages: *Inspect a photo*, *Results* (the numbers of step 5) and *How it works*.

**The models of the app.** PatchCore and YOLO with k = 40 of split 1, with the limits of notebook 5.1: the same two models as in notebook 5.3. Split 1 and k = 40 were fixed in notebook 4.2, before any test result: split 1 is the first split, and k = 40 is YOLO with all 40 labeled defect photos, so YOLO at its strongest. Notebook 4.2 also trained a 5-class model for the demo, but step 5 never tested it and never set a limit for it, so the app does not use it.

**Photos to try.** The app has 3 sets of photos, and an upload button:

| Set | Photos | Where the result comes from |
| --- | --- | --- |
| Test photos | 5 test photos of split 1 | the saved results of notebooks 4.1 and 4.3, so exactly what step 5 measured |
| Hard cases | 4 of the 9 test photos of split 1 that one of the two models got wrong | the same saved results |
| New photos | 9 new camera photos in `test_photos/` | checked live by both models |
| Upload | your own photo: a full camera photo, or a cut-out gum of 928 × 608 pixels | checked live by both models |

A live check first cuts out the gum with the rule of notebook 3.1. An uploaded photo that is a photo of the dataset is found by its fingerprint (a tiny grey copy of the crop, 58 × 38 pixels): the app then shows its right answer, and warns when the models learned from it or when it helped to set the limits.

**How the new photos were made (notebook 6.1).** A factory camera sends photos that no model has seen. To try the app like that, notebook 6.1 makes 9 new photos (4 good, 5 with a defect), only from photos that no split 1 model learned from:

- **Good photos:** 4 of the 6 photos left out in notebook 1.2, mirrored left to right (so the light still comes from below). Each is the second photo of a gum whose first photo is a validation or test photo of split 1. The other 2 left-out photos are not used, because the first photo of their gum is a train photo of split 1.
- **Defect photos:** 5 good test photos of split 1, mirrored, each with a real defect copied in from a defect test photo of split 1. The defect goes to the same place on the gum, its edge is made soft, and its brightness is matched to the new gum.
- **Not copies:** the fingerprint of every new photo differs by at least 6.40 from every crop of the dataset. The same photo saved again as JPEG differs by at most 0.27, and the app takes a difference of 0.64 or less as the same photo.

`test_photos/answers.csv` gives the right answer of each new photo, its defect types, how it was made and the real defect boxes. The gallery photos and the new photos come from VisA (CC BY 4.0); the new photos were changed as described above.

**Start the app** (after Setup):

```
.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py
```

Then open the address it shows (Local URL, `http://localhost:8501`) in the browser, and stop the app with Ctrl+C in the terminal. The app only reads files that are in this repo, so it works without the VisA download and without running the notebooks: the two models of the demo are in the repo (`models/patchcore/split_1_memory_bank.npz` and `models/yolo/split_1_k40/weights/last.pt`). The first live check downloads the ImageNet weights of WideResNet-50-2 into PyTorch's cache folder, if notebook 4.1 has not done that yet. Live checks use the GPU if there is one, and show the time of each model for that photo.

**Online.** The app also runs at [visual-defect-inspection-gumcheck.streamlit.app](https://visual-defect-inspection-gumcheck.streamlit.app), on [Streamlit Community Cloud](https://streamlit.io/cloud), for free. There it has no GPU, so a live check takes a few seconds, and the first live check also downloads the ImageNet weights of WideResNet-50-2, which can take up to a minute. An app without visitors for 12 hours goes to sleep; the next visitor wakes it with one click. Two files are only for the cloud: `app/requirements.txt` (the libraries, with the CPU-only build of PyTorch) and `packages.txt` (two Linux libraries that OpenCV needs; YOLO uses OpenCV).

Nothing in step 6 changes a model, a limit or a result.

## Project steps

| Step | Notebook | What it does | Status |
| --- | --- | --- | --- |
| 0. Setup | `0.1_check_environment` | checks Python, the library versions and that PyTorch can use the GPU | done |
| 1. Data setup | `1.1_check_raw_data` | shows what is inside the raw data and checks that it is complete and correct (25 checks) | done |
| | `1.2_find_same_gum_photos` | finds photos of the same gum taken twice, and chooses which ones to leave out | done |
| | `1.3_split_data` | counts VisA's own splits, then makes our 3 train / validation / test splits | done |
| | `1.4_check_splits` | checks the split files, including the data leakage checks (83 checks) | done |
| 2. EDA | `2.1_explore_data` | defect types, masks, defect sizes and positions, mask pieces | done |
| 3. Preprocessing | `3.1_crop_gum` | cuts a 928 × 608 window around the gum out of every photo and mask | done |
| | `3.2_make_yolo_labels` | turns the masks into YOLO boxes, and writes the label files, photo lists and dataset files | done |
| | `3.3_check_preprocessing` | checks every file of step 3, including the data leakage checks (109 checks) | done |
| 4. Modeling | `4.1_train_patchcore` | builds PatchCore, learns the normal patches of each split, and saves a score and a heatmap for every validation and test photo | done |
| | `4.2_train_yolo` | trains YOLO with 5, 10, 20 and 40 labeled defect photos on each split (12 models), plus a 5-class model (not used, see Demo) | done |
| | `4.3_score_yolo` | gives every validation and test photo a YOLO score: the highest confidence of its boxes | done |
| 5. Evaluation | `5.1_compare_on_test` | sets the PASS/REJECT limit of every model on the validation photos, then compares both models on the test photos | done |
| | `5.2_look_at_mistakes` | checks whether each model points at the real defect, and shows the missed defects and the false rejects with heatmaps and boxes | done |
| | `5.3_time_per_photo` | times PatchCore and YOLO on the same test photos, one photo at a time, on the GPU and on the CPU | done |
| 6. Demo | `6.1_prepare_demo` | makes the gallery photos of the app, 9 new camera photos to try it, and a fingerprint of every photo | done |
| | `app/streamlit_app.py` | the app GumCheck: both models on the same photo, with the heatmap and the boxes | done |

## Folder structure

```
visual-defect-inspection/
├── data/
│   ├── raw/
│   │   ├── chewinggum/                    <- original VisA photos (not in this repo, see How to run)
│   │   └── split_csv/                     <- VisA's own split files
│   └── processed/
│       └── crops/                         <- made by notebooks 3.1 and 3.2 (not in this repo)
│           ├── images/                    <- 603 crops (PNG), in Normal/ and Anomaly/
│           ├── masks/                     <- 100 cropped masks (PNG)
│           ├── labels/                    <- 603 YOLO label files (.txt)
│           ├── crop_boxes.csv             <- where each photo was cut
│           ├── yolo_boxes.csv             <- all YOLO boxes
│           ├── split_1_k5_train.txt, ...  <- photo lists: train (per split and k), val and test (per split)
│           └── split_1_k5.yaml, ...       <- YOLO dataset files (per split and k)
├── splits/                                <- our files, made by notebooks 1.2 and 1.3
│   ├── same_gum_photos.csv
│   ├── split_1.csv
│   ├── split_2.csv
│   └── split_3.csv
├── results/                               <- scores of the models, and the test results of step 5
│   ├── patchcore/                         <- made by notebook 4.1
│   │   ├── split_1_scores.csv, ...        <- one score per validation and test photo (per split)
│   │   └── split_1_patch_scores.npz, ...  <- the score of every patch, for the heatmaps (per split)
│   ├── yolo/                              <- made by notebook 4.3
│   │   ├── split_1_scores.csv, ...        <- one score per validation and test photo and per k (per split)
│   │   └── split_1_boxes.csv, ...         <- the boxes YOLO found in these photos (per split)
│   └── evaluation/                        <- made by notebook 5.1
│       ├── score_limits.csv               <- the PASS/REJECT limit of every model (per split and k)
│       ├── test_decisions.csv             <- PASS or REJECT for every test photo and every model
│       └── test_results.csv               <- catch rate, false-reject rate and AUROC (per split and model)
├── models/                                <- trained models (only the 2 models of the demo are in this repo)
│   ├── patchcore/                         <- made by notebook 4.1
│   │   └── split_1_memory_bank.npz, ...   <- what PatchCore remembers (per split; split 1 is in this repo)
│   └── yolo/                              <- made by notebook 4.2
│       ├── yolo26n.pt                     <- pretrained YOLO26n (downloaded)
│       └── split_1_k5/, ...               <- one folder per model: weights/last.pt, results.csv
│                                             (split_1_k40/weights/last.pt is in this repo)
├── notebooks/
│   ├── 0_setup/
│   │   └── 0.1_check_environment.ipynb
│   ├── 1_data_setup/
│   │   ├── 1.1_check_raw_data.ipynb
│   │   ├── 1.2_find_same_gum_photos.ipynb
│   │   ├── 1.3_split_data.ipynb
│   │   └── 1.4_check_splits.ipynb
│   ├── 2_eda/
│   │   └── 2.1_explore_data.ipynb
│   ├── 3_preprocessing/
│   │   ├── 3.1_crop_gum.ipynb
│   │   ├── 3.2_make_yolo_labels.ipynb
│   │   └── 3.3_check_preprocessing.ipynb
│   ├── 4_modeling/
│   │   ├── 4.1_train_patchcore.ipynb
│   │   ├── 4.2_train_yolo.ipynb
│   │   └── 4.3_score_yolo.ipynb
│   ├── 5_evaluation/
│   │   ├── 5.1_compare_on_test.ipynb
│   │   ├── 5.2_look_at_mistakes.ipynb
│   │   └── 5.3_time_per_photo.ipynb
│   └── 6_demo/
│       └── 6.1_prepare_demo.ipynb
├── app/                                   <- the demo app (step 6)
│   ├── streamlit_app.py                   <- start here: streamlit run app/streamlit_app.py
│   ├── requirements.txt                   <- libraries for the online app (Streamlit Community Cloud)
│   ├── common.py                          <- paths, settings, cutting out the gum, loading the results
│   ├── inference.py                       <- runs both models on a new photo (live checks)
│   ├── views/                             <- the 3 pages: inspect.py, results.py, how_it_works.py
│   ├── data/                              <- made by notebook 6.1: gallery photos, gallery.csv, fingerprints.npz
│   └── static/                            <- logo, and the fonts Geist and Geist Mono by Vercel (license: fonts/LICENSE.TXT)
├── test_photos/                           <- made by notebook 6.1: 9 new camera photos and answers.csv
├── .streamlit/
│   └── config.toml                        <- colours and fonts of the app
├── README.md
├── packages.txt                           <- 2 Linux libraries for the online app (OpenCV needs them)
├── requirements.txt                       <- libraries, with exact versions
├── requirements-torch.txt                 <- PyTorch with GPU support
├── requirements-yolo.txt                  <- Ultralytics (YOLO), installed after PyTorch
└── .gitignore
```

## Setup (Windows)

This is done once per computer. Type the commands in the terminal of VS Code (**Terminal > New Terminal**), which opens in the project folder. Every command calls the Python inside `.venv` directly, so the environment does not need to be "activated".

1. **Check the NVIDIA driver.** Run `nvidia-smi`. It shows your GPU, and at the top right `CUDA Version`. That number must be 13.0 or higher. If it is lower, or the command is not found, update the driver (NVIDIA App > Drivers), restart the laptop and check again.
2. **Make a virtual environment with Python 3.14** (a private Python for this project, in the folder `.venv`). Use the path of your Python 3.14, for example:
   ```
   C:\Python314\python.exe -m venv .venv
   ```
   Do not use Python 3.14.1: torchvision does not support that exact version.
3. **Update pip** (the tool that installs libraries):
   ```
   .venv\Scripts\python.exe -m pip install --upgrade pip
   ```
4. **Install the libraries:**
   ```
   .venv\Scripts\python.exe -m pip install -r requirements.txt
   ```
5. **Install PyTorch with GPU support** (a large download):
   ```
   .venv\Scripts\python.exe -m pip install -r requirements-torch.txt
   ```
   This file downloads PyTorch from the PyTorch website, because the normal package site only has the CPU-only build of PyTorch for Windows. If pip cannot find these versions, open [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/), choose Windows, Pip, Python and the newest CUDA, and copy the `--index-url` it shows into `requirements-torch.txt`.
6. **Install Ultralytics (YOLO)**, always after PyTorch:
   ```
   .venv\Scripts\python.exe -m pip install -r requirements-yolo.txt
   ```
   Ultralytics needs PyTorch. Installed before PyTorch, pip would take the CPU-only build of PyTorch, and the GPU could not be used.
7. **Choose the kernel in VS Code.** Open a notebook, click **Select Kernel** at the top right, then **Python Environments**, then `.venv`.
8. **Run `notebooks/0_setup/0.1_check_environment.ipynb`.** It must end with **ALL CHECKS PASSED**. If not, its last section says what to do.

Library versions are pinned in `requirements.txt`, `requirements-torch.txt` and `requirements-yolo.txt`, and notebook 0.1 checks that the installed versions match.

## How to run

1. Download VisA from the [official repository](https://github.com/amazon-science/spot-diff) (the README there links to the file `VisA_20220922.tar`) and copy the `chewinggum` folder into `data/raw/`. This file must exist: `data/raw/chewinggum/image_anno.csv`. The `split_csv` folder is already in this repo.
2. Do the Setup above once.
3. Run the notebooks in order: 0.1 → 1.1 → 1.2 → 1.3 → 1.4 → 2.1 → 3.1 → 3.2 → 3.3 → 4.1 → 4.2 → 4.3 → 5.1 → 5.2 → 5.3 → 6.1. In VS Code, use **Restart** and then **Run All** for each notebook, so no old result stays on screen. Notebooks 0.1, 1.1, 1.4 and 3.3 must end with **ALL CHECKS PASSED**.
4. Start the app: `.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py` (see Demo). The app alone only needs step 2.

The files in `splits/` are already in this repo. Running notebooks 1.2 and 1.3 again makes the same files. The folder `data/processed/` (about 170 MB) is not in this repo: notebooks 3.1 and 3.2 make it. The folder `models/` is not in this repo either, except the 2 models of the demo: notebooks 4.1 and 4.2 make it. The first time, notebook 4.1 downloads the ImageNet weights of WideResNet-50-2 into PyTorch's cache folder, and notebook 4.2 downloads the pretrained YOLO26n into `models/yolo/`. Notebook 4.2 takes several hours; if it stops, run it again and it continues where it stopped.

Developed on a Windows laptop with an Intel Core Ultra 7 255HX, an NVIDIA GeForce RTX 5070 Laptop GPU (8 GB VRAM) and 32 GB RAM.
