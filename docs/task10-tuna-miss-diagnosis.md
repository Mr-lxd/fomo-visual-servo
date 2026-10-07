# Task 10 phase A / A' — completed tuna screen-domain diagnosis

Scope: diagnosis only. No training, runtime threshold change, deployment, or phase B.

## Reproduce section C

Run `scripts/diagnose_tuna_cv.py --reference-root <task08-checkout> --dataset-root <lab_pool_v2_vis> --work <task08-checkout>/outputs/epoch-study/work --results <results>/C` with NumPy, OpenCV, matplotlib and PyYAML.

Reference code: `claude/epoch-study`, commit `789748f473e3fc4d0389340eee36306d3f585f96` (PR #14). This script reuses its `load_pool_samples`, `_predictions`, letterbox geometry and `match_image`; the reference checkout is read only, including suppression of Python bytecode writes. Cached held-out B epoch150 and C epoch100 raw outputs are the exact Task 08 outputs, not newly trained weights. Task 08 retained output snapshots rather than per-epoch fold weights; `provenance.json` records cache/meta/config/image/annotation SHA-256. Full deployment model is separate from these cross-validation caches.

Each of sessions 001–004 contributes only its held-out predictions. Session 005 is never read. Matches are computed once per complete image, class agnostic, one to one, using the existing centroid-in-box matcher; size and source slices are then attributed to GT rather than rematched independently. Ignore boxes are excluded from the denominator, with their original matcher behavior retained. Size is sqrt(width * height) in original pixels, with lower-inclusive boundaries 80/150/250/350. Source labels and visibility come from the original v2 JSON; tuna maps to fish for model decoding only.

Foreground maximum is softmax(logits), max over all seven foreground classes, restricted to stride-8 grid centres whose inverse-letterbox coordinates lie inside the clipped source box. Empty grid-centre regions are recorded as null. These are foreground response maxima, not confidence of a source-specific tuna class (there is no separate tuna output).

## Observed section C results

The pooled P/R/F1 exactly reproduce Task 08: B150@0.60 = 0.635/0.648/0.642; B150@0.40 = 0.549/0.714/0.621; C100@0.35 = 0.617/0.555/0.584. Total scored GT = 290, excluding 32 ignore boxes. Source GT: jellyfish 148, fish 49, tuna 93. Tuna visibility: full 52, truncated 41, occluded 0.

B150@0.60 tuna recall is 61/93 (65.6%), comparable to jellyfish 97/148 (65.5%) and above fish 30/49 (61.2%). Full tuna recall is 40/52 (76.9%) versus truncated 21/41 (51.2%). At 0.40 these become 66/93, 43/52 and 23/41. Large tuna >=350 px: 18/22 at 0.60, 19/22 at 0.40. These observations do not support a general size-driven tuna failure.

Of the 32 tuna misses at 0.60, box maximum foreground scores are <0.2 for 17, 0.2–0.3 for 3, and 0.3–0.6 for 12. Lowering the offline threshold recovers a net five tuna, while increasing pooled false positives from 108 to 170; it is not a complete explanation or a deployment recommendation.

True tuna box maxima: median 0.795; reflection tuna (27 boxes) median 0.0216, maximum 0.588. The model suppresses these reflection boxes on held-out images. This comparison cannot establish whether reflection-negative training causally suppresses true tuna, or whether merging tuna into fish is harmful; there is no counterfactual model in phase A.

## Evidence

Results: `D:\RoboBeetle-results\task10-tuna-miss-2026-10-07\C\` (`recall.csv`, `objects.csv`, `missed_tuna.csv`, `reflection_tuna.csv`, `summary.json`, `provenance.json`, `tuna_scores.png`). Complete per-size/per-source/visibility slices are in `recall.csv`.

Deployment ONNX SHA-256 verified read only: `05acdc7a83264100be6d19ca8ba5641ba7c6448337038bc99f3a74326d225c10`.

Task brief v3 replaces the previous recording/timestamp request with eight numbered screen snapshots. No recording or video timestamps are needed. The eight snapshots were retrieved by read-only scp from 192.168.137.86 after the user reported completion. User explicitly reports no real tuna-in-water snapshots.

No analysis unit tests were added or run, per task instructions. Verification is execution on the requested held-out data, exact pooled-metric reproduction, per-object CSV accounting and visual inspection of the histogram. UI change is separate RoboBeetle PR #56 (merged on user authorization).

## Phase A' numbered screen package (v3)

Run `scripts/prepare_tuna_screen_test.py --dataset-root <lab_pool_v2_vis> --model <lab_pool_v2_fomo_seed42_e150.onnx> --report <onnx-sidecar.json> --output <results>/screen-test`.

Eight unchanged 640x480 JPGs are copied into `screen-test`, numbered 01–08. 01/02 are small tuna (<150 px), 03/04 medium tuna (150–250 px), 05/06 large tuna (>=350 px), 07 jellyfish, 08 source fish. Eligible boxes have source visibility full, correct-class raw probability >=0.8 and a detection whose centroid lies in the source box at offline threshold 0.60. The highest-scoring distinct source images in each group are chosen deterministically. Source annotation visibility is retained; this is not a new visibility review.

Target scores for 01–08: 0.9800, 0.9551, 0.9904, 0.9869, 0.9254, 0.8343, 0.99997, 0.99980. Tuna is displayed as fish under the fixed D2 class mapping. These deliberately selected known positives from training data are a paired screen-domain diagnostic, not an independent model-accuracy evaluation.

`expected.csv` records target-box score/location/class and ROI, whole-image highest foreground score/location/class, deployed-threshold detection score, source image, original-pixel size, model and image hashes. Keeping target and global peaks separate matters when other objects have higher scores. Source and copied image hashes agree; all eight dimensions and scores were verified. An annotated contact sheet outside the image package was visually inspected; the eight display originals have no overlays or altered pixels.

User procedure: open JPG 01 through 08 full screen in order, aim the USB camera at the display and click Snapshot once per image. No recording and no timestamps. After capture, use read-only scp for the eight snapshots and optional direct-camera real tuna-in-water snapshots; infer with the same model at offline analysis threshold 0.05 and produce paired probability overlays and a score table. No runtime threshold changes, deployment, retraining or phase B.

UI PR #56 was moved to ready and merged with a merge commit on explicit user authorization: `43b6b71e3093447031e243746aa5d76ec4237cb9`. Reviewed head remains `4fdfb63418486ecc4e93da0752fcc44e0d9af843`; its previously completed single Qt regression was 31/31. fomo PR #16 remains draft.

## Phase A' paired-snapshot results

Capture session `capture-20261007-001`, ordered snapshots 1–8, 2026-10-07 15:18:20–15:19:05 Asia/Shanghai. Visual comparison confirms each is the corresponding numbered original. Read-only scp retrieved the service unit to locate capture output, metadata, eight JPEGs, and the configured deployment ONNX/sidecar. ONNX from Pi has SHA-256 `05acdc7a83264100be6d19ca8ba5641ba7c6448337038bc99f3a74326d225c10`, identical to the image-preparation artifact. No SSH commands or Pi writes were issued.

Capture path: `/home/pi/fomo-vision-runtime/datasets_raw/hardware_gate_g_restart/20261007/capture-20261007-001/`. The copied unit points to `artifacts/lab_pool_v2_fomo_seed42_e150.onnx`. Camera metadata: 640x480, YUYV, observed 25 FPS. Unit configuration and file hash identify configured files; no remote process introspection was performed.

Command: `scripts/analyze_tuna_screen_test.py --originals <results>/screen-test --captures <results>/screen-captures --model <results>/screen-captures/lab_pool_v2_fomo_seed42_e150.onnx --report <same>.onnx.json --results <results>/screen-comparison`.

Inference runs once on each unchanged original and full, unchanged camera snapshot using `OnnxRuntimePredictor` and shared RGB/letterbox preprocessing. Diagnostic decoding uses 0.05; detection at the unchanged deployment threshold 0.60 is also reported. SIFT matches (ratio 0.8) and RANSAC homography (4 px) transfer the original target box into the snapshot solely for measuring/reporting that region, not for rectifying inference inputs. All eight projected ROIs and image pairs were visually checked. A homography approximates the visible distortion and is not a lens calibration. Whole-frame maxima are retained separately to avoid confusing another object with tuna.

| ID | Source target | Original class score | Snapshot class score | Target detected @0.60 |
|---|---|---:|---:|---|
| 01 | small tuna | 0.9800 | 0.1064 | no |
| 02 | small tuna | 0.9551 | 0.1168 | no |
| 03 | medium tuna | 0.9904 | 0.6795 | yes |
| 04 | medium tuna | 0.9869 | 0.8586 | yes |
| 05 | large tuna | 0.9254 | 0.5898 | no |
| 06 | large tuna | 0.8343 | 0.7982 | yes |
| 07 | jellyfish | 0.99997 | 0.9650 | yes |
| 08 | fish | 0.99980 | 0.8112 | yes |

The tuna class score is the fish-channel maximum inside the transferred source box, following the fixed class mapping. For 01–05, 07 and 08 it equals the max across foreground classes in that ROI. For 06, all-foreground ROI maximum is 0.9115 (jellyfish) on background near the upper left; visual inspection shows that is not a tuna response. The intended fish response is 0.7982 at the body centre. `comparison.csv` retains both scores, their peak locations, whole-image maxima, hashes and @0.60 detections.

Visual peak positions: 01 near the front body/head region, 02 body middle, 03 rear body, 04 body middle, 05 rear body/dorsal-fin transition, 06 body middle (fish response). These anatomical descriptions are approximate visual observations, not new keypoint labels. Scores 01/02 <0.2 are not rescued at 0.40; 05 lies between 0.40 and 0.60. Three of six intended tuna detections survive photographing the display; both controls survive. All eight correct-class scores fall, but the decrease is much smaller for jellyfish and non-uniform across tuna.

**Judgment: reason 2 is supported, with a threshold effect (reason 1) on image 05.** The identical, known-positive sources produce much lower tuna responses after screen capture. In 01/02 this is severe low-score failure; in 05 the photo response falls just below 0.60. Original image scores are high in all six tuna examples, so these pairs do not support reason 3 as the main explanation. The 06 pair and the prior CV results also contradict a universal large-target failure. Both controls drop somewhat but remain detectable; the experiment does not support "all classes fail on screen" or "all large tuna fail".

These pairs establish a difference for the full screen/camera capture condition. They do not isolate blur, exposure/color, monitor artefacts, perspective, visible geometric distortion or changed effective image scale as separate causes. Real toy-in-water generalization remains untested because the user did not take that optional set. There is no recommendation to change runtime thresholds or train B-box on this evidence.

Outputs: eight `01–08_paired_heatmaps.jpg`, `comparison.csv`, `local_peaks.csv`, `registration.json`, `provenance.json` under `screen-comparison`. Heatmaps use a fixed probability color range 0–1; green outline is the target ROI; all 8-neighbour local foreground maxima >=0.05 are marked with scores. Provenance records Pi ONNX/sidecar, metadata, expected table, script and every image hash. Original target scores reproduce `expected.csv` within 1e-6. Analysis scripts have no unit tests; no product code changed in this phase and no Qt regression was rerun. Phase A/A' is complete; stop for Reviewer, with PR #16 still draft and phase B unstarted.
