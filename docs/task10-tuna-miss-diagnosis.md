# Task 10 phase A — tuna miss diagnosis (C complete, A' screen package ready)

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

## Evidence and pending inputs

Results: `D:\RoboBeetle-results\task10-tuna-miss-2026-10-07\C\` (`recall.csv`, `objects.csv`, `missed_tuna.csv`, `reflection_tuna.csv`, `summary.json`, `provenance.json`, `tuna_scores.png`). Complete per-size/per-source/visibility slices are in `recall.csv`.

Deployment ONNX SHA-256 verified read only: `05acdc7a83264100be6d19ca8ba5641ba7c6448337038bc99f3a74326d225c10`.

Task brief v3 replaces the previous recording/timestamp request with eight numbered screen snapshots. No recording or video timestamps are needed. Current Pi address and capture paths are needed before read-only scp; the latest user-provided address is 192.168.137.86. No Pi connection has been made.

No analysis unit tests were added or run, per task instructions. Verification is execution on the requested held-out data, exact pooled-metric reproduction, per-object CSV accounting and visual inspection of the histogram. UI change is a separate RoboBeetle draft PR.

## Phase A' numbered screen package (v3)

Run `scripts/prepare_tuna_screen_test.py --dataset-root <lab_pool_v2_vis> --model <lab_pool_v2_fomo_seed42_e150.onnx> --report <onnx-sidecar.json> --output <results>/screen-test`.

Eight unchanged 640x480 JPGs are copied into `screen-test`, numbered 01–08. 01/02 are small tuna (<150 px), 03/04 medium tuna (150–250 px), 05/06 large tuna (>=350 px), 07 jellyfish, 08 source fish. Eligible boxes have source visibility full, correct-class raw probability >=0.8 and a detection whose centroid lies in the source box at offline threshold 0.60. The highest-scoring distinct source images in each group are chosen deterministically. Source annotation visibility is retained; this is not a new visibility review.

Target scores for 01–08: 0.9800, 0.9551, 0.9904, 0.9869, 0.9254, 0.8343, 0.99997, 0.99980. Tuna is displayed as fish under the fixed D2 class mapping. These deliberately selected known positives from training data are a paired screen-domain diagnostic, not an independent model-accuracy evaluation.

`expected.csv` records target-box score/location/class and ROI, whole-image highest foreground score/location/class, deployed-threshold detection score, source image, original-pixel size, model and image hashes. Keeping target and global peaks separate matters when other objects have higher scores. Source and copied image hashes agree; all eight dimensions and scores were verified. An annotated contact sheet outside the image package was visually inspected; the eight display originals have no overlays or altered pixels.

User procedure: open JPG 01 through 08 full screen in order, aim the USB camera at the display and click Snapshot once per image. No recording and no timestamps. After capture, use read-only scp for the eight snapshots and optional direct-camera real tuna-in-water snapshots; infer with the same model at offline analysis threshold 0.05 and produce paired probability overlays and a score table. No runtime threshold changes, deployment, retraining or phase B.

UI PR #56 was moved to ready and merged with a merge commit on explicit user authorization: `43b6b71e3093447031e243746aa5d76ec4237cb9`. Reviewed head remains `4fdfb63418486ecc4e93da0752fcc44e0d9af843`; its previously completed single Qt regression was 31/31. fomo PR #16 remains draft.
