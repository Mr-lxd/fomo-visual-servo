"""Task 10 A': pair eight numbered originals with ordered Pi snapshots.

Inference uses shared RGB/letterbox preprocessing and OnnxRuntimePredictor.
SIFT/RANSAC only transfers the original target ROI to unmodified snapshots;
the model always sees the full unrectified camera frame. Coordinates are pixels.
"""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2
import numpy as np
from fomo_servo.inference.ort_predictor import OnnxRuntimePredictor
from fomo_servo.inference.preprocessing import preprocess_rgb_image, prediction_from_numpy_logits


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def inference(predictor, bgr):
    """Return raw foreground probabilities [C,G,G] and original-pixel grid centres."""
    c = predictor.contract
    prepared = preprocess_rgb_image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), input_size=c.input_shape[-1])
    logits = predictor.predict_logits(prepared.input_tensor)
    ex = np.exp(logits[0] - logits[0].max(axis=0, keepdims=True))
    fg_classes = (ex/ex.sum(axis=0, keepdims=True))[1:]
    yy, xx = np.indices(fg_classes.shape[1:])
    t = prepared.transform
    ox = ((xx+.5)*c.output_stride-t.pad_left)/t.scale
    oy = ((yy+.5)*c.output_stride-t.pad_top)/t.scale
    # Diagnostic decoding at 0.05; deployed-threshold decoding is reported separately.
    preds = {}
    for threshold in (.05, .60):
        preds[threshold] = prediction_from_numpy_logits(
            prepared, logits, class_names=c.class_names, output_stride=c.output_stride,
            confidence_threshold=threshold, class_thresholds=c.class_thresholds,
            component_mode=c.component_mode, confidence_mode=c.confidence_mode).detections
    return fg_classes, prepared, ox, oy, preds


def peak(prob, ox, oy, polygon, class_names, class_id=None):
    """Maximum foreground (or specified class) at grid centres inside a polygon."""
    values = prob.max(axis=0) if class_id is None else prob[class_id]
    mask = np.array([cv2.pointPolygonTest(polygon, (float(x), float(y)), False)>=0
                     for x,y in zip(ox.ravel(),oy.ravel())]).reshape(values.shape)
    gy,gx = np.unravel_index(np.where(mask, values, -1).argmax(), values.shape)
    assert mask[gy,gx], "target ROI contains no heatmap cell centre"
    cid = int(prob[:,gy,gx].argmax()) if class_id is None else class_id
    return {"score":float(values[gy,gx]),"x":float(ox[gy,gx]),"y":float(oy[gy,gx]),"class":class_names[cid]}


def overlay(bgr, prob, prepared, ox, oy, polygon):
    """Fixed 0–1 foreground heatmap, all 8-neighbour local maxima >=0.05."""
    fg=prob.max(axis=0)
    h,w=bgr.shape[:2]; yy,xx=np.indices((h,w),dtype=np.float32)
    t=prepared.transform; stride=t.input_size/fg.shape[0]
    sampled=cv2.remap(fg, (xx*t.scale+t.pad_left)/stride-.5,
                      (yy*t.scale+t.pad_top)/stride-.5, cv2.INTER_LINEAR)
    heat=cv2.applyColorMap(np.uint8(np.clip(sampled*255,0,255)),cv2.COLORMAP_TURBO)
    alpha=np.where(sampled>=.05,.45,0)[...,None]
    out=np.uint8(bgr*(1-alpha)+heat*alpha)
    cv2.polylines(out,[np.int32(polygon)],True,(0,255,0),2)
    maxima=(fg>=cv2.dilate(fg,np.ones((3,3),np.uint8))) & (fg>=.05)
    n,labels=cv2.connectedComponents(maxima.astype(np.uint8),connectivity=8)
    peaks=[]
    for label in range(1,n):
        gy,gx=np.unravel_index(np.where(labels==label,fg,-1).argmax(),fg.shape)
        x,y=float(ox[gy,gx]),float(oy[gy,gx])
        if not 0<=x<w or not 0<=y<h:
            continue
        score=float(fg[gy,gx]); cid=int(prob[:,gy,gx].argmax())
        peaks.append({"score":score,"x":x,"y":y,"class_id":cid})
        cv2.circle(out,(round(x),round(y)),4,(255,255,255),1)
        cv2.putText(out,f"{score:.2f}",(min(w-45,max(0,round(x)+5)),max(12,round(y)-5)),
                    cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1,cv2.LINE_AA)
    return out,peaks


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--originals",type=Path,required=True)
    p.add_argument("--captures",type=Path,required=True)
    p.add_argument("--model",type=Path,required=True)
    p.add_argument("--report",type=Path,required=True)
    p.add_argument("--results",type=Path,required=True)
    a=p.parse_args()
    predictor=OnnxRuntimePredictor.from_files(a.model,a.report)
    c=predictor.contract
    rows=list(csv.DictReader((a.originals/"expected.csv").open(encoding="utf-8-sig")))
    metadata=json.loads((a.captures/"metadata.json").read_text(encoding="utf-8"))
    snapshots=sorted(metadata["snapshots"],key=lambda s:s["capture_timestamp_ns"])
    assert len(rows)==len(snapshots)==8
    a.results.mkdir(parents=True,exist_ok=True)
    scores=[]; registrations=[]; all_peaks=[]
    sift=cv2.SIFT_create()
    for r,snapshot in zip(rows,snapshots):
        original=a.originals/r["file"]
        captured=a.captures/Path(snapshot["filename"]).name
        assert sha(original)==r["image_sha256"]
        assert c.onnx_sha256==r["model_sha256"]
        src=cv2.imread(str(original)); cap=cv2.imread(str(captured))
        x0,y0,x1,y1=[float(r[k]) for k in ("roi_x0","roi_y0","roi_x1","roi_y1")]
        roi=np.float32([[x0,y0],[x1,y0],[x1,y1],[x0,y1]])
        k1,d1=sift.detectAndCompute(cv2.cvtColor(src,cv2.COLOR_BGR2GRAY),None)
        k2,d2=sift.detectAndCompute(cv2.cvtColor(cap,cv2.COLOR_BGR2GRAY),None)
        matches=[m for m,n in cv2.BFMatcher().knnMatch(d1,d2,k=2) if m.distance<.8*n.distance]
        p1=np.float32([k1[m.queryIdx].pt for m in matches]); p2=np.float32([k2[m.trainIdx].pt for m in matches])
        H,inliers=cv2.findHomography(p1,p2,cv2.RANSAC,4)
        capture_roi=cv2.perspectiveTransform(roi[None],H)[0]
        registrations.append({"id":r["id"],"homography":H.tolist(),"matches":len(matches),
                              "inliers":int(inliers.sum()),"snapshot_roi_polygon":capture_roi.tolist()})
        sides=[]
        row={"id":r["id"],"group":r["group"],"source_class":r["source_class"],
             "source_image":r["file"],"snapshot":captured.name,"saved_utc":snapshot["saved_utc"],
             "original_sha256":sha(original),"snapshot_sha256":sha(captured),"model_sha256":c.onnx_sha256,
             "source_size_px":r["gt_size_px"]}
        intended=1 if r["source_class"]=="jellyfish" else 0
        for label,bgr,polygon in (("original",src,roi),("snapshot",cap,capture_roi)):
            prob,prepared,ox,oy,preds=inference(predictor,bgr)
            response=peak(prob,ox,oy,polygon,c.class_names)
            correct=peak(prob,ox,oy,polygon,c.class_names,intended)
            global_roi=np.float32([[0,0],[bgr.shape[1],0],[bgr.shape[1],bgr.shape[0]],[0,bgr.shape[0]]])
            global_peak=peak(prob,ox,oy,global_roi,c.class_names)
            for key,value in response.items(): row[label+"_target_"+key]=value
            row[label+"_correct_class_score"]=correct["score"]
            row[label+"_correct_class_x"]=correct["x"]
            row[label+"_correct_class_y"]=correct["y"]
            for key,value in global_peak.items(): row[label+"_global_"+key]=value
            hits=[d for d in preds[.60] if d.class_id==intended and cv2.pointPolygonTest(polygon,(d.original_x,d.original_y),False)>=0]
            row[label+"_target_detected_at_0_60"]=int(bool(hits))
            row[label+"_detections_at_0_05"]=len(preds[.05])
            if label=="original":
                assert abs(correct["score"]-float(r["target_score"]))<1e-6
            rendered,peaks=overlay(bgr,prob,prepared,ox,oy,polygon)
            for peak_row in peaks:
                all_peaks.append({"id":r["id"],"side":label,**peak_row,"class":c.class_names[peak_row["class_id"]]})
            canvas=np.zeros((520,640,3),np.uint8)
            canvas[:480]=rendered
            title=f"{r['id']} {label}: ROI max {response['score']:.3f} ({response['class']}); {correct['class']} {correct['score']:.3f}"
            cv2.putText(canvas,title,(10,507),cv2.FONT_HERSHEY_SIMPLEX,.46,(255,255,255),1,cv2.LINE_AA)
            sides.append(canvas)
        row["score_drop"]=row["original_target_score"]-row["snapshot_target_score"]
        row["correct_class_score_drop"]=row["original_correct_class_score"]-row["snapshot_correct_class_score"]
        scores.append(row)
        cv2.imwrite(str(a.results/(r["id"]+"_paired_heatmaps.jpg")),np.hstack(sides))
        print(r["id"],r["group"],"ROI foreground",row["original_target_score"],"->",row["snapshot_target_score"],
              "snapshot class",row["snapshot_target_class"],"correct class",row["snapshot_correct_class_score"],
              "global",row["snapshot_global_score"],"detected@.60",row["snapshot_target_detected_at_0_60"])
    write_csv(a.results/"comparison.csv",scores)
    write_csv(a.results/"local_peaks.csv",all_peaks)
    (a.results/"registration.json").write_text(json.dumps(registrations,indent=2),encoding="utf-8")
    (a.results/"provenance.json").write_text(json.dumps({"model_sha256":c.onnx_sha256,"sidecar_sha256":sha(a.report),
        "metadata_sha256":sha(a.captures/"metadata.json"),"expected_sha256":sha(a.originals/"expected.csv"),
        "script_sha256":sha(Path(__file__)),"session":metadata["session_id"],"analysis_threshold":.05,
        "deployment_threshold_unchanged":.60,"real_tuna_in_water_snapshots":"not taken (user)"},indent=2),encoding="utf-8")


if __name__ == "__main__":
    main()
