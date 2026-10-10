"""Task14 train-only cached detection/keypoint evaluation; no training or deployment."""
from __future__ import annotations
import argparse,csv,hashlib,json,sys
from pathlib import Path
import cv2
import numpy as np

REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO/'src'),str(REPO/'scripts')]
from evaluate_bbox_study import probabilities,decode,predictions,write_csv,gt_size
from run_centernet_cv import _load_cfg
from run_epoch_study import select
from fomo_servo.centernet.evaluation import evaluate_threshold,match_image,spearman
from fomo_servo.centernet.keypoints import load_keypoint_samples,decode_offsets
from fomo_servo.geometry.letterbox import LetterboxTransform
from fomo_servo.postprocess.connected_components import find_connected_components

EPOCHS=(20,40,60,100,150,200,250)
THRESHOLDS=tuple(round(i/100,3) for i in range(5,96,5))+tuple(round(i/1000,3) for i in range(960,996,5))

def finite(value):
    """Replace undefined numeric statistics with JSON null, recursively."""
    if isinstance(value,dict):return {k:finite(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [finite(v) for v in value]
    if isinstance(value,(float,np.floating)) and not np.isfinite(value):return None
    return value

def angle(pred,gt):
    """Directed vector error in degrees 0..180; zero-length prediction is undefined."""
    norm=np.linalg.norm(pred)*np.linalg.norm(gt)
    return float(np.degrees(np.arccos(np.clip(np.dot(pred,gt)/norm,-1,1)))) if norm>1e-8 else float('nan')

def matched_rows(probs,offsets,samples,transforms,cfg,epoch,threshold,scope):
    """Decode at predicted component centre; GT is used only after centroid matching."""
    ds=decode(probs,transforms,cfg,threshold); rows=[]
    stride=cfg['model']['output_stride']
    for probability,offset,det,sample,tr in zip(probs,offsets,ds,samples,transforms):
        components=[comp for cid in range(probability.shape[0]-1)
                    for comp in find_connected_components(probability[cid+1]>=threshold,connectivity=8)]
        assert len(components)==len(det)
        boxes=[(b,k) for b,k in zip(sample.boxes,sample.keypoints) if b.visibility!='ignore']
        matches=match_image(predictions([det])[0],sample.boxes,class_agnostic=True)
        for pi,gi,distance in matches.pairs:
            box,kp=boxes[gi]
            if kp.orientation not in ('side','oblique') or (kp.head is None and kp.tail is None):continue
            d=det[pi]
            x,y=min(components[pi].cells,key=lambda c:((c[0]+.5-d.heatmap_x)**2+(c[1]+.5-d.heatmap_y)**2,c[1],c[0]))
            points=decode_offsets(offset[:,y,x],cell_x=x,cell_y=y,stride=stride)
            head,tail=np.asarray([tr.inverse_point(*p) for p in points])
            diag=float(np.hypot(box.x_max-box.x_min,box.y_max-box.y_min))
            row={'selection_scope':scope,'epoch':epoch,'threshold':threshold,'image':sample.image_path.name,
                 'session':sample.session,'source_class':kp.source_class,'orientation':kp.orientation,
                 'nonignore_gt_index':gi,
                 'visibility':box.visibility,'head_state':kp.head_state,'tail_state':kp.tail_state,
                 'sample_cell_x':x,'sample_cell_y':y,'component_area_cells':d.component_area_cells,
                 'center_error_px':distance,'gt_size_sqrt_area_px':gt_size(box),'gt_box_diag_px':diag,
                 'pred_head_x':float(head[0]),'pred_head_y':float(head[1]),'pred_tail_x':float(tail[0]),'pred_tail_y':float(tail[1]),
                 'gt_head_x':None,'gt_head_y':None,'gt_tail_x':None,'gt_tail_y':None,
                 'head_error_box_diag':None,'tail_error_box_diag':None,'angle_error_deg':None,
                 'axis_error_deg':None,'oracle_long_edge_axis_error_deg':None,
                 'gt_length_px':None,'pred_length_px':float(np.linalg.norm(tail-head)),'length_ratio':None}
            for name,pred in [('head',head),('tail',tail)]:
                point=getattr(kp,name)
                if point is not None:
                    row['gt_'+name+'_x'],row['gt_'+name+'_y']=point
                    row[name+'_error_box_diag']=float(np.linalg.norm(pred-np.asarray(point))/diag)
            if kp.head is not None and kp.tail is not None:
                vector=np.asarray(kp.tail)-np.asarray(kp.head)
                error=angle(tail-head,vector)
                row['angle_error_deg']=error
                row['axis_error_deg']=min(error,180-error)
                axis=np.asarray([1.,0.]) if box.x_max-box.x_min>=box.y_max-box.y_min else np.asarray([0.,1.])
                ae=angle(axis,vector)
                row['oracle_long_edge_axis_error_deg']=min(ae,180-ae)
                row['gt_length_px']=float(np.linalg.norm(vector))
                row['length_ratio']=row['pred_length_px']/row['gt_length_px'] if row['gt_length_px']>0 else None
            rows.append(row)
    return rows

def summarize(rows):
    """Aggregate matched-only statistics with explicit denominators; no confidence tuning."""
    result=[]
    for label in ['all','fish','tuna']:
        for ori in ['all','side','oblique']:
            subset=[r for r in rows if (label=='all' or r['source_class']==label) and (ori=='all' or r['orientation']==ori)]
            pairs=[r for r in subset if r['gt_length_px'] is not None]
            valid=[r for r in pairs if r['angle_error_deg'] is not None and np.isfinite(r['angle_error_deg'])]
            med=lambda key,rs:float(np.median([r[key] for r in rs if r[key] is not None])) if any(r[key] is not None for r in rs) else float('nan')
            result.append({'source_class':label,'orientation':ori,'matched_annotated_boxes':len(subset),
                           'complete_pairs':len(pairs),'valid_directions':len(valid),'undefined_directions':len(pairs)-len(valid),
                           'angle_median_deg':med('angle_error_deg',valid),
                           'angle_under30_fraction':sum(r['angle_error_deg']<30 for r in valid)/len(valid) if valid else float('nan'),
                           'angle_over90_fraction':sum(r['angle_error_deg']>90 for r in valid)/len(valid) if valid else float('nan'),
                           'head_n':sum(r['head_error_box_diag'] is not None for r in subset),'head_error_median_box_diag':med('head_error_box_diag',subset),
                           'tail_n':sum(r['tail_error_box_diag'] is not None for r in subset),'tail_error_median_box_diag':med('tail_error_box_diag',subset),
                           'length_ratio_median':med('length_ratio',pairs),'keypoint_axis_error_median_deg':med('axis_error_deg',valid),
                           'oracle_long_edge_axis_error_median_deg':med('oracle_long_edge_axis_error_deg',pairs),
                           'pred_length_vs_gt_size_rho':spearman([r['pred_length_px'] for r in pairs],[r['gt_size_sqrt_area_px'] for r in pairs]),
                           'gt_length_vs_gt_size_rho':spearman([r['gt_length_px'] for r in pairs],[r['gt_size_sqrt_area_px'] for r in pairs]),
                           'component_area_vs_gt_size_rho_same_subset':spearman([r['component_area_cells'] for r in pairs],[r['gt_size_sqrt_area_px'] for r in pairs])})
    return result

def plots(out,rows,table,samples,baseline_f1):
    """Save error distributions, tuna-side length scatter and deterministic quantile overlays."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    for label,color in [('fish','tab:blue'),('tuna','tab:orange')]:
        rs=[r for r in rows if r['source_class']==label and r['angle_error_deg'] is not None]
        axes[0].hist([r['angle_error_deg'] for r in rs],bins=np.arange(0,181,15),alpha=.5,label=label,color=color)
    axes[0].set(xlabel='Directed head-to-tail error (deg)',ylabel='Matched pairs');axes[0].legend()
    rs=[r for r in rows if r['source_class']=='tuna' and r['orientation']=='side' and r['gt_length_px'] is not None]
    axes[1].scatter([r['gt_size_sqrt_area_px'] for r in rs],[r['pred_length_px'] for r in rs],label='predicted')
    axes[1].scatter([r['gt_size_sqrt_area_px'] for r in rs],[r['gt_length_px'] for r in rs],marker='x',label='GT')
    axes[1].set(xlabel='Tuna side GT sqrt(box area), px',ylabel='Head-tail length, px');axes[1].legend()
    for threshold in [.96]:
        epochs=[r['epoch'] for r in table if r['threshold']==threshold]
        values=[r['f1'] for r in table if r['threshold']==threshold]
        axes[2].plot(epochs,values,'o-',label='keypoint @0.96')
    axes[2].axhline(baseline_f1,color='grey',linestyle='--',label='B-box50 on v3 GT')
    axes[2].set(xlabel='Epoch',ylabel='Pooled detection F1');axes[2].legend()
    fig.tight_layout();fig.savefig(out/'metrics.png',dpi=150);plt.close(fig)
    valid=sorted([r for r in rows if r['angle_error_deg'] is not None and np.isfinite(r['angle_error_deg'])],key=lambda r:r['angle_error_deg'])
    chosen=[valid[round(q*(len(valid)-1))] for q in np.linspace(0,1,6)] if valid else []
    by_name={s.image_path.name:s for s in samples}
    fig,axes=plt.subplots(2,3,figsize=(15,8))
    for ax,r in zip(axes.flat,chosen):
        im=cv2.cvtColor(cv2.imread(str(by_name[r['image']].image_path)),cv2.COLOR_BGR2RGB);ax.imshow(im)
        for prefix,color in [('gt','lime'),('pred','orange')]:
            a=np.asarray([r[prefix+'_head_x'],r[prefix+'_head_y']]);b=np.asarray([r[prefix+'_tail_x'],r[prefix+'_tail_y']])
            ax.arrow(*a,*(b-a),color=color,width=2,head_width=12,length_includes_head=True)
            ax.plot(*a,'o',color=color)
        ax.set_title(f"{r['source_class']} {r['orientation']} error={r['angle_error_deg']:.1f}°\n{r['image']}",fontsize=8);ax.axis('off')
    for ax in axes.flat:ax.axis('off')
    fig.suptitle('Matched angle-error quantiles 0/20/40/60/80/100%; GT green, prediction orange',fontsize=11)
    fig.tight_layout();fig.savefig(out/'angle_quantile_examples.png',dpi=130);plt.close(fig)
    write_csv(out/'example_selection.csv',chosen)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['config','dataset-root','work','baseline-work','results']:p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();cfg=_load_cfg(args.config);issues=[]
    samples=load_keypoint_samples(args.dataset_root,issues=issues);by_name={s.image_path.name:s for s in samples}
    names=[];cache={e:[] for e in EPOCHS};offsets={e:[] for e in EPOCHS};baseline=[];manifest=[]
    for fold in cfg['data']['held_out_sessions']:
        directory=args.work/(fold+'__B-box50-kp');meta=json.loads((directory/'meta.json').read_text())
        assert all(by_name[n].session==fold for n in meta['test_images_list'])
        bm=json.loads((args.baseline_work/(fold+'__B-box50')/'meta.json').read_text())
        assert bm['test_images_list']==meta['test_images_list']
        names.extend(meta['test_images_list'])
        for epoch in EPOCHS:
            path=directory/f'outputs_e{epoch}.npz'
            with np.load(path) as z:
                a=z['outputs'];b=z['offsets']
                assert a.shape==(len(meta['test_images_list']),8,24,24) and b.shape==(len(meta['test_images_list']),4,24,24)
                assert a.dtype==b.dtype==np.float32 and np.isfinite(a).all() and np.isfinite(b).all()
                cache[epoch].append(a);offsets[epoch].append(b)
            manifest.append({'path':str(path.resolve()),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        bp=args.baseline_work/(fold+'__B-box50')/'outputs_e250.npz'
        baseline.append(np.load(bp)['outputs']);manifest.append({'path':str(bp.resolve()),'sha256':hashlib.sha256(bp.read_bytes()).hexdigest()})
    pooled=[by_name[n] for n in names];transforms=[]
    for sample in pooled:
        a=sample.image_path.with_suffix('.json')
        if a.exists():d=json.loads(a.read_text(encoding='utf-8'));w,h=d['imageWidth'],d['imageHeight']
        else:h,w=cv2.imread(str(sample.image_path)).shape[:2]
        transforms.append(LetterboxTransform.from_image_size(w,h,cfg['model']['input_size']))
    gt=[s.boxes for s in pooled];table=[];data={e:probabilities(np.concatenate(cache[e])) for e in EPOCHS}
    for epoch in EPOCHS:
        for threshold in THRESHOLDS:
            metrics=evaluate_threshold(predictions(decode(data[epoch],transforms,cfg,threshold)),gt,class_agnostic=True,adjacent_distance=cfg['data']['distance_threshold_px'])
            table.append({'epoch':epoch,'threshold':threshold,**{k:metrics[k] for k in ['tp','fp','fn','precision','recall','f1']}})
        print('scan complete',epoch,flush=True)
    selected=select({(r['epoch'],r['threshold']):r['f1'] for r in table})
    base=evaluate_threshold(predictions(decode(probabilities(np.concatenate(baseline)),transforms,cfg,.96)),gt,class_agnostic=True,adjacent_distance=cfg['data']['distance_threshold_px'])
    baseline_areas={}
    baseline_detections=decode(probabilities(np.concatenate(baseline)),transforms,cfg,.96)
    for sample,ds in zip(pooled,baseline_detections):
        for pi,gi,_ in match_image(predictions([ds])[0],sample.boxes,class_agnostic=True).pairs:
            baseline_areas[(sample.image_path.name,gi)]=ds[pi].component_area_cells
    selected_metrics=next(r for r in table if (r['epoch'],r['threshold'])==selected)
    fixed=next(r for r in table if (r['epoch'],r['threshold'])==(250,.96))
    all_rows=[];summaries={}
    for scope,(epoch,threshold) in [('selected',selected),('fixed250_0.96',(250,.96))]:
        rows=matched_rows(data[epoch],np.concatenate(offsets[epoch]),pooled,transforms,cfg,epoch,threshold,scope)
        for r in rows:r['baseline_component_area_cells']=baseline_areas.get((r['image'],r['nonignore_gt_index']))
        all_rows.extend(rows);summaries[scope]=summarize(rows)
    common=[r for r in all_rows if r['selection_scope']=='selected' and r['source_class']=='tuna' and r['orientation']=='side' and r['gt_length_px'] is not None and r['baseline_component_area_cells'] is not None]
    common_comparison={'n':len(common),
                       'pred_length_vs_gt_size_rho':spearman([r['pred_length_px'] for r in common],[r['gt_size_sqrt_area_px'] for r in common]),
                       'task13_baseline_area_vs_gt_size_rho':spearman([r['baseline_component_area_cells'] for r in common],[r['gt_size_sqrt_area_px'] for r in common]),
                       'keypoint_model_area_vs_gt_size_rho':spearman([r['component_area_cells'] for r in common],[r['gt_size_sqrt_area_px'] for r in common])}
    counts={}
    for s in samples:
        for k in s.keypoints:
            if k.head is not None or k.tail is not None:
                key=k.source_class+'/'+str(k.orientation)
                c=counts.setdefault(key,{'annotated_boxes':0,'complete_pairs':0,'heads':0,'tails':0})
                c['annotated_boxes']+=1;c['complete_pairs']+=int(k.head is not None and k.tail is not None);c['heads']+=int(k.head is not None);c['tails']+=int(k.tail is not None)
    out=args.results;out.mkdir(parents=True,exist_ok=False)
    write_csv(out/'threshold_scan.csv',table);write_csv(out/'matched_keypoints.csv',all_rows)
    write_csv(out/'group_metrics.csv',[{'selection_scope':scope,**r} for scope,rs in summaries.items() for r in rs])
    summary={'selected':selected_metrics,'fixed250_0.96':fixed,'baseline_Bbox50_250_0.96_v3':base,
             'baseline_task13_F1':.7142857142857143,
             'selected_detection_F1_drop_gt_0.02':selected_metrics['f1']<base['f1']-.02,
             'fixed250_detection_F1_drop_gt_0.02':fixed['f1']<base['f1']-.02,
             'groups':summaries,'annotation_counts':counts,'annotation_issues':issues,'thresholds':THRESHOLDS,
             'tuna_side_common_matched_baseline_comparison':common_comparison,
             'epochs':EPOCHS,'input_manifest':manifest,'heldout_image_order':names,
             'decode_rule':'nearest component cell to probability-weighted detection centroid; tie y,x; no GT at inference',
             'axis_baseline':'oracle GT-box long edge; unsigned modulo180; box not provided by FOMO',
             'offset_reference':'geometric centre of centre-containing grid cell, signed log1p(grid-cell offsets)',
             'selection_rule':'Task08 select; same CV folds for selection/report; no independent labelled test'}
    (out/'summary.json').write_text(json.dumps(finite(summary),indent=2,allow_nan=False),encoding='utf-8')
    plots(out,[r for r in all_rows if r['selection_scope']=='selected'],table,pooled,base['f1'])
    print(json.dumps(finite({'selected':selected_metrics,'groups':summaries['selected']}),indent=2,allow_nan=False),flush=True)

if __name__=='__main__':main()
