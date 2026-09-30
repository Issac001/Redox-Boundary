"""Reusable B1 audit, B2 single-transition estimation and B3 chemistry routines."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from core import crossings
from boundary_model import TransitionGrid, fit_step

CENTER_RULE_VERSION = "2026-09-30-width-separate"


def center_diagnostics(fits):
    """Classify saved sigmoid fits without changing their numerical estimates.

    ``supported_shape`` retains the legacy joint position/width rule. The
    current ``supported_center`` removes only the width-grid-edge veto;
    ``width_grid_edge`` remains an independent width-resolution diagnostic.
    These operational labels are not calibrated accuracy probabilities.
    """
    edges = {}
    for name in ("grid_edge", "b_grid_edge", "width_grid_edge"):
        parsed = fits[name].astype(str).str.lower().map({"true": True, "false": False})
        if parsed.isna().any():
            raise ValueError(f"Missing or invalid sigmoid boolean values in {name}")
        edges[name] = parsed.astype(bool)
    if not edges["grid_edge"].equals(edges["b_grid_edge"] | edges["width_grid_edge"]):
        raise ValueError("Combined grid-edge flags disagree with position/width flags")
    common = (fits.amplitude.ge(100) & fits.delta_aicc_vs_no_transition.lt(0)
              & fits.b_interval_width25.le(20))
    return pd.DataFrame({
        "supported_shape": common & ~edges["grid_edge"],
        "supported_center": common & ~edges["b_grid_edge"],
    }, index=fits.index)


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2,
                    default=lambda x: x.item() if isinstance(x, np.generic) else str(x)))


def save(out, name, rows):
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    frame.to_csv(out / f"{name}.csv", index=False)
    return frame


def time_rmse(actual, prediction, time_keys):
    mse = pd.DataFrame({"t":np.asarray(time_keys), "e":(np.asarray(actual)-np.asarray(prediction))**2})
    return float(np.sqrt(mse.groupby("t").e.mean().mean()))


def ridge_forward(x, y, tr, ca, te, times, alphas):
    """Select only on the calibration times, then refit using all past labels."""
    scaler = StandardScaler().fit(x[tr])
    scores = []
    for alpha in alphas:
        model = Ridge(alpha=alpha).fit(scaler.transform(x[tr]), y[tr])
        scores.append(time_rmse(y[ca], model.predict(scaler.transform(x[ca])), times[ca]))
    alpha = alphas[int(np.argmin(scores))]
    past = tr | ca
    scaler = StandardScaler().fit(x[past])
    model = Ridge(alpha=alpha).fit(scaler.transform(x[past]), y[past])
    return model.predict(scaler.transform(x[te])), alpha


def aicc(sse, n, parameter_count):
    return n*np.log(np.maximum(sse/n, 1e-12))+2*parameter_count+2*parameter_count*(parameter_count+1)/(n-parameter_count-1)


def diagnose(data, cfg, out):
    records=[]
    for case, ix in data.meta.groupby("Case").groups.items():
        for th in cfg["thresholds_mV"]:
            locations=[crossings(data.eh[i], data.depth, th)[0] for i in ix]
            counts=np.array([len(v) for v in locations])
            records.append({"case":case,"threshold_mV":th,"n":len(ix),
                "single":int((counts==1).sum()),"multiple":int((counts>1).sum()),"none":int((counts==0).sum())})
    save(out,"B1_threshold_audit",records)
    audit=[]
    for (case,depth),part in data.rows.groupby(["Case","Depth"]):
        for marker in ["NO3","NH4","Fe","Mn"]:
            obs=part[part[marker].notna()]
            audit.append({"case":case,"depth_cm":depth,"analyte":marker,"n":len(obs),
                          "n_times":obs.Time.nunique(),"n_zero":int((obs[marker]==0).sum()),
                          "spatial_status":"unverified_outlet" if depth==95 else "main_spatial"})
    save(out,"B1_chemistry_audit",audit)


def estimate_boundaries(data, cfg, out):
    y,z=data.eh,data.depth
    fits={kind:TransitionGrid(z,kind=kind).fit(y) for kind in ["sigmoid","sigmoid_trend"]}
    fits["step"]=fit_step(y,z)
    X=np.column_stack([np.ones(len(z)),(z-z.mean())/85.])
    linear=y @ np.linalg.pinv(X).T @ X.T
    constant=np.repeat(y.mean(axis=1)[:,None],len(z),axis=1)
    null_aic=np.minimum(aicc(np.sum((y-linear)**2,axis=1),len(z),3),
                        aicc(np.sum((y-constant)**2,axis=1),len(z),2))
    records=[]
    interval=TransitionGrid(z).profile_interval(y,sigma=25)
    for kind,fit in fits.items():
        record=data.meta.copy()
        record["model"]=kind
        for key in ["b","w","amplitude","sse","grid_edge","b_grid_edge","width_grid_edge","plateau_unobserved","identifiable"]:
            if key in fit: record[key]=fit[key]
        record["transition_width_10_90_cm"]=2*np.log(9)*record.get("w",np.nan) if kind!="step" else np.nan
        record["fit_rmse_mV"]=np.sqrt(fit["sse"]/len(z))
        record["delta_aicc_vs_no_transition"]=aicc(fit["sse"],len(z),{"sigmoid":5,"sigmoid_trend":6,"step":4}[kind])-null_aic
        if kind=="sigmoid":
            record["b_low_working25"]=interval["b_low"]
            record["b_high_working25"]=interval["b_high"]
            record["b_interval_width25"]=interval["b_high"]-interval["b_low"]
            record["working_interval_components"]=interval["support_components"]
            record["working_interval_truncated"]=interval["grid_truncated"]
            for key, values in center_diagnostics(record).items():
                record[key] = values
        if kind=="step":
            for key in ["lower_gap","upper_gap"]:
                if key in fit: record[key]=fit[key]
        records.append(record)
    table=save(out,"B2_boundary_estimates",pd.concat(records,ignore_index=True))
    sensitivity=[]
    for sigma in cfg["interval_sigmas_mV"]:
        inter=interval if sigma==25 else TransitionGrid(z).profile_interval(y,sigma=sigma)
        sensitivity.append(pd.DataFrame({"profile_id":data.meta.profile_id,"case":data.meta.Case,
            "sigma_mV":sigma,"b_low":inter["b_low"],"b_high":inter["b_high"],
            "working_width_cm":inter["b_high"]-inter["b_low"],
            "support_components":inter["support_components"],"grid_truncated":inter["grid_truncated"]}))
    save(out,"B2_interval_sensitivity",pd.concat(sensitivity,ignore_index=True))
    # Each held-out depth is absent from fitting; candidate grid remains fixed.
    errors={k:[] for k in ["linear","sigmoid","sigmoid_trend","step"]}
    for j in range(len(z)):
        mask=np.arange(len(z))!=j
        x=np.column_stack([np.ones(mask.sum()),(z[mask]-z[mask].mean())/85])
        beta=y[:,mask]@np.linalg.pinv(x).T
        prediction=beta[:,0]+beta[:,1]*(z[j]-z[mask].mean())/85
        errors["linear"].append((prediction-y[:,j])**2)
        for kind in ["sigmoid","sigmoid_trend","step"]:
            if kind=="step":
                fitted=fit_step(y[:,mask],z[mask])
                prediction=np.where(z[j]<fitted["b"],fitted["left_mean"],fitted["right_mean"])
            else:
                model=TransitionGrid(z[mask],kind=kind)
                fitted=model.fit(y[:,mask])
                prediction=model.predict(fitted,np.array([z[j]]))[:,0]
            errors[kind].append((prediction-y[:,j])**2)
    loo=[]
    for kind,values in errors.items():
        mse=np.nanmean(np.array(values),axis=0)
        for case,ix in data.meta.groupby("Case").groups.items():
            loo.append({"case":case,"model":kind,"n_profiles":len(ix),"rmse_mV":float(np.sqrt(np.mean(mse[ix])))})
    save(out,"B2_leave_depth_out",loo)
    return fits,table


def chemical_features(rows, fits, meta):
    lookup=pd.Series(np.arange(len(meta)),index=pd.MultiIndex.from_frame(meta[["Case","Time"]]))
    ids=lookup.reindex(pd.MultiIndex.from_frame(rows[["Case","Time"]])).to_numpy(int)
    z=rows.Depth.to_numpy(float); base=np.column_stack([z/85,(z/85)**2,rows.WL.to_numpy()/100])
    features={"depth_water":base,"plus_Eh":np.column_stack([base,rows.Eh.to_numpy()/500])}
    features["fixed_depth_zones"]=np.column_stack([base,*(z>cut for cut in [35,55,75])])
    for th in [0,200,400]: features[f"threshold_{th}"]=np.column_stack([base,(rows.Eh.to_numpy()>th).astype(float)])
    for kind,fit in fits.items():
        b=np.asarray(fit["b"])[ids]
        missing=~np.isfinite(b)
        distance=np.nan_to_num((z-b)/85,nan=0.)
        if kind=="step": q=(z<b).astype(float)
        else: q=expit((b-z)/np.asarray(fit["w"])[ids])
        # Missing is explicit, neutral imputation never silently drops difficult profiles.
        q=np.nan_to_num(q,nan=.5)
        features[f"boundary_{kind}"]=np.column_stack([base,q,distance,missing.astype(float)])
    return features


def chemistry_validation(data, fits, cfg, out):
    metrics=[]; predictions=[]; pertime=[]; foldrows=[]
    for include_outlet in [False,True]:
        for fold in cfg["folds"]:
            rows=data.rows[data.rows.Case.eq(fold["case"])].copy()
            # Use the same audited cycle metadata, never infer cycles from chemical values.
            rows=rows.drop(columns="Cycle").merge(data.meta[["Case","Time","Cycle"]],on=["Case","Time"],validate="many_to_one")
            if not include_outlet: rows=rows[rows.Depth<=90]
            for analyte in ["NO3","NH4","Fe","Mn"]:
                obs=rows[rows[analyte].notna()].sort_values(["Time","Depth"])
                tr=(obs.Cycle<=fold["train_end"]).to_numpy()
                ca=((obs.Cycle>fold["train_end"])&(obs.Cycle<=fold["cal_end"])).to_numpy()
                te=((obs.Cycle>fold["cal_end"])&(obs.Cycle<=fold["test_end"])).to_numpy()
                if min(tr.sum(),ca.sum(),te.sum())<2: continue
                y=np.log1p(obs[analyte].to_numpy(float)); times=obs.Time.to_numpy()
                common={"case":fold["case"],"fold":fold["fold"],"role":fold["role"],"analyte":analyte,"include_95cm":include_outlet}
                foldrows.append({**common,"train_rows":tr.sum(),"cal_rows":ca.sum(),"test_rows":te.sum(),
                    "train_times":len(np.unique(times[tr])),"cal_times":len(np.unique(times[ca])),"test_times":len(np.unique(times[te]))})
                features=chemical_features(obs,fits,data.meta)
                for name,x in features.items():
                    pred,alpha=ridge_forward(x,y,tr,ca,te,times,cfg["ridge_alphas"])
                    metrics.append({**common,"model":name,"rmse_log1p":time_rmse(y[te],pred,times[te]),
                         "n_rows":te.sum(),"n_times":len(np.unique(times[te])),"alpha":alpha})
                    test=obs.loc[te]
                    for i,(_,row) in enumerate(test.iterrows()):
                        predictions.append({**common,"model":name,"time_day":row.Time,"depth_cm":row.Depth,
                            "observed_log1p":y[te][i],"predicted_log1p":pred[i]})
                    for t in np.unique(times[te]):
                        same=times[te]==t
                        pertime.append({**common,"model":name,"time_day":t,"n_depths":same.sum(),
                            "mse_log1p":float(np.mean((pred[same]-y[te][same])**2))})
    save(out,"B3_chemistry_metrics",metrics);save(out,"B3_chemistry_predictions",predictions)
    save(out,"B3_chemistry_by_time",pertime);save(out,"B3_fold_audit",foldrows)
    # Chemical-only transitions are descriptive; none enter held-out predictions above.
    centers=[]
    for (case,t),part in data.rows[data.rows.Depth<=90].groupby(["Case","Time"]):
        for analyte,direction in [("NO3",1),("NH4",-1),("Fe",-1),("Mn",-1)]:
            obs=part[part[analyte].notna()].sort_values("Depth")
            if len(obs)<5: continue
            y=direction*np.log1p(obs[analyte].to_numpy(float))
            fitted=TransitionGrid(obs.Depth.to_numpy()).fit(y)
            centers.append({"case":case,"time_day":t,"analyte":analyte,"n_depths":len(obs),
                "b":fitted["b"][0],"w":fitted["w"][0],"amplitude_log1p":fitted["amplitude"][0],
                "grid_edge":fitted["grid_edge"][0],"role":"descriptive_chemical_only"})
    save(out,"B3_chemical_transition_centers",centers)
    # Equal-channel standardized profile losses: descriptive joint fit only.
    # These chemical labels never enter the prediction features evaluated above.
    shared=[]
    for case in cfg["cases"]:
        cutoff=next(f["train_end"] for f in cfg["folds"] if f["case"]==case)
        past_times=data.meta.loc[data.meta.Case.eq(case)&data.meta.Cycle.le(cutoff),"Time"]
        past=data.rows[data.rows.Case.eq(case)&data.rows.Time.isin(past_times)&data.rows.Depth.le(90)]
        scales={}
        for marker in ["Eh","NO3","NH4","Fe","Mn"]:
            available=past[marker].dropna().to_numpy(float)
            if len(available)<2: continue
            transformed=available if marker=="Eh" else np.log1p(available)
            scales[marker]=max(float(np.std(transformed,ddof=1)),.05)
        for t,part in data.rows[data.rows.Case.eq(case)&data.rows.Depth.le(90)].groupby("Time"):
            profiles=[]; best={}
            for marker,direction in [("Eh",1),("NO3",1),("NH4",-1),("Fe",-1),("Mn",-1)]:
                if marker not in scales: continue
                obs=part[part[marker].notna()].sort_values("Depth")
                if len(obs)<5: continue
                yy=direction*(obs[marker].to_numpy() if marker=="Eh" else np.log1p(obs[marker].to_numpy()))
                grid=TransitionGrid(obs.Depth.to_numpy())
                result=grid.profile_interval(yy,sigma=1.)
                curve=result["profile_sse"][0]/(len(obs)*scales[marker]**2)
                profiles.append(curve);best[f"b_{marker}"]=result["b"][0]
            if len(profiles)<3: continue
            total=np.sum(profiles,axis=0)
            if not np.isfinite(total).all(): raise ValueError("Nonfinite shared transition objective")
            best_index=int(np.argmin(total));separate=float(sum(np.min(c) for c in profiles))
            shared.append({"case":case,"time_day":t,"common_b":grid.b_grid[best_index],
                 "n_channels":len(profiles),"shared_standardized_loss":float(total[best_index]),
                 "separate_standardized_loss":separate,"constraint_loss_increase":float(total[best_index]-separate),
                 "shared_grid_edge":bool(best_index in [0,len(grid.b_grid)-1]),
                 "role":"descriptive_joint_fit_not_validation",**best})
    save(out,"B3_shared_transition_diagnostic",shared)
