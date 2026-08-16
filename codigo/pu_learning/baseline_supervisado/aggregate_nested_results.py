#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
import pandas as pd


def main():
    p=argparse.ArgumentParser(); p.add_argument("--run-dir",required=True); args=p.parse_args()
    root=Path(args.run_dir); rows=[]; errors=[]
    for task in sorted((root/"results").glob("task_*")):
        s=task/"summary.json"
        if not s.exists(): errors.append({"task":task.name,"error":"missing_summary"}); continue
        try:
            row=json.loads(s.read_text()); row["result_dir"]=str(task); rows.append(row)
        except Exception as e: errors.append({"task":task.name,"error":str(e)})
    df=pd.DataFrame(rows)
    if len(df):
        df.to_csv(root/"aggregate_metrics.csv",index=False)
        complete=df[(df.status=="completed") & (df.n_outer_folds==5)].copy()
        if len(complete):
            complete.sort_values(["target","feature_set","average_precision","gmean","mcc"],ascending=[True,True,False,False,False]).to_csv(root/"complete_ranked.csv",index=False)
    report={"expected_tasks":30,"summary_files":len(rows),"complete_tasks":int(((df.status=="completed") & (df.n_outer_folds==5)).sum()) if len(df) else 0,"errors":errors}
    (root/"aggregate_report.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)); print(json.dumps(report,indent=2,ensure_ascii=False))

if __name__=="__main__": main()
