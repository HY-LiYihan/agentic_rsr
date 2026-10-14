from pathlib import Path
import json, subprocess, sys

ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'runs/sim_processed_stage3_all'
records=[]
for robot in ('piper','franka'):
    for task_dir in sorted((ROOT/'data/processed'/robot/'stage3').iterdir()):
        if not task_dir.is_dir(): continue
        source=task_dir/'mujoco_v8/scene.xml'
        target=out/robot/task_dir.name/'scene_for_sim.xml'
        cmd=[sys.executable,str(ROOT/'scripts/prepare_sim_scene.py'),'--source',str(source),'--output',str(target)]
        result=subprocess.run(cmd,capture_output=True,text=True)
        rec={'robot':robot,'task':task_dir.name,'source':str(source),'output':str(target),'ok':result.returncode==0}
        if result.returncode: rec['error']=result.stderr[-1200:]
        else:
            try: rec.update(json.loads(result.stdout))
            except Exception: pass
        records.append(rec)
        print(json.dumps(rec,ensure_ascii=False),flush=True)
(out/'prepare_all_report.json').parent.mkdir(parents=True,exist_ok=True)
(out/'prepare_all_report.json').write_text(json.dumps(records,ensure_ascii=False,indent=2))
print('prepared',sum(r['ok'] for r in records),'of',len(records))
