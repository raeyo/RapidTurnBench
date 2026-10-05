"""CSV, static public report, plots and video overlay for the Isaac benchmark."""
from pathlib import Path
import csv
import html
import json
import numpy as np
from PIL import Image,ImageDraw,ImageFont

LABELS={90:'LEFT',-90:'RIGHT',180:'TURN BACK',0:'STRAIGHT'}
def overlay(rgb,t,angle,speed,actual,model,switch):
    frame=Image.fromarray(rgb[:,:,:3].astype('uint8'));d=ImageDraw.Draw(frame)
    try:font=ImageFont.truetype('DejaVuSans.ttf',28);small=ImageFont.truetype('DejaVuSans.ttf',17)
    except OSError:font=small=ImageFont.load_default()
    w,h=frame.size
    d.rectangle((0,0,w,94),fill=(15,23,35))
    label=LABELS.get(int(angle),str(angle)) if t>=switch else 'FORWARD'
    if t<1:label='STAND'
    text=label+'  |  '+(f'{t-switch:.2f} s' if t>=switch else f't={t:.2f} s')
    bbox=d.textbbox((0,0),text,font=font);d.text(((w-bbox[2])/2,10),text,font=font,fill='white')
    # Arrows drawn as shapes so labels do not depend on Unicode font support.
    cx=w//2
    if t>=switch and angle in (90,-90):
        sign=-1 if angle==90 else 1;start=cx-100*sign;end=cx+100*sign
        d.line((start,62,end,62),fill=(80,220,170),width=6)
        d.polygon([(end,62),(end-16*sign,51),(end-16*sign,73)],fill=(80,220,170))
    elif t>=switch and angle==180:
        d.arc((cx-35,48,cx+35,84),0,270,fill=(80,220,170),width=5)
        d.polygon([(cx-35,62),(cx-24,50),(cx-20,69)],fill=(80,220,170))
    d.rectangle((0,h-46,w,h),fill=(15,23,35))
    d.text((12,h-38),f'{model} | Isaac Lab | target {speed:.2f} / actual {actual:.2f} m/s',font=small,fill='white')
    return np.asarray(frame)

def write_report(directory,records,config):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    with (directory/'episodes.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    rows=[]
    for speed,angle in sorted(set((r['requested_speed_m_s'],r['turn_angle_deg']) for r in records)):
        group=[r for r in records if (r['requested_speed_m_s'],r['turn_angle_deg'])==(speed,angle)]
        def mean(key,subset=None):
            values=[r[key] for r in (group if subset is None else subset) if r.get(key) is not None]
            return float(np.mean(values)) if values else None
        complete=[r for r in group if r.get('transient_window_complete')]
        rows.append(dict(model=group[0]['model'],requested_speed_m_s=speed,turn_angle_deg=angle,N=len(group),
                         approach_ready_N=sum(bool(r.get('approach_ready')) for r in group),
                         reached_N=sum(bool(r.get('turn_reached')) for r in group),
                         turn_time_s=mean('turn_time_s'),actual_approach_speed_m_s=mean('pre_cue_actual_speed_m_s'),
                         velocity_error_m_s=mean('transient_velocity_error_m_s',complete),velocity_error_N=len(complete),
                         peak_tilt_deg=mean('peak_tilt_deg'),fall_N=sum(bool(r.get('fall')) for r in group),
                         error_N=sum(r['status'] in ('policy_error','invalid_action','initialization_failed') for r in group)))
    with (directory/'summary.csv').open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
    def cell(x):return '—' if x is None else f'{x:.3f}' if isinstance(x,float) else str(x)
    headers=list(rows[0]);table='<table><thead><tr>'+''.join(f'<th>{html.escape(k)}</th>' for k in headers)+'</tr></thead><tbody>'
    for row in rows:table+='<tr>'+''.join(f'<td>{html.escape(cell(row[k]))}</td>' for k in headers)+'</tr>'
    table+='</tbody></table>'
    videos=''.join(f'<p>{html.escape(p.stem)}</p><video controls preload="metadata" src="{p.relative_to(directory)}"></video>' for p in sorted((directory/'videos').glob('*.mp4')) if p.name!='demo.mp4')
    plots=''.join(f'<img src="{p.relative_to(directory)}" alt="{html.escape(p.stem)}">' for p in sorted((directory/'figures').glob('*.png')))
    content=f'''<!doctype html><meta charset="utf-8"><title>RapidTurn Isaac benchmark</title>
<style>body{{font-family:system-ui;background:#101724;color:#eef;margin:32px}}table{{border-collapse:collapse;font-size:13px}}td,th{{padding:9px;border:1px solid #526}}video,img{{max-width:900px;width:100%}}a{{color:#7ed}}</style>
<h1>RapidTurn · Isaac Lab</h1><p>Functional quickcheck · {html.escape(str(records[0]['model']))}</p>
<p>Direction change after an abrupt cue. Simulation physics: Isaac Lab / PhysX. These are measured states, not an animation.</p>
<p>Times average reached episodes only; reached N and approach-ready N remain visible. Missing values are not zero. Velocity error averages complete 1.5 s windows only. No aggregate leaderboard score is defined.</p>
<p><a href="summary.csv">Summary CSV</a> · <a href="episodes.csv">Episode CSV</a> · <a href="provenance.json">Provenance</a> · <a href="videos/demo.mp4">Combined demo</a></p>{table}{plots}{videos}
<p>Videos support human qualitative review; no automatic qualitative score is claimed. Representative seed is fixed in the profile.</p>'''
    (directory/'report.html').write_text(content)
    md='| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'
    md+='\n'.join('| '+' | '.join(cell(row[k]) for k in headers)+' |' for row in rows)
    (directory/'summary.md').write_text(md+'\n')
    (directory/'records.json').write_text(json.dumps(records,indent=2))
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots()
        for record in records:
            path=directory/'episodes'/record['episode_id']/'episode.npz'
            if path.exists():
                with np.load(path) as log:xy=log['root_position'][:,:2]
                ax.plot(xy[:,0],xy[:,1],label=f"{record['turn_angle_deg']:+g} deg seed {record['seed']}")
        ax.set(xlabel='World X (m)',ylabel='World Y (m)',title='Actual Isaac root trajectories');ax.axis('equal');ax.legend(fontsize=6)
        (directory/'figures').mkdir(exist_ok=True);fig.savefig(directory/'figures/trajectory_xy.png',dpi=140);plt.close(fig)
        # Rebuild once to include plots generated above.
        if not plots:(directory/'report.html').write_text(content.replace(table,table+'<img src="figures/trajectory_xy.png" alt="Actual root trajectories">'))
    except ImportError:pass
