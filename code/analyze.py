"""Analyze results/*.json -> summary tables + SVG figures. No matplotlib (SVG by hand)."""
import json
import os
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(BASE, "results")
FIG = os.path.join(BASE, "results", "figures")
os.makedirs(FIG, exist_ok=True)

B_GRID = [0, 1, 2, 4, 6, 10, 18]


def load(name):
    p = os.path.join(RES, name + ".json")
    return json.load(open(p)) if os.path.exists(p) else None


def svg_bars(path, title, labels, series, ylabel):
    """series: list of (name, values). Simple grouped-bar SVG."""
    W, H, pad = 640, 380, 60
    n, m = len(labels), len(series)
    gw = (W - 2 * pad) / max(n, 1)
    bw = gw / (m + 1)
    vmax = max(max(v) for _, v in series) or 1.0
    vmax = vmax * 1.1
    colors = ["#2563eb", "#dc2626", "#16a34a", "#d97706", "#7c3aed"]
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}">']
    s.append(f'<text x="{W//2}" y="28" text-anchor="middle" font-size="16" '
             f'font-family="sans-serif">{title}</text>')
    for i in range(11):
        y = H - pad - i * (H - 2 * pad) / 10
        v = vmax * i / 10
        s.append(f'<line x1="{pad}" y1="{y}" x2="{W-pad}" y2="{y}" stroke="#e5e7eb"/>')
        s.append(f'<text x="{pad-6}" y="{y+4}" text-anchor="end" font-size="10" '
                 f'font-family="sans-serif">{v:.2f}</text>')
    for gi, lab in enumerate(labels):
        for si, (nm, vals) in enumerate(series):
            v = vals[gi]
            h = (H - 2 * pad) * v / vmax
            x = pad + gi * gw + si * bw + bw * 0.25
            y = H - pad - h
            s.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw*0.9:.1f}" '
                     f'height="{h:.1f}" fill="{colors[si % len(colors)]}">'
                     f'<title>{nm}: {v:.4f}</title></rect>')
        s.append(f'<text x="{pad + gi*gw + gw/2:.1f}" y="{H-pad+16}" text-anchor="middle" '
                 f'font-size="10" font-family="sans-serif">{lab}</text>')
    for si, (nm, _) in enumerate(series):
        s.append(f'<rect x="{pad + si*130}" y="{H-18}" width="12" height="12" '
                 f'fill="{colors[si % len(colors)]}"/>')
        s.append(f'<text x="{pad + si*130 + 16}" y="{H-8}" font-size="11" '
                 f'font-family="sans-serif">{nm}</text>')
    s.append(f'<text x="16" y="{H//2}" text-anchor="middle" font-size="11" '
             f'font-family="sans-serif" transform="rotate(-90 16 {H//2})">{ylabel}</text>')
    s.append('</svg>')
    open(path, "w").write("\n".join(s))


def main():
    rows = []
    for f in sorted(os.listdir(RES)):
        if f.endswith(".json") and f != "summary.json":
            d = json.load(open(os.path.join(RES, f)))
            rows.append(d)
    # main table: method/attack/frac -> accuracy, cert@B, asr
    print(f"{'config':32s} {'acc':>7s} " + " ".join(f"cB{b:>2d}" for b in [0, 2, 6, 10, 18])
          + f" {'asr':>6s} {'t':>3s} {'wall_m':>6s}")
    for d in rows:
        c = d["config"]
        cw = d.get("cert_worst", {})
        line = (f"{c['name']:32s} {d.get('accuracy', float('nan')):7.4f} " +
                " ".join(f"{cw.get(str(b), float('nan')):7.4f}" for b in [0, 2, 6, 10, 18]) +
                f" {d.get('backdoor_asr', float('nan')):6.4f} "
                f"{d.get('t_effective', -1):3d} {d['wall_seconds']/60:6.1f}")
        print(line)
    # figure 1: certified accuracy vs B for key configs
    key = ["cd_clean", "cd_logit_targeted_30", "cd_labelflip_30", "cao_clean", "cao_labelflip_20"]
    labels = [f"B={b}" for b in B_GRID]
    series = []
    for k in key:
        d = load(k)
        if d and "cert_worst" in d:
            series.append((k, [d["cert_worst"][str(b)] for b in B_GRID]))
    if series:
        svg_bars(os.path.join(FIG, "certified_accuracy.svg"),
                 "Certified accuracy vs # malicious clients (worst-case)",
                 labels, series, "certified accuracy")
        print("wrote figures/certified_accuracy.svg")
    # figure 2: clean accuracy under attacks
    atk_labels, cd_vals, fdf_vals, fr_vals = [], [], [], []
    for atk in ["labelflip", "logit_invert", "logit_targeted", "logit_random", "backdoor"]:
        d = load(f"cd_{atk}_30")
        if d:
            atk_labels.append(atk.replace("logit_", ""))
            cd_vals.append(d["accuracy"])
            m = load(f"feddf_mean_{atk}_20")
            fdf_vals.append(m["accuracy"] if m else 0)
            r = load(f"fedrad_{atk}_20")
            fr_vals.append(r["accuracy"] if r else 0)
    base = load("cd_clean")
    if base and atk_labels:
        svg_bars(os.path.join(FIG, "attack_accuracy.svg"),
                 "Test accuracy under 30% Byzantine clients (heterogeneous)",
                 ["clean"] + atk_labels,
                 [("certdistill", [base["accuracy"]] + cd_vals)],
                 "accuracy")
        print("wrote figures/attack_accuracy.svg")
    json.dump([{"name": d["config"]["name"], "accuracy": d.get("accuracy"),
                "cert_worst": d.get("cert_worst"), "backdoor_asr": d.get("backdoor_asr"),
                "t_effective": d.get("t_effective"),
                "vote_flip_vs_clean": d.get("vote_flip_vs_clean"),
                "wall_seconds": d["wall_seconds"]} for d in rows],
              open(os.path.join(RES, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
