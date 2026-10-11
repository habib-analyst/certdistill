"""Architecture diagram for certdistill README (hand-built SVG)."""
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(BASE, "results", "figures")
os.makedirs(FIG, exist_ok=True)

W, H = 900, 420
s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="sans-serif">']
s.append('<rect width="100%" height="100%" fill="#ffffff"/>')

def box(x, y, w, h, fill, label, sub=None):
    s.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{fill}" stroke="#1f2937" stroke-width="1.5"/>')
    s.append(f'<text x="{x+w/2}" y="{y+24}" text-anchor="middle" font-size="13" font-weight="bold" fill="#111827">{label}</text>')
    if sub:
        for i, line in enumerate(sub):
            s.append(f'<text x="{x+w/2}" y="{y+44+i*17}" text-anchor="middle" font-size="11" fill="#374151">{line}</text>')

def arrow(x1, y1, x2, y2, label=None):
    s.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#4b5563" stroke-width="2" marker-end="url(#ah)"/>')
    if label:
        s.append(f'<text x="{(x1+x2)/2}" y="{(y1+y2)/2 - 6}" text-anchor="middle" font-size="11" fill="#4b5563">{label}</text>')

s.append('<defs><marker id="ah" markerWidth="10" markerHeight="7" refX="10" refY="3.5" orient="auto">'
         '<polygon points="0 0, 10 3.5, 0 7" fill="#4b5563"/></marker></defs>')
s.append(f'<text x="{W/2}" y="30" text-anchor="middle" font-size="18" font-weight="bold" fill="#111827">CertDistill: certified robustness for heterogeneous federated learning</text>')

# clients
box(30, 70, 150, 110, "#dbeafe", "Client pool", ["60 clients", "3 architectures", "CNN-S · CNN-D · MLP"])
# subsets
box(240, 70, 150, 110, "#fef3c7", "K=20 subsets", ["disjoint triples", "1 arch each", "stratified"])
# local train
box(450, 70, 170, 110, "#e0e7ff", "Local training", ["E=3 epochs", "own architecture", "Byzantines may", "poison data"])
# logits + trim
box(680, 70, 190, 110, "#fce7f3", "Robust distillation", ["logits on probe set", "α-trimmed mean", "per (sample, class)"])
# students
box(240, 230, 150, 110, "#d1fae5", "K students", ["distilled", "fixed arch", "warm-started"])
# vote
box(450, 230, 170, 110, "#ede9fe", "Majority vote", ["n_a vs n_b", "margin = n_a − n_b"])
# certificate
box(680, 230, 190, 110, "#ffedd5", "Certificate", ["margin > 2t  ⇒", "provably robust", "t ≤ B (tightened)"])

arrow(180, 125, 240, 125)
arrow(390, 125, 450, 125)
arrow(620, 125, 680, 125)
arrow(755, 180, 755, 230)
arrow(315, 180, 315, 230)
arrow(390, 285, 450, 285)
arrow(620, 285, 680, 285)
s.append('<text x="90" y="365" font-size="12" fill="#374151">Lemma 1: below-capacity Byzantines can perturb, not control, distilled targets.</text>')
s.append('<text x="90" y="385" font-size="12" fill="#374151">Theorem: vote margin &gt; 2t certifies the prediction — architecture-agnostic, tighter than Cao when t &lt; B.</text>')
s.append('</svg>')
open(os.path.join(FIG, "architecture.svg"), "w").write("\n".join(s))
print("wrote architecture.svg")
