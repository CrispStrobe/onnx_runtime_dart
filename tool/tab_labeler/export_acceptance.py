# Export held-out (val guitarist = 05) GuitarSet songs as column sequences for
# the Dart-side acceptance test: heuristic arrangeTab vs the model, each scored
# against the human fingering. JSON: [{id, columns:[[midi,...]], human:[[[str,
# fret],...]]}]. String index 0 = high e (the CometBeat decoder order).
import glob
import json
import os
import sys

import jams

ANNO = sys.argv[1] if len(sys.argv) > 1 else "anno"
OUT = sys.argv[2] if len(sys.argv) > 2 else "acceptance.json"
OPEN = [64, 59, 55, 50, 45, 40]
MAXFRET = 19
ONSET_TOL = 0.045


def columns_for(jam):
    events = []
    for gs_str, anno in enumerate(jam.search(namespace="note_midi")):
        ds = 5 - gs_str
        for obs in anno.data:
            midi = int(round(obs.value))
            fret = midi - OPEN[ds]
            if 0 <= fret <= MAXFRET:
                events.append((float(obs.time), ds, fret, midi))
    events.sort(key=lambda e: e[0])
    cols = []
    i = 0
    while i < len(events):
        t0 = events[i][0]
        placed, pitches = [], []
        j = i
        seen = set()
        while j < len(events) and events[j][0] - t0 <= ONSET_TOL:
            _, ds, fret, midi = events[j]
            if ds not in seen:
                seen.add(ds)
                placed.append([ds, fret])
                pitches.append(midi)
            j += 1
        cols.append((pitches, placed))
        i = j
    return cols


out = []
for f in sorted(glob.glob(os.path.join(ANNO, "*.jams"))):
    if os.path.basename(f)[:2] != "05":
        continue
    cols = columns_for(jams.load(f))
    if not cols:
        continue
    out.append({
        "id": os.path.basename(f)[:-5],
        "columns": [p for (p, _) in cols],
        "human": [pl for (_, pl) in cols],
    })
json.dump(out, open(OUT, "w"))
n = sum(len(s["columns"]) for s in out)
print(f"{len(out)} held-out songs, {n} columns → {OUT}")
