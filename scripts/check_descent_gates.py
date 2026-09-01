"""Run find_descents' per-chain gate block for real, on synthetic chains.

Extracts the loop body from the source so it exercises the SHIPPED code,
not a paraphrase — which is exactly what the last round missed.
"""
import ast, logging, math, os, secrets, sys, threading, time
import numpy as np

SRC = "backend/app/services/debug3.py"
src = open(SRC).read()
tree = ast.parse(src)

mod = ast.parse(src)
# Everything at module level except the imports we cannot satisfy here.
keep = []
for n in mod.body:
    if isinstance(n, (ast.Import, ast.ImportFrom)):
        continue
    keep.append(n)
g = {"np": np, "math": math, "log": type("L", (), {
    "debug": lambda *a, **k: None, "info": lambda *a, **k: None,
    "warning": lambda *a, **k: None})(), "HAS_CV": False, "cv2": None,
    "Path": __import__("pathlib").Path, "Iterable": list,
    "logging": logging, "os": os, "time": time,
    "secrets": secrets, "threading": threading}
try:
    exec(compile(ast.Module(body=keep, type_ignores=[]), SRC, "exec"), g)
except Exception as exc:
    print(f"module-level exec failed: {type(exc).__name__}: {exc}")
    sys.exit(1)

g["np"] = np
g["HAS_NP"] = True

fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
          and n.name == "find_descents")
# The gate block: from `_why: list = []` to just before the row append.
body = fn.body
def find_loop(nodes):
    for n in nodes:
        if isinstance(n, ast.For):
            for sub in ast.walk(n):
                if (isinstance(sub, ast.AnnAssign)
                        and getattr(sub.target, "id", "") == "_why"):
                    return n
        if hasattr(n, "body"):
            r = find_loop(getattr(n, "body", []))
            if r:
                return r
        if hasattr(n, "orelse"):
            r = find_loop(getattr(n, "orelse", []))
            if r:
                return r
    return None
loop = find_loop(body)
stmts = []
started = False
for st in loop.body:
    if (isinstance(st, ast.AnnAssign) and getattr(st.target, "id", "") == "_why"):
        started = True
    if started:
        # Stop at the row dict: everything before it is the gates.
        if (isinstance(st, ast.Assign)
                and getattr(st.targets[0], "id", "") == "_entry"):
            break
        stmts.append(st)
print(f"extracted {len(stmts)} statements of the real gate block\n")

def run(pts, fps=30.0, frame_h=720):
    ns = dict(g)
    ns.update({"pts": pts, "_fps": fps, "frame_h": frame_h,
               "min_drop": 0.05 * frame_h, "min_points": g["MIN_DESCENT_POINTS"],
               "rate_lo": g["DESCENT_RATE_LO"], "rate_hi": g["DESCENT_RATE_HI"],
               "max_bend_px": 1.0, "_rej": lambda *a, **k: None,
               "frame_w": 1280, "edge_px": 0.04 * 720,
               "land": pts[-1], "kept": pts,
               "_by_frame": {}, "dets": [], "tk": {},
               "considered": [], "events": [],
               "f_hi": pts[-1]["frame"] + 200,
               "f_lo": max(0, pts[0]["frame"] - 200),
               "_fps": fps, "merge_sec": 1.0})
    exec(compile(ast.Module(body=stmts, type_ignores=[]), "<gates>", "exec"), ns)
    return ns

def chain(ys, xs, f0=1000):
    return [{"frame": f0 + i, "x": xs[i], "y": ys[i]} for i in range(len(ys))]

YS = [100, 112, 126, 142, 160, 180, 202, 226, 252]
def raw(rows, f0=0):
    return [{"frame": f, "x": x, "y": y} for f, x, y in rows]

CASES = [
    # The chain the operator pointed at: eight points of a real descent
    # plus one that arrives 19.7px off their line. 30fps green.
    # The chain the operator pointed at, exactly as the table lists it:
    # eight points of a real descent (including a duplicate sighting at
    # f2293/f2294, both y=306), a ninth 19.7px off their line, and the
    # bounce-and-roll the tracker carried on with.
    ("upload's chain #3, the real one",
     raw([(2290,535,252),(2291,534,270),(2292,533,288),(2293,532,306),
          (2294,532,306),(2296,532,346),(2297,530,365),(2298,530,385),
          (2302,546,483),(2303,566,478),(2305,590,474),(2306,614,476),
          (2307,640,479),(2308,668,481)]), True),
    # ...and the same eight points with a ninth that is genuinely on the
    # line: nothing should be trimmed and it should still pass.
    ("same chain, touchdown on the line",
     raw([(2290,535,252),(2291,534,270),(2292,533,288),(2293,532,306),
          (2294,532,326),(2296,532,346),(2297,530,365),(2298,530,385),
          (2302,526,483)]), True),
    # A ball that really does stall mid-fall must still be refused —
    # the duplicate collapse must not rescue it.
    ("a fall that genuinely stalls",
     raw([(0,500,100),(1,500,118),(2,500,136),(3,500,140),(4,500,144),
          (5,500,162),(6,500,180),(7,500,198)]), False),
    ("straight fall, touchdown kicks 14px",
     chain(YS, [500,502,504,506,508,510,512,514,530]), True),
    ("straight fall, clean touchdown",
     chain(YS, [500,502,504,506,508,510,512,514,516]), True),
    ("bent the whole way down",
     chain(YS, [500,512,528,548,572,600,632,668,708]), False),
    ("scattered speckle",
     chain(YS, [500,540,505,560,495,545,510,570,500]), False),
    # At the floor nothing is dropped -- taking one from four leaves
    # three, which cannot be judged -- so a 4-point chain IS judged on
    # its touchdown, and a touchdown that kicks sideways refuses it.
    ("4-point chain, touchdown kicks 16px",
     chain([100,118,140,166], [500,502,504,520]), False),
    ("4-point chain, straight through",
     chain([100,118,140,166], [500,502,504,506]), True),
]
ok = 0
for name, pts, want in CASES:
    try:
        ns = run(pts)
    except Exception as exc:
        print(f"{name:38s} CRASHED: {type(exc).__name__}: {exc}")
        continue
    why = ns["_why"]
    got = not why
    ok += (got == want)
    _land = ns.get("land", {})
    print(f"{name:38s} bend {ns['bend']:5.2f} (whole {ns['bend_all']:5.2f})"
          f"  lands f{_land.get('frame')} "
          f"({_land.get('x')},{_land.get('y')})  -> "
          + (", ".join(why) if why else "accepted")
          + ("" if got == want else "   <-- MISMATCH"))
print(f"\n{ok}/{len(CASES)} as intended")
