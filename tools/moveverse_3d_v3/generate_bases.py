#!/usr/bin/env python3
import argparse, json, os, sys, traceback
from pathlib import Path

import mhcore


def first_asset(h, kind, keywords=()):
    try:
        assets = list(h.list_available(kind))
    except Exception:
        return None
    if not assets:
        return None
    lower = [(p, Path(p).name.lower()) for p in assets]
    for kw in keywords:
        for p, name in lower:
            if kw.lower() in name:
                return p
    return assets[0]


def maybe_apply(h, key, value):
    try:
        mods = set(h.list_modifiers())
        if key in mods:
            h.apply_modifier(key, float(value))
            return True
    except Exception:
        pass
    return False


def build_one(spec, outdir):
    cid = int(spec["id"])
    tag = f"coach_{cid:02d}_{spec['name'].lower()}"
    dst = outdir / tag
    dst.mkdir(parents=True, exist_ok=True)

    h = mhcore.new_human()
    b = spec["body"]
    gender = 0.0 if spec["gender"] == "female" else 1.0
    h.set_gender(gender)
    h.set_age(0.48)
    h.set_weight(float(b.get("weight", 0.5)))
    h.set_muscle(float(b.get("muscle", 0.5)))
    h.set_height(float(b.get("height", 0.5)))

    # Character-specific facial/body variation. These are deliberately subtle;
    # the final visual identity comes from Blender hair/outfit/accessories.
    fid = cid % 6
    modifier_sets = [
        ("head/head-oval", -0.22 + 0.09 * fid),
        ("nose/nose-scale-vert-decr|incr", -0.12 + 0.05 * (cid % 5)),
        ("mouth/mouth-scale-horiz-decr|incr", -0.10 + 0.04 * (cid % 6)),
        ("chin/chin-width-decr|incr", -0.10 + 0.04 * ((cid + 2) % 6)),
    ]
    for k, v in modifier_sets:
        maybe_apply(h, k, v)

    # Deform skeleton is clean enough for a game export and is preserved by FBX.
    try:
        h.set_skeleton(mhcore.data_path("rigs/default.mhskel"))
    except Exception as e:
        print("WARN skeleton:", e)

    # Keep anatomy assets but no MakeHuman clothes/hair: all visible styling is
    # built specifically for the 24 MoveVerse references in Blender.
    for kind, fn, kws in [
        ("eyes", h.equip_eyes, ("high", "default")),
        ("eyebrows", h.equip_eyebrows, ("default",)),
        ("eyelashes", h.equip_eyelashes, ("default",)),
        ("teeth", h.equip_teeth, ("default",)),
        ("tongue", h.equip_tongue, ("default",)),
    ]:
        asset = first_asset(h, kind, kws)
        if asset:
            try:
                fn(asset)
            except Exception as e:
                print(f"WARN {kind}: {e}")

    skin = first_asset(h, "skins", ("young", "middle", "caucasian"))
    if skin:
        try:
            h.set_skin(skin)
        except Exception:
            pass

    mhm = h.save_mhm(str(dst / f"{tag}.mhm"))
    fbx = h.export(str(dst / f"{tag}.fbx"), format="fbx", rig="deform", feet_on_ground=True)
    print("BUILT", tag, mhm, fbx)
    return {"id": cid, "tag": tag, "mhm": str(mhm), "fbx": str(fbx)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    specs = json.load(open(args.specs, "r", encoding="utf-8"))["characters"]
    if args.limit:
        specs = specs[:args.limit]
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for s in specs:
        try:
            manifest.append(build_one(s, outdir))
        except Exception as e:
            traceback.print_exc()
            raise RuntimeError(f"Failed character {s['id']}: {e}")
    (outdir / "base_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
