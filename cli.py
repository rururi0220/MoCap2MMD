"""CLI interface for MoCap2MMD."""
from __future__ import annotations

import argparse
import os
import sys

from backend.parser.bvh_parser import parse_bvh
from backend.parser.fbx_parser import parse_fbx
from backend.parser.pmx_parser import load_pmx
from backend.retarget.retargeter import Retargeter, RetargetConfig
from backend.exporter.vmd_exporter import write_vmd


def main():
    parser = argparse.ArgumentParser(description="MoCap2MMD: Convert FBX/BVH motion to MMD VMD")
    parser.add_argument("-s", "--source", required=True, help="Path to source motion (.fbx or .bvh)")
    parser.add_argument("-t", "--target", required=True, help="Path to target PMX model (.pmx)")
    parser.add_argument("-o", "--output", required=True, help="Path to output VMD file (.vmd)")
    parser.add_argument("--scale", type=float, default=1.0, help="Scale multiplier for root translation")
    parser.add_argument("--no-ik", action="store_true", help="Disable foot IK generation (use FK instead)")
    parser.add_argument("--no-twist", action="store_true", help="Disable arm/hand twist bone separation")
    parser.add_argument("--no-fingers", action="store_true", help="Disable finger retargeting")
    parser.add_argument("--fps", type=float, default=30.0, help="Output framerate (default: 30.0)")
    parser.add_argument("--root-rot-y", type=float, default=0.0, help="Rotate 全ての親 (root) bone around Y axis in degrees (default: 0.0)")

    args = parser.parse_args()

    ext = os.path.splitext(args.source)[1].lower()
    print(f"[1/4] Loading source motion: {args.source}")
    if ext == ".bvh":
        src = parse_bvh(args.source)
    elif ext == ".fbx":
        src = parse_fbx(args.source)
    else:
        print(f"Error: Unsupported motion format '{ext}'. Must be .fbx or .bvh", file=sys.stderr)
        sys.exit(1)

    print(f"      Parsed {src.num_joints} joints, {src.num_frames} frames @ {src.fps:.2f} fps")

    print(f"[2/4] Loading PMX model: {args.target}")
    tgt = load_pmx(args.target)
    print(f"      Model name: {tgt.name} ({len(tgt.bones)} bones)")

    print(f"[3/4] Retargeting motion...")
    cfg = RetargetConfig(
        enable_foot_ik=not args.no_ik,
        enable_twist_bones=not args.no_twist,
        enable_fingers=not args.no_fingers,
        scale_multiplier=args.scale,
        target_fps=args.fps,
        root_rotation_y=args.root_rot_y,
    )
    retargeter = Retargeter(src, tgt, cfg)
    vmd = retargeter.retarget()
    print(f"      Generated {len(vmd.bones)} bone tracks, {vmd.num_bone_keys} total keyframes")

    print(f"[4/4] Writing VMD: {args.output}")
    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    write_vmd(args.output, vmd)
    print(f"Done! Output saved to: {args.output}")


if __name__ == "__main__":
    main()
