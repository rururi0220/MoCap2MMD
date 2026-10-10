"""PyWebView API bridge for MoCap2MMD."""
from __future__ import annotations

import base64
import os
import io
import webview

from backend.parser.bvh_parser import parse_bvh
from backend.parser.fbx_parser import parse_fbx
from backend.parser.pmx_parser import load_pmx
from backend.retarget.retargeter import Retargeter, RetargetConfig
from backend.retarget.bone_mapping import map_source, map_target
from backend.exporter.vmd_exporter import write_vmd, VMDMotion


class MoCapAPI:
    def __init__(self):
        self._window: webview.Window | None = None
        self._cached_source = None
        self._cached_target = None
        self._last_vmd: VMDMotion | None = None

    def set_window(self, window: webview.Window):
        self._window = window

    def select_source_file(self) -> dict | None:
        """Open file dialog for source motion (.fbx, .bvh)."""
        if not self._window:
            return None
        file_types = ("Motion Files (*.fbx;*.bvh)", "FBX Files (*.fbx)", "BVH Files (*.bvh)", "All Files (*.*)")
        dialog_type = getattr(webview, "FileDialog", None)
        open_dlg = dialog_type.OPEN if dialog_type else webview.OPEN_DIALOG
        res = self._window.create_file_dialog(open_dlg, allow_multiple=False, file_types=file_types)
        if not res or len(res) == 0:
            return None
        path = res[0]
        return self.inspect_source(path)

    def select_pmx_file(self) -> dict | None:
        """Open file dialog for target PMX model (.pmx)."""
        if not self._window:
            return None
        file_types = ("MMD PMX Model (*.pmx)", "All Files (*.*)")
        dialog_type = getattr(webview, "FileDialog", None)
        open_dlg = dialog_type.OPEN if dialog_type else webview.OPEN_DIALOG
        res = self._window.create_file_dialog(open_dlg, allow_multiple=False, file_types=file_types)
        if not res or len(res) == 0:
            return None
        path = res[0]
        return self.inspect_pmx(path)

    def inspect_source(self, path: str) -> dict:
        """Parse and inspect source motion."""
        try:
            ext = os.path.splitext(path)[1].lower()
            if ext == ".bvh":
                src = parse_bvh(path)
            elif ext == ".fbx":
                src = parse_fbx(path)
            else:
                return {"success": False, "error": f"Unsupported format: {ext}"}

            self._cached_source = (path, src)
            s_map = map_source(src)

            return {
                "success": True,
                "path": path,
                "filename": os.path.basename(path),
                "format": src.format.upper(),
                "num_joints": src.num_joints,
                "num_frames": src.num_frames,
                "fps": round(src.fps, 2),
                "duration_sec": round(src.num_frames / max(src.fps, 1), 2),
                "preset": s_map.preset,
                "joints": src.names,
                "mapping": {k: src.names[v] for k, v in s_map.bones.items()},
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def inspect_pmx(self, path: str) -> dict:
        """Parse and inspect PMX model."""
        try:
            tgt = load_pmx(path)
            self._cached_target = (path, tgt)
            t_map = map_target(tgt)

            # Read PMX binary into base64 so frontend Three.js can load it via Blob
            with open(path, "rb") as f:
                pmx_data = f.read()
            pmx_b64 = base64.b64encode(pmx_data).decode("ascii")

            # Load textures as Base64 data URLs for frontend Three.js viewer
            pmx_dir = os.path.dirname(os.path.abspath(path))
            textures_map = {}
            for t_name in tgt.textures:
                norm = t_name.replace("\\", "/")
                cand_paths = [
                    os.path.join(pmx_dir, t_name),
                    os.path.join(pmx_dir, norm),
                    os.path.join(pmx_dir, os.path.basename(t_name)),
                    os.path.join(pmx_dir, "tex", os.path.basename(t_name)),
                ]
                full_path = None
                for cp in cand_paths:
                    if os.path.exists(cp) and os.path.isfile(cp):
                        full_path = cp
                        break
                if full_path:
                    try:
                        ext = os.path.splitext(full_path)[1].lower()
                        mime = "image/png"
                        if ext in (".jpg", ".jpeg"):
                            mime = "image/jpeg"
                        elif ext == ".bmp":
                            mime = "image/bmp"
                        elif ext == ".tga":
                            mime = "image/x-tga"
                        with open(full_path, "rb") as tf:
                            t_bytes = tf.read()
                        b64 = base64.b64encode(t_bytes).decode("ascii")
                        data_url = f"data:{mime};base64,{b64}"
                        textures_map[t_name] = data_url
                        textures_map[norm] = data_url
                        textures_map[os.path.basename(t_name)] = data_url
                    except Exception:
                        pass

            return {
                "success": True,
                "path": path,
                "filename": os.path.basename(path),
                "model_name": tgt.name,
                "model_name_en": tgt.name_en,
                "num_bones": len(tgt.bones),
                "bones": [b.name for b in tgt.bones],
                "has_foot_ik": t_map.get("legIK.L") >= 0 and t_map.get("legIK.R") >= 0,
                "pmx_base64": pmx_b64,
                "base_dir": pmx_dir.replace("\\", "/") + "/",
                "textures": textures_map,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def execute_retarget(self, options: dict) -> dict:
        """Perform retargeting and return VMD base64 for preview."""
        try:
            if not self._cached_source or not self._cached_target:
                return {"success": False, "error": "Source motion or PMX model not loaded"}

            path_src, _ = self._cached_source
            path_tgt, _ = self._cached_target

            # Always reload fresh source motion from original file
            ext = os.path.splitext(path_src)[1].lower()
            if ext == ".bvh":
                src = parse_bvh(path_src)
            elif ext == ".fbx":
                src = parse_fbx(path_src)
            else:
                return {"success": False, "error": f"Unsupported format: {ext}"}

            # Always reload fresh PMX target model from original file
            tgt = load_pmx(path_tgt)

            cfg = RetargetConfig(
                enable_foot_ik=options.get("enable_foot_ik", True),
                enable_twist_bones=options.get("enable_twist_bones", True),
                enable_fingers=options.get("enable_fingers", True),
                scale_multiplier=float(options.get("scale_multiplier", 1.0)),
                auto_detect_a_pose=options.get("auto_detect_a_pose", True),
                target_fps=float(options.get("target_fps", 30.0)),
                smooth_sigma=float(options.get("smooth_sigma", 1.0)),
                root_rotation_y=float(options.get("root_rotation_y", 0.0)),
                overrides=options.get("overrides", {}),
            )

            retargeter = Retargeter(src, tgt, cfg)
            vmd = retargeter.retarget()
            self._last_vmd = vmd

            # Export to in-memory bytes
            buf = io.BytesIO()
            tmp_path = "_temp_preview.vmd"
            write_vmd(tmp_path, vmd)
            with open(tmp_path, "rb") as f:
                vmd_data = f.read()
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

            vmd_b64 = base64.b64encode(vmd_data).decode("ascii")

            return {
                "success": True,
                "num_tracks": len(vmd.bones),
                "num_keys": vmd.num_bone_keys,
                "total_frames": int(vmd.bones[0].frames[-1] + 1) if vmd.bones else 0,
                "vmd_base64": vmd_b64,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def save_vmd_file(self, default_name: str = "motion.vmd") -> dict:
        """Prompt user for destination and save the last converted VMD."""
        if not self._window:
            return {"success": False, "error": "No GUI window available"}
        if not self._last_vmd:
            return {"success": False, "error": "No motion converted yet"}

        file_types = ("Vocaloid Motion Data (*.vmd)", "All Files (*.*)")
        dialog_type = getattr(webview, "FileDialog", None)
        save_dlg = dialog_type.SAVE if dialog_type else webview.SAVE_DIALOG
        res = self._window.create_file_dialog(
            save_dlg,
            save_filename=default_name,
            file_types=file_types,
        )
        if not res:
            return {"cancelled": True}

        out_path = res if isinstance(res, str) else res[0]
        if not out_path.lower().endswith(".vmd"):
            out_path += ".vmd"

        try:
            write_vmd(out_path, self._last_vmd)
            return {"success": True, "path": out_path}
        except Exception as e:
            return {"success": False, "error": str(e)}
