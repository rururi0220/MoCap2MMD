"""Test and CLI entrypoint to verify MoCap2MMD conversion pipeline."""
import os
import sys
import numpy as np

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.parser.bvh_parser import parse_bvh
from backend.parser.fbx_parser import parse_fbx
from backend.parser.pmx_parser import load_pmx, PMXModel, PMXBone
from backend.retarget.retargeter import Retargeter, RetargetConfig
from backend.exporter.vmd_exporter import write_vmd, read_vmd


def create_sample_bvh(path: str):
    """Generate a minimal sample BVH walking/dancing motion for testing."""
    content = """HIERARCHY
ROOT Hips
{
  OFFSET 0.00 0.00 0.00
  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
  JOINT Spine
  {
    OFFSET 0.00 15.00 0.00
    CHANNELS 3 Zrotation Xrotation Yrotation
    JOINT Neck
    {
      OFFSET 0.00 20.00 0.00
      CHANNELS 3 Zrotation Xrotation Yrotation
      JOINT Head
      {
        OFFSET 0.00 10.00 0.00
        CHANNELS 3 Zrotation Xrotation Yrotation
        End Site
        {
          OFFSET 0.00 5.00 0.00
        }
      }
    }
    JOINT LeftShoulder
    {
      OFFSET 5.00 18.00 0.00
      CHANNELS 3 Zrotation Xrotation Yrotation
      JOINT LeftArm
      {
        OFFSET 15.00 0.00 0.00
        CHANNELS 3 Zrotation Xrotation Yrotation
        JOINT LeftForeArm
        {
          OFFSET 20.00 0.00 0.00
          CHANNELS 3 Zrotation Xrotation Yrotation
          JOINT LeftHand
          {
            OFFSET 15.00 0.00 0.00
            CHANNELS 3 Zrotation Xrotation Yrotation
            End Site
            {
              OFFSET 5.00 0.00 0.00
            }
          }
        }
      }
    }
    JOINT RightShoulder
    {
      OFFSET -5.00 18.00 0.00
      CHANNELS 3 Zrotation Xrotation Yrotation
      JOINT RightArm
      {
        OFFSET -15.00 0.00 0.00
        CHANNELS 3 Zrotation Xrotation Yrotation
        JOINT RightForeArm
        {
          OFFSET -20.00 0.00 0.00
          CHANNELS 3 Zrotation Xrotation Yrotation
          JOINT RightHand
          {
            OFFSET -15.00 0.00 0.00
            CHANNELS 3 Zrotation Xrotation Yrotation
            End Site
            {
              OFFSET -5.00 0.00 0.00
            }
          }
        }
      }
    }
  }
  JOINT LeftUpLeg
  {
    OFFSET 10.00 -5.00 0.00
    CHANNELS 3 Zrotation Xrotation Yrotation
    JOINT LeftLeg
    {
      OFFSET 0.00 -35.00 0.00
      CHANNELS 3 Zrotation Xrotation Yrotation
      JOINT LeftFoot
      {
        OFFSET 0.00 -35.00 0.00
        CHANNELS 3 Zrotation Xrotation Yrotation
        End Site
        {
          OFFSET 0.00 0.00 10.00
        }
      }
    }
  }
  JOINT RightUpLeg
  {
    OFFSET -10.00 -5.00 0.00
    CHANNELS 3 Zrotation Xrotation Yrotation
    JOINT RightLeg
    {
      OFFSET 0.00 -35.00 0.00
      CHANNELS 3 Zrotation Xrotation Yrotation
      JOINT RightFoot
      {
        OFFSET 0.00 -35.00 0.00
        CHANNELS 3 Zrotation Xrotation Yrotation
        End Site
        {
          OFFSET 0.00 0.00 10.00
        }
      }
    }
  }
}
MOTION
Frames: 3
Frame Time: 0.0333333
0.0 80.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
0.0 81.0 1.0 5.0 0.0 0.0 2.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 5.0 0.0 0.0 10.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 -5.0 0.0 0.0 -10.0 0.0 0.0 0.0 0.0 5.0 0.0 0.0 10.0 0.0 0.0 0.0 0.0 0.0 -5.0 0.0 0.0 -10.0 0.0 0.0 0.0 0.0 0.0
0.0 82.0 2.0 10.0 0.0 0.0 4.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 10.0 0.0 0.0 20.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 -10.0 0.0 0.0 -20.0 0.0 0.0 0.0 0.0 10.0 0.0 0.0 20.0 0.0 0.0 0.0 0.0 0.0 -10.0 0.0 0.0 -20.0 0.0 0.0 0.0 0.0 0.0
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def create_sample_pmx() -> PMXModel:
    """Create a mock PMX model with standard MMD bones."""
    bones = [
        PMXBone(0, "全ての親", "Root", np.array([0.0, 0.0, 0.0]), -1, 0, 0x0002 | 0x0004),
        PMXBone(1, "センター", "Center", np.array([0.0, 10.0, 0.0]), 0, 0, 0x0002 | 0x0004),
        PMXBone(2, "下半身", "LowerBody", np.array([0.0, 10.0, 0.0]), 1, 0, 0x0002),
        PMXBone(3, "上半身", "UpperBody", np.array([0.0, 12.0, 0.0]), 1, 0, 0x0002),
        PMXBone(4, "上半身2", "UpperBody2", np.array([0.0, 15.0, 0.0]), 3, 0, 0x0002),
        PMXBone(5, "首", "Neck", np.array([0.0, 18.0, 0.0]), 4, 0, 0x0002),
        PMXBone(6, "頭", "Head", np.array([0.0, 20.0, 0.0]), 5, 0, 0x0002),
        PMXBone(7, "左肩", "LeftShoulder", np.array([1.0, 17.5, 0.0]), 4, 0, 0x0002),
        PMXBone(8, "左腕", "LeftArm", np.array([2.5, 17.0, 0.0]), 7, 0, 0x0002),
        PMXBone(9, "左ひじ", "LeftElbow", np.array([5.0, 17.0, 0.0]), 8, 0, 0x0002),
        PMXBone(10, "左手首", "LeftWrist", np.array([7.5, 17.0, 0.0]), 9, 0, 0x0002),
        PMXBone(11, "右肩", "RightShoulder", np.array([-1.0, 17.5, 0.0]), 4, 0, 0x0002),
        PMXBone(12, "右腕", "RightArm", np.array([-2.5, 17.0, 0.0]), 11, 0, 0x0002),
        PMXBone(13, "右ひじ", "RightElbow", np.array([-5.0, 17.0, 0.0]), 12, 0, 0x0002),
        PMXBone(14, "右手首", "RightWrist", np.array([-7.5, 17.0, 0.0]), 13, 0, 0x0002),
        PMXBone(15, "左足", "LeftLeg", np.array([1.5, 9.5, 0.0]), 2, 0, 0x0002),
        PMXBone(16, "左ひざ", "LeftKnee", np.array([1.5, 5.0, 0.0]), 15, 0, 0x0002),
        PMXBone(17, "左足首", "LeftAnkle", np.array([1.5, 1.0, 0.0]), 16, 0, 0x0002),
        PMXBone(18, "右足", "RightLeg", np.array([-1.5, 9.5, 0.0]), 2, 0, 0x0002),
        PMXBone(19, "右ひざ", "RightKnee", np.array([-1.5, 5.0, 0.0]), 18, 0, 0x0002),
        PMXBone(20, "右足首", "RightAnkle", np.array([-1.5, 1.0, 0.0]), 19, 0, 0x0002),
        # IK Bones
        PMXBone(21, "左足ＩＫ", "LeftLegIK", np.array([1.5, 1.0, 0.0]), 0, 0, 0x0002 | 0x0004 | 0x0020, ik_target=17),
        PMXBone(22, "右足ＩＫ", "RightLegIK", np.array([-1.5, 1.0, 0.0]), 0, 0, 0x0002 | 0x0004 | 0x0020, ik_target=20),
    ]
    return PMXModel(name="SampleModel", name_en="SampleModel", version=2.0, bones=bones)


def test_pipeline():
    test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_temp")
    os.makedirs(test_dir, exist_ok=True)
    bvh_path = os.path.join(test_dir, "test.bvh")
    vmd_path = os.path.join(test_dir, "output.vmd")

    print("[1] Creating sample BVH...")
    create_sample_bvh(bvh_path)

    print("[2] Parsing BVH...")
    src_motion = parse_bvh(bvh_path)
    print(f"    Joints: {src_motion.num_joints}, Frames: {src_motion.num_frames}, FPS: {src_motion.fps}")

    print("[3] Creating sample PMX target...")
    pmx_target = create_sample_pmx()
    print(f"    Target bones: {len(pmx_target.bones)}")

    print("[4] Retargeting...")
    retargeter = Retargeter(src_motion, pmx_target, RetargetConfig(enable_foot_ik=True))
    vmd_motion = retargeter.retarget()
    print(f"    Retargeted tracks: {len(vmd_motion.bones)}, Total keys: {vmd_motion.num_bone_keys}")

    print("[5] Exporting VMD...")
    write_vmd(vmd_path, vmd_motion)
    print(f"    VMD written to {vmd_path} (size: {os.path.getsize(vmd_path)} bytes)")

    print("[6] Verifying VMD read back...")
    read_back = read_vmd(vmd_path)
    print(f"    Read back model: {read_back.model_name}, tracks: {len(read_back.bones)}")
    assert len(read_back.bones) == len(vmd_motion.bones)
    print(">>> PIPELINE VERIFICATION SUCCESSFUL! <<<")


if __name__ == "__main__":
    test_pipeline()
