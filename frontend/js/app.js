/**
 * MoCap2MMD Frontend Application Logic
 */
document.addEventListener("DOMContentLoaded", () => {
  const viewer = new MMDViewer("three-canvas-container");

  // State
  let sourceLoaded = false;
  let pmxLoaded = false;
  let isConverting = false;

  // DOM Elements
  const logConsole = document.getElementById("log-console");
  const sourceDropzone = document.getElementById("source-dropzone");
  const pmxDropzone = document.getElementById("pmx-dropzone");
  const sourceInfo = document.getElementById("source-info");
  const pmxInfo = document.getElementById("pmx-info");
  const motionBadge = document.getElementById("motion-badge");
  const pmxBadge = document.getElementById("pmx-badge");
  const btnRetarget = document.getElementById("btn-retarget");
  const btnExportVmd = document.getElementById("btn-export-vmd");

  const btnPlayPause = document.getElementById("btn-play-pause");
  const btnStop = document.getElementById("btn-stop");
  const timelineSlider = document.getElementById("timeline-slider");
  const timeCurrent = document.getElementById("time-current");
  const timeTotal = document.getElementById("time-total");
  const chkLoop = document.getElementById("chk-loop");

  const btnCameraReset = document.getElementById("btn-camera-reset");
  const btnToggleGrid = document.getElementById("btn-toggle-grid");

  const sliderScale = document.getElementById("slider-scale");
  const valScale = document.getElementById("val-scale");
  const sliderRootRot = document.getElementById("slider-root-rot");
  const valRootRot = document.getElementById("val-root-rot");
  const rootRotChips = document.querySelectorAll(".preset-chip-row .btn-chip");
  const chkIK = document.getElementById("chk-ik");
  const chkTwist = document.getElementById("chk-twist");
  const chkFingers = document.getElementById("chk-fingers");
  const chkAPose = document.getElementById("chk-a-pose");
  const mappingTableList = document.getElementById("mapping-table-list");

  // Logging Helper
  function log(msg, type = "info") {
    const entry = document.createElement("div");
    entry.className = `log-entry ${type}`;
    entry.textContent = `> ${msg}`;
    logConsole.appendChild(entry);
    logConsole.scrollTop = logConsole.scrollHeight;
  }

  // Check readiness
  function updateReadyState() {
    btnRetarget.disabled = !(sourceLoaded && pmxLoaded) || isConverting;
  }

  // Slider scale display
  sliderScale.addEventListener("input", (e) => {
    valScale.textContent = `${parseFloat(e.target.value).toFixed(2)}x`;
  });

  // Root rotation display & presets
  function setRootRotation(angle) {
    if (!sliderRootRot || !valRootRot) return;
    sliderRootRot.value = angle;
    valRootRot.textContent = `${angle}°`;
    rootRotChips.forEach((btn) => {
      if (parseFloat(btn.dataset.angle) === angle) {
        btn.classList.add("active");
      } else {
        btn.classList.remove("active");
      }
    });
  }

  if (sliderRootRot && valRootRot) {
    sliderRootRot.addEventListener("input", (e) => {
      const val = parseFloat(e.target.value);
      valRootRot.textContent = `${val}°`;
      rootRotChips.forEach((btn) => {
        if (parseFloat(btn.dataset.angle) === val) {
          btn.classList.add("active");
        } else {
          btn.classList.remove("active");
        }
      });
    });

    rootRotChips.forEach((btn) => {
      btn.addEventListener("click", () => {
        const angle = parseFloat(btn.dataset.angle);
        setRootRotation(angle);
      });
    });
  }

  // Source selection
  sourceDropzone.addEventListener("click", async () => {
    if (!window.pywebview) return log("デスクトップAPIが利用できません", "error");
    log("ソースモーション選択中...");
    const res = await window.pywebview.api.select_source_file();
    if (!res) return;
    if (!res.success) {
      log(`モーション読込エラー: ${res.error}`, "error");
      return;
    }
    sourceLoaded = true;
    motionBadge.textContent = res.format;
    motionBadge.style.color = "var(--accent-cyan)";
    document.getElementById("info-src-name").textContent = res.filename;
    document.getElementById("info-src-frames").textContent = `${res.num_frames}F (${res.fps}fps)`;
    document.getElementById("info-src-preset").textContent = res.preset;
    sourceInfo.style.display = "flex";

    // Update mapping table
    renderMappingTable(res.mapping);
    log(`ソースモーション読込完了: ${res.filename} (${res.preset})`);
    updateReadyState();
  });

  // PMX selection
  pmxDropzone.addEventListener("click", async () => {
    if (!window.pywebview) return log("デスクトップAPIが利用できません", "error");
    log("PMXモデル選択中...");
    const res = await window.pywebview.api.select_pmx_file();
    if (!res) return;
    if (!res.success) {
      log(`PMX読込エラー: ${res.error}`, "error");
      return;
    }
    pmxLoaded = true;
    pmxBadge.textContent = "PMX";
    pmxBadge.style.color = "var(--accent-green)";
    document.getElementById("info-pmx-name").textContent = res.model_name || res.filename;
    document.getElementById("info-pmx-bones").textContent = `${res.num_bones} ボーン`;
    document.getElementById("info-pmx-ik").textContent = res.has_foot_ik ? "あり" : "なし";
    pmxInfo.style.display = "flex";

    log(`3Dモデル読込中: ${res.model_name || res.filename}...`);
    try {
      await viewer.loadPMXFromBase64(res.pmx_base64, res.filename, res.base_dir, res.textures || {});
      log("3Dプレビュー表示完了", "info");
    } catch (err) {
      console.error(err);
      log(`3Dプレビュー表示失敗: ${err.message || err}`, "warn");
    }

    updateReadyState();
  });

  // Render Mapping Table
  function renderMappingTable(mapping) {
    mappingTableList.innerHTML = "";
    const entries = Object.entries(mapping);
    if (entries.length === 0) {
      mappingTableList.innerHTML = '<div style="padding: 12px; color: var(--text-muted); font-size: 11px; text-align: center;">対応ボーンなし</div>';
      return;
    }
    entries.forEach(([sem, name]) => {
      const row = document.createElement("div");
      row.className = "mapping-row";
      row.innerHTML = `<span class="mapping-sem">${sem}</span><span class="mapping-val">${name}</span>`;
      mappingTableList.appendChild(row);
    });
  }

  // Convert & Preview
  btnRetarget.addEventListener("click", async () => {
    if (!window.pywebview || isConverting) return;
    isConverting = true;
    btnRetarget.disabled = true;
    log("リターゲティング計算開始...");

    const options = {
      enable_foot_ik: chkIK.checked,
      enable_twist_bones: chkTwist.checked,
      enable_fingers: chkFingers.checked,
      scale_multiplier: parseFloat(sliderScale.value),
      root_rotation_y: parseFloat(sliderRootRot ? sliderRootRot.value : 0),
      auto_detect_a_pose: chkAPose.checked,
      target_fps: 30.0,
      overrides: {},
    };

    try {
      const res = await window.pywebview.api.execute_retarget(options);
      if (!res.success) {
        log(`リターゲティング失敗: ${res.error}`, "error");
        isConverting = false;
        updateReadyState();
        return;
      }

      log(`リターゲティング完了: ${res.num_tracks} トラック / ${res.num_keys} キーフレーム`);
      log("モーションプレビュー読み込み中...");

      await viewer.loadVMDFromBase64(res.vmd_base64);
      btnPlayPause.textContent = "⏸";
      timelineSlider.disabled = false;
      btnExportVmd.disabled = false;
      log("プレビュー再生開始！", "info");
    } catch (err) {
      log(`実行エラー: ${err.message}`, "error");
    } finally {
      isConverting = false;
      updateReadyState();
    }
  });

  // Export VMD
  btnExportVmd.addEventListener("click", async () => {
    if (!window.pywebview) return;
    log("VMDファイル保存先を選択中...");
    const res = await window.pywebview.api.save_vmd_file("motion.vmd");
    if (res.cancelled) {
      log("保存キャンセル");
      return;
    }
    if (res.success) {
      log(`VMD保存完了: ${res.path}`, "info");
    } else {
      log(`保存エラー: ${res.error}`, "error");
    }
  });

  // Playback Controls
  btnPlayPause.addEventListener("click", () => {
    if (viewer.isPlaying) {
      viewer.pause();
      btnPlayPause.textContent = "▶";
    } else {
      viewer.play();
      btnPlayPause.textContent = "⏸";
    }
  });

  btnStop.addEventListener("click", () => {
    viewer.stop();
    btnPlayPause.textContent = "▶";
  });

  timelineSlider.addEventListener("input", (e) => {
    const ratio = parseFloat(e.target.value) / 100.0;
    viewer.seek(ratio);
  });

  chkLoop.addEventListener("change", (e) => {
    viewer.setLoop(e.target.checked);
  });

  // Frame update from viewer
  viewer.onFrameUpdate = (currentTime, totalTime) => {
    if (totalTime > 0) {
      const ratio = (currentTime / totalTime) * 100;
      timelineSlider.value = ratio;

      const curSec = Math.floor(currentTime);
      const curFrame = Math.floor(currentTime * 30);
      const totSec = Math.floor(totalTime);
      const totFrame = Math.floor(totalTime * 30);

      timeCurrent.textContent = `${String(Math.floor(curSec / 60)).padStart(2, "0")}:${String(curSec % 60).padStart(2, "0")} / F: ${curFrame}`;
      timeTotal.textContent = `${String(Math.floor(totSec / 60)).padStart(2, "0")}:${String(totSec % 60).padStart(2, "0")} / F: ${totFrame}`;
    }
  };

  // Viewport buttons
  btnCameraReset.addEventListener("click", () => viewer.resetCamera());
  btnToggleGrid.addEventListener("click", () => viewer.toggleGrid());
});
