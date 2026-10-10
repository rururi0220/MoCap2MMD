/**
 * 3D Viewport controller using Three.js and MMDLoader
 */
class MMDViewer {
  constructor(containerId) {
    this.container = document.getElementById(containerId);
    this.scene = null;
    this.camera = null;
    this.renderer = null;
    this.controls = null;
    this.helper = null;
    this.loader = null;
    this.grid = null;

    this.currentMesh = null;
    this.currentAction = null;
    this.mixer = null;
    this.clock = new THREE.Clock();

    this.isPlaying = false;
    this.isLooping = true;
    this.totalDuration = 0;
    this.onFrameUpdate = null;

    this.init();
  }

  init() {
    const width = this.container.clientWidth;
    const height = this.container.clientHeight;

    // 1. Scene
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x0e1118);

    // 2. Camera
    this.camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 2000);
    this.camera.position.set(0, 15, 45);

    // 3. Renderer
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setSize(width, height);
    this.renderer.setPixelRatio(window.devicePixelRatio);
    this.renderer.shadowMap.enabled = true;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 0.95;
    this.container.appendChild(this.renderer.domElement);

    // 4. OrbitControls
    this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
    this.controls.target.set(0, 10, 0);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.05;

    // 5. Lights (calibrated to prevent white blowout)
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.4);
    this.scene.add(ambientLight);

    const dirLight = new THREE.DirectionalLight(0xffffff, 0.6);
    dirLight.position.set(15, 30, 20);
    dirLight.castShadow = true;
    this.scene.add(dirLight);

    const fillLight = new THREE.DirectionalLight(0xffffff, 0.25);
    fillLight.position.set(-15, 20, -15);
    this.scene.add(fillLight);

    const hemiLight = new THREE.HemisphereLight(0xffffff, 0x333333, 0.15);
    hemiLight.position.set(0, 50, 0);
    this.scene.add(hemiLight);

    // 6. Grid
    this.grid = new THREE.GridHelper(50, 50, 0x00d2ff, 0x222838);
    this.grid.position.y = 0;
    this.scene.add(this.grid);

    // 7. MMD Loader & Animation Helper
    this.loader = new THREE.MMDLoader();
    this.helper = new THREE.MMDAnimationHelper({ afterglow: 2.0, resetPhysicsOnLoop: false });

    // Window resize
    window.addEventListener("resize", () => this.onWindowResize());

    // Render loop
    this.animate = this.animate.bind(this);
    requestAnimationFrame(this.animate);
  }

  onWindowResize() {
    if (!this.container) return;
    const width = this.container.clientWidth;
    const height = this.container.clientHeight;
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height);
  }

  animate() {
    requestAnimationFrame(this.animate);

    const delta = this.clock.getDelta();

    if (this.helper && this.currentMesh && this.isPlaying) {
      this.helper.update(delta);

      if (this.mixer && this.onFrameUpdate) {
        const time = this.mixer.time;
        this.onFrameUpdate(time, this.totalDuration);
      }
    }

    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }

  resetCamera() {
    this.camera.position.set(0, 15, 45);
    this.controls.target.set(0, 10, 0);
    this.controls.update();
  }

  toggleGrid() {
    if (this.grid) {
      this.grid.visible = !this.grid.visible;
    }
  }

  loadPMXFromBase64(base64Data, filename = "model.pmx", resourcePath = "", textures = {}) {
    return new Promise((resolve, reject) => {
      try {
        // Remove old mesh
        if (this.currentMesh) {
          this.scene.remove(this.currentMesh);
          this.helper.remove(this.currentMesh);
          this.currentMesh = null;
        }

        // Set URL modifier on LoadingManager to resolve textures from Base64 data URLs
        this.loader.manager.setURLModifier((url) => {
          if (!url) return url;
          let decoded = url;
          try {
            decoded = decodeURI(url);
          } catch (_) {}
          const normalized = decoded.replace(/\\/g, "/");

          for (const [key, dataUrl] of Object.entries(textures)) {
            const normKey = key.replace(/\\/g, "/");
            const baseKey = normKey.split("/").pop();
            if (
              normalized.endsWith(normKey) ||
              decoded.endsWith(key) ||
              normalized.endsWith("/" + baseKey) ||
              normalized === baseKey
            ) {
              return dataUrl;
            }
          }
          return url;
        });

        const byteCharacters = atob(base64Data);
        const byteArray = new Uint8Array(byteCharacters.length);
        for (let i = 0; i < byteCharacters.length; i++) {
          byteArray[i] = byteCharacters.charCodeAt(i);
        }

        // Direct parse using MMDParser
        const parser = this.loader._getParser();
        const data = parser.parsePmx(byteArray.buffer, true);
        const builder = this.loader.meshBuilder;
        const mesh = builder.build(data, resourcePath || "");

        // Enhance material visibility, calibrate lighting, and ensure double-sided rendering
        if (mesh.material) {
          const mats = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
          mats.forEach((mat) => {
            mat.side = THREE.DoubleSide;
            // Prevent severe overexposure blowout: MMDLoader maps ambient directly to emissive
            if (mat.emissive) {
              mat.emissive.multiplyScalar(0.05);
            }
            // Prevent invisible rendering if texture failed or alpha is zero unintentionally
            if (mat.opacity === 0 && (!mat.name || !mat.name.toLowerCase().includes("shadow"))) {
              mat.opacity = 1.0;
              mat.transparent = false;
            }
            mat.needsUpdate = true;
          });
        }

        this.currentMesh = mesh;
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        this.scene.add(mesh);
        this.helper.add(mesh, { physics: false });

        // Adjust camera to model center
        if (mesh.geometry) {
          mesh.geometry.computeBoundingBox();
          const bbox = mesh.geometry.boundingBox;
          if (bbox) {
            const center = new THREE.Vector3();
            bbox.getCenter(center);
            this.controls.target.set(center.x, center.y, center.z);
            const height = bbox.max.y - bbox.min.y;
            this.camera.position.set(center.x, center.y, center.z + Math.max(height * 1.5, 25));
            this.controls.update();
          }
        }

        resolve(mesh);
      } catch (err) {
        reject(err);
      }
    });
  }

  loadVMDFromBase64(base64Data) {
    return new Promise((resolve, reject) => {
      try {
        if (!this.currentMesh) {
          return reject(new Error("PMX model not loaded in viewer"));
        }

        // 1. Stop current playback and unbind previous actions
        this.isPlaying = false;
        if (this.mixer) {
          this.mixer.stopAllAction();
        }

        // 2. Remove mesh from helper before modifying skeleton
        if (this.helper && this.helper.objects.has(this.currentMesh)) {
          this.helper.remove(this.currentMesh);
        }

        // 3. Reset all bones to pristine rest bind pose
        // AnimationBuilder uses bone.position as basePosition; it must NOT be posed!
        if (this.currentMesh.geometry && this.currentMesh.geometry.bones) {
          const gbones = this.currentMesh.geometry.bones;
          const bones = this.currentMesh.skeleton.bones;
          for (let i = 0; i < bones.length; i++) {
            const gb = gbones[i];
            if (gb) {
              bones[i].position.fromArray(gb.pos);
              bones[i].quaternion.fromArray(gb.rotq || [0, 0, 0, 1]);
              if (gb.scl) bones[i].scale.fromArray(gb.scl);
            }
          }
        }
        if (typeof this.currentMesh.pose === "function") {
          this.currentMesh.pose();
        }
        this.currentMesh.updateMatrixWorld(true);

        const byteCharacters = atob(base64Data);
        const byteArray = new Uint8Array(byteCharacters.length);
        for (let i = 0; i < byteCharacters.length; i++) {
          byteArray[i] = byteCharacters.charCodeAt(i);
        }

        // Direct parse using MMDParser
        const parser = this.loader._getParser();
        const vmd = parser.parseVmd(byteArray.buffer, true);
        const builder = this.loader.animationBuilder;
        const animation = builder.build(vmd, this.currentMesh);

        this.helper.add(this.currentMesh, {
          animation: animation,
          physics: false,
        });

        this.mixer = this.helper.objects.get(this.currentMesh).mixer;
        this.totalDuration = animation.duration;
        this.isPlaying = true;
        resolve(animation);
      } catch (err) {
        reject(err);
      }
    });
  }

  play() {
    this.isPlaying = true;
  }

  pause() {
    this.isPlaying = false;
  }

  stop() {
    this.isPlaying = false;
    if (this.mixer) {
      this.mixer.setTime(0);
      if (this.onFrameUpdate) {
        this.onFrameUpdate(0, this.totalDuration);
      }
    }
  }

  seek(ratio) {
    if (this.mixer && this.totalDuration > 0) {
      const targetTime = ratio * this.totalDuration;
      this.mixer.setTime(targetTime);
      if (this.onFrameUpdate) {
        this.onFrameUpdate(targetTime, this.totalDuration);
      }
    }
  }

  setLoop(loop) {
    this.isLooping = loop;
    if (this.mixer) {
      // Configure loop mode if action exists
      const obj = this.helper.objects.get(this.currentMesh);
      if (obj && obj.action) {
        obj.action.loop = loop ? THREE.LoopRepeat : THREE.LoopOnce;
      }
    }
  }
}
