/**
 * FloorViz VR Integration Module
 * ================================
 * Production-ready WebXR/Three.js VR system for FloorViz
 * Compatible with Three.js r128 (the version used by index_v9.html via CDN).
 *
 * Architecture:
 *   - FloorVizVR mixin extends any FloorVizApp-shaped object in-place
 *   - PlayerRig manages XR camera + controller hierarchy
 *   - VRLocomotion handles teleport + smooth movement
 *   - VRCollision provides wall-sliding bounding-box collision
 *   - VRUIPanel renders controller-attached floating canvas menu
 *   - VRAudio stubs spatial audio architecture (ready for future positional audio)
 *
 * ── Required global dependencies (load BEFORE this script) ───────────────────
 *   1. three.min.js  r128  (already loaded by index_v9.html)
 *   2. XRControllerModelFactory.js  r128 UMD build
 *      <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/webxr/XRControllerModelFactory.js"></script>
 *   3. VRButton shim (inline shim in index_v9.html — no separate file needed)
 *      Provides: window.VRButton
 *
 * ── Usage from index_v9.html ──────────────────────────────────────────────────
 *   The wiring shim in index_v9.html does:
 *     FloorVizVR.mixin(appProxy);   // extend proxy with VR methods
 *     appProxy.initVR();            // check WebXR support + enable button
 *   Then from renderer.setAnimationLoop:
 *     appProxy.handleVRFrame(time, frame);
 *   And after buildModel():
 *     appProxy.onModelReady();
 *
 * ── IMPORTS (commented out — using UMD globals instead of ES modules) ─────────
 * import { VRButton }                from 'three/examples/jsm/webxr/VRButton.js';
 * import { XRControllerModelFactory } from 'three/examples/jsm/webxr/XRControllerModelFactory.js';
 * import { XRHandModelFactory }       from 'three/examples/jsm/webxr/XRHandModelFactory.js';
 */

'use strict';

/* ═══════════════════════════════════════════════════════════════════════════
   §1  CONSTANTS & CONFIG
═══════════════════════════════════════════════════════════════════════════ */

const VR_CONFIG = {
  PLAYER_HEIGHT:        1.6,   // metres – eye height
  PLAYER_RADIUS:        0.25,  // metres – collision capsule radius
  TELEPORT_MAX_DIST:    20,    // metres – max teleport ray length
  TELEPORT_ARC_SEGS:    20,    // parabolic arc segments
  SMOOTH_SPEED:         3.0,   // m/s
  SNAP_TURN_ANGLE:      Math.PI / 4, // 45°
  SNAP_TURN_DEADZONE:   0.6,   // joystick threshold
  FADE_DURATION:        0.25,  // seconds
  UI_DISTANCE:          0.4,   // metres from camera
  UI_WIDTH:             0.35,  // metres
  UI_HEIGHT:            0.22,  // metres
  FLOOR_Y_TOLERANCE:    0.05,  // metres – floor hit tolerance
  SHADOW_MAP_SIZE:      512,   // reduced for VR perf
  XR_FRAMERATE_HINT:    72,    // Hz
};

/* ═══════════════════════════════════════════════════════════════════════════
   §2  WEBXR COMPATIBILITY GUARD
═══════════════════════════════════════════════════════════════════════════ */

const VRCompat = {
  /**
   * Returns a Promise<boolean> — true if immersive-vr is supported.
   */
  async isSupported() {
    if (!navigator.xr) return false;
    try {
      return await navigator.xr.isSessionSupported('immersive-vr');
    } catch {
      return false;
    }
  },

  /**
   * Inject a polite unsupported banner into the DOM.
   */
  showUnsupportedBanner() {
    const el = document.createElement('div');
    el.id = 'vr-unsupported';
    el.innerHTML = `
      <span>⚠️ WebXR not available in this browser.</span>
      <a href="https://immersiveweb.dev/" target="_blank" rel="noopener">Learn more</a>`;
    document.body.appendChild(el);
  },
};

/* ═══════════════════════════════════════════════════════════════════════════
   §3  PLAYER RIG
   Hierarchy: playerRig (Object3D) → camera, controller0, controller1
═══════════════════════════════════════════════════════════════════════════ */

class PlayerRig {
  /**
   * @param {THREE.WebGLRenderer} renderer
   * @param {THREE.Scene}         scene
   * @param {THREE.Camera}        existingCamera
   */
  constructor(renderer, scene, existingCamera) {
    this.renderer  = renderer;
    this.scene     = scene;
    this.camera    = existingCamera;

    // Root transform that moves the player through the world
    this.rig = new THREE.Group();
    this.rig.name = 'XR_PlayerRig';

    // Spawn at the first tour room if available, otherwise fall back to
    // projecting the orbit camera position down to floor level.
    // This puts the player INSIDE the building at eye height, not at the
    // bird's-eye orbit position.
    const firstRoom = window._tourRooms?.[0];
    if (firstRoom) {
      // Use the tour's pre-computed eye position (already nudged clear of walls)
      this.rig.position.set(firstRoom.eyeX ?? firstRoom.cx, 0, firstRoom.eyeZ ?? firstRoom.cz);
    } else {
      // Fallback: project orbit camera onto floor plane
      this.rig.position.set(existingCamera.position.x, 0, existingCamera.position.z);
    }

    scene.add(this.rig);

    // The renderer's XR camera is automatically inserted as a child of the
    // scene, but we need it parented to our rig for locomotion to work.
    // We achieve this by *not* adding the PerspectiveCamera ourselves —
    // WebXR provides its own ArrayCamera. We instead parent the rig and
    // move the rig to move the player.
    this.rig.add(this.camera); // camera becomes rig-relative

    this._initControllers();
  }

  _initControllers() {
    const factory = new XRControllerModelFactory();
    const renderer = this.renderer;

    // Controller grips (visual model) + input sources
    this.controllers = [0, 1].map(i => {
      const grip        = renderer.xr.getControllerGrip(i);
      const controller  = renderer.xr.getController(i);
      const model       = factory.createControllerModel(grip);

      grip.add(model);
      this.rig.add(grip);
      this.rig.add(controller);

      // Ray pointer visual
      const ray = this._buildRayLine();
      controller.add(ray);
      controller._rayLine = ray;

      return { grip, controller, model, index: i };
    });
  }

  _buildRayLine() {
    const mat = new THREE.LineBasicMaterial({
      color:       0x00ffcc,
      transparent: true,
      opacity:     0.6,
      linewidth:   2,
    });
    // Will be updated each frame
    const geo = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(0, 0, 0),
      new THREE.Vector3(0, 0, -1),
    ]);
    const line = new THREE.Line(geo, mat);
    line.name = 'ControllerRay';
    line.scale.z = 5;
    line.visible = true;
    return line;
  }

  /**
   * Move rig in world space, used by locomotion system.
   * @param {THREE.Vector3} delta  world-space displacement
   */
  move(delta) {
    this.rig.position.add(delta);
  }

  /**
   * Teleport rig to a world floor position.
   * @param {THREE.Vector3} floorPos
   */
  teleportTo(floorPos) {
    this.rig.position.x = floorPos.x;
    this.rig.position.z = floorPos.z;
    // Y stays at floor level; rig.y == 0 means camera is at PLAYER_HEIGHT
    this.rig.position.y = floorPos.y;
  }

  /**
   * Reset XR camera orientation (recenter).
   */
  recenter() {
    // WebXR provides recenter via reference space, but a simple approach:
    const session = this.renderer.xr.getSession();
    if (session && session.requestReferenceSpace) {
      // Reset by requesting a fresh local reference space
      session.requestReferenceSpace('local-floor').then(rs => {
        this.renderer.xr.setReferenceSpace(rs);
      }).catch(() => {});
    }
  }

  get position() { return this.rig.position; }
}

/* ═══════════════════════════════════════════════════════════════════════════
   §4  COLLISION SYSTEM
   Axis-aligned bounding box (AABB) wall collision with sliding.
═══════════════════════════════════════════════════════════════════════════ */

class VRCollision {
  /**
   * @param {THREE.Scene} scene  – we'll extract wall meshes automatically
   */
  constructor(scene) {
    this.wallBoxes = [];  // Array<THREE.Box3>
    this._buildFromScene(scene);
  }

  /**
   * Scan scene for meshes named 'wall*' and build AABB list.
   */
  _buildFromScene(scene) {
    scene.traverse(obj => {
      if (
        obj.isMesh &&
        (obj.name.toLowerCase().startsWith('wall') ||
         obj.userData?.type === 'wall')
      ) {
        const box = new THREE.Box3().setFromObject(obj);
        // Expand slightly so thin walls are never missed
        box.expandByScalar(0.02);
        this.wallBoxes.push(box);
      }
    });
    console.log(`[VRCollision] Built ${this.wallBoxes.length} wall AABBs`);
  }

  /**
   * Rebuild — call after renderModel() adds geometry.
   */
  rebuild(scene) {
    this.wallBoxes = [];
    this._buildFromScene(scene);
  }

  /**
   * Resolve a desired position against wall AABBs.
   * Returns the allowed position (with wall-sliding).
   *
   * @param {THREE.Vector3} current  current rig position
   * @param {THREE.Vector3} desired  proposed new rig position
   * @returns {THREE.Vector3}        collision-resolved position
   */
  resolve(current, desired) {
    const r = VR_CONFIG.PLAYER_RADIUS;

    // Player capsule on the floor plane — treat as circle
    let px = desired.x;
    let pz = desired.z;

    for (const box of this.wallBoxes) {
      // Expand box by player radius (Minkowski sum)
      const minX = box.min.x - r;
      const maxX = box.max.x + r;
      const minZ = box.min.z - r;
      const maxZ = box.max.z + r;

      if (px > minX && px < maxX && pz > minZ && pz < maxZ) {
        // Penetration detected — slide along the axis with least overlap
        const overlapLeft  = px - minX;
        const overlapRight = maxX - px;
        const overlapFront = pz - minZ;
        const overlapBack  = maxZ - pz;

        const minOverlap = Math.min(overlapLeft, overlapRight, overlapFront, overlapBack);

        if (minOverlap === overlapLeft)       px = minX;
        else if (minOverlap === overlapRight) px = maxX;
        else if (minOverlap === overlapFront) pz = minZ;
        else                                  pz = maxZ;
      }
    }

    return new THREE.Vector3(px, desired.y, pz);
  }
}

/* ═══════════════════════════════════════════════════════════════════════════
   §5  TELEPORT SYSTEM
   Parabolic arc cast onto floor geometry; fade + move on trigger release.
═══════════════════════════════════════════════════════════════════════════ */

class VRTeleport {
  /**
   * @param {THREE.Scene}    scene
   * @param {PlayerRig}      rig
   * @param {VRCollision}    collision
   * @param {VRFade}         fade
   */
  constructor(scene, rig, collision, fade) {
    this.scene     = scene;
    this.rig       = rig;
    this.collision = collision;
    this.fade      = fade;

    this.active     = false;   // is a controller currently aiming?
    this.validHit   = false;
    this.hitPoint   = new THREE.Vector3();

    this._raycaster = new THREE.Raycaster();
    this._floorMeshes = [];

    this._buildMarker();
    this._buildArcLine();
    this._gatherFloorMeshes();
  }

  _gatherFloorMeshes() {
    this.scene.traverse(obj => {
      if (
        obj.isMesh &&
        (obj.name.toLowerCase().includes('floor') ||
         obj.userData?.type === 'floor')
      ) {
        this._floorMeshes.push(obj);
      }
    });
  }

  /** Refresh floor meshes after model load. */
  rebuild() {
    this._floorMeshes = [];
    this._gatherFloorMeshes();
    console.log(`[VRTeleport] Floor meshes: ${this._floorMeshes.length}`);
  }

  _buildMarker() {
    // Glowing ring on the floor at the target point
    const geo = new THREE.RingGeometry(0.15, 0.25, 32);
    const mat = new THREE.MeshBasicMaterial({
      color:       0x00ffcc,
      side:        THREE.DoubleSide,
      transparent: true,
      opacity:     0.85,
    });
    this.marker = new THREE.Mesh(geo, mat);
    this.marker.rotation.x = -Math.PI / 2;
    this.marker.visible = false;
    this.marker.name = 'TeleportMarker';

    // Inner dot
    const dotGeo = new THREE.CircleGeometry(0.06, 16);
    const dot    = new THREE.Mesh(dotGeo, new THREE.MeshBasicMaterial({ color: 0xffffff }));
    dot.rotation.x = -Math.PI / 2;
    dot.position.y = 0.001;
    this.marker.add(dot);

    this.scene.add(this.marker);
  }

  _buildArcLine() {
    const points = Array.from({ length: VR_CONFIG.TELEPORT_ARC_SEGS + 1 },
      () => new THREE.Vector3());
    const geo = new THREE.BufferGeometry().setFromPoints(points);
    const mat = new THREE.LineBasicMaterial({
      color:       0x00ffcc,
      transparent: true,
      opacity:     0.5,
    });
    this.arcLine = new THREE.Line(geo, mat);
    this.arcLine.visible = false;
    this.arcLine.name = 'TeleportArc';
    this.arcLine.frustumCulled = false;
    this.scene.add(this.arcLine);
  }

  /**
   * Called each frame while the controller trigger is held.
   * @param {THREE.Object3D} controller
   */
  update(controller) {
    this.active   = true;
    this.validHit = false;

    // Parabolic arc cast from controller tip
    const origin    = new THREE.Vector3();
    const direction = new THREE.Vector3(0, 0, -1);
    controller.getWorldPosition(origin);
    controller.getWorldDirection(direction).negate();

    const gravity = -9.8;
    const speed   = 8;
    const dt      = 0.04; // time step per segment

    const arcPoints = [];
    const vel = direction.clone().multiplyScalar(speed);
    const pos = origin.clone();

    let hit = null;

    for (let i = 0; i <= VR_CONFIG.TELEPORT_ARC_SEGS; i++) {
      arcPoints.push(pos.clone());

      // Cast a short segment to check intersection
      if (i > 0) {
        const segDir  = pos.clone().sub(arcPoints[i - 1]);
        const segLen  = segDir.length();
        this._raycaster.set(arcPoints[i - 1], segDir.normalize());
        this._raycaster.far = segLen + 0.05;

        const hits = this._raycaster.intersectObjects(this._floorMeshes, false);
        if (hits.length > 0) {
          hit = hits[0];
          arcPoints.push(hit.point.clone());
          break;
        }
      }

      vel.y += gravity * dt;
      pos.addScaledVector(vel, dt);

      // Stop if we've gone below floor level
      if (pos.y < -0.5) break;
    }

    // Update arc geometry
    const posArr = this.arcLine.geometry.attributes.position;
    for (let i = 0; i < arcPoints.length; i++) {
      if (i < VR_CONFIG.TELEPORT_ARC_SEGS + 1) {
        posArr.setXYZ(i, arcPoints[i].x, arcPoints[i].y, arcPoints[i].z);
      }
    }
    posArr.needsUpdate = true;
    this.arcLine.visible = true;

    if (hit) {
      this.validHit = true;
      this.hitPoint.copy(hit.point);
      this.marker.position.copy(hit.point);
      this.marker.position.y += 0.01;
      this.marker.visible = true;
    } else {
      this.marker.visible = false;
    }
  }

  /**
   * Called when trigger is released. Executes the teleport.
   */
  async execute() {
    this.active = false;
    this.arcLine.visible = false;
    this.marker.visible  = false;

    if (!this.validHit) return;

    // Resolve against collision system
    const resolved = this.collision.resolve(
      this.rig.position,
      new THREE.Vector3(this.hitPoint.x, this.rig.position.y, this.hitPoint.z)
    );

    await this.fade.fadeOut();
    this.rig.teleportTo(resolved);
    await this.fade.fadeIn();
  }

  cancel() {
    this.active = false;
    this.arcLine.visible = false;
    this.marker.visible  = false;
  }
}

/* ═══════════════════════════════════════════════════════════════════════════
   §6  SMOOTH LOCOMOTION
   Left joystick → headset-relative movement + right joystick → snap turn.
═══════════════════════════════════════════════════════════════════════════ */

class VRSmoothLocomotion {
  /**
   * @param {PlayerRig}   rig
   * @param {THREE.Camera} camera
   * @param {VRCollision}  collision
   */
  constructor(rig, camera, collision) {
    this.rig       = rig;
    this.camera    = camera;
    this.collision = collision;

    this._snapCooldown = false; // prevent rapid snap turns
    this._moveDir = new THREE.Vector3();
    this._camDir  = new THREE.Vector3();
  }

  /**
   * @param {XRGamepad[]} gamepads  — from session.inputSources
   * @param {number}      delta     — seconds
   */
  update(gamepads, delta) {
    if (!gamepads || gamepads.length === 0) return;

    const leftPad  = gamepads[0];
    const rightPad = gamepads[1];

    if (leftPad?.axes) {
      const ax = leftPad.axes[2] ?? 0; // X
      const ay = leftPad.axes[3] ?? 0; // Y

      if (Math.abs(ax) > 0.15 || Math.abs(ay) > 0.15) {
        // Get camera horizontal facing (ignore camera pitch)
        this.camera.getWorldDirection(this._camDir);
        this._camDir.y = 0;
        this._camDir.normalize();

        const right = new THREE.Vector3().crossVectors(this._camDir, new THREE.Vector3(0, 1, 0)).normalize();

        this._moveDir
          .copy(this._camDir).multiplyScalar(-ay)
          .addScaledVector(right, ax);

        const speed   = VR_CONFIG.SMOOTH_SPEED;
        const desired = this.rig.position.clone().addScaledVector(this._moveDir, speed * delta);
        const safe    = this.collision.resolve(this.rig.position, desired);
        this.rig.rig.position.copy(safe);
      }
    }

    if (rightPad?.axes) {
      const ax = rightPad.axes[2] ?? 0;

      if (Math.abs(ax) > VR_CONFIG.SNAP_TURN_DEADZONE && !this._snapCooldown) {
        const dir = Math.sign(ax);
        this.rig.rig.rotateY(-dir * VR_CONFIG.SNAP_TURN_ANGLE);
        this._snapCooldown = true;
        setTimeout(() => { this._snapCooldown = false; }, 350);
      }
    }
  }
}

/* ═══════════════════════════════════════════════════════════════════════════
   §7  FADE OVERLAY
   Full-screen black quad attached to camera for fade-to-black transitions.
═══════════════════════════════════════════════════════════════════════════ */

class VRFade {
  /**
   * @param {THREE.Camera} camera
   */
  constructor(camera) {
    this.camera = camera;
    this._opacity = 0;
    this._mesh = this._build();
    camera.add(this._mesh);
  }

  _build() {
    const geo = new THREE.PlaneGeometry(2, 2);
    const mat = new THREE.MeshBasicMaterial({
      color:       0x000000,
      transparent: true,
      opacity:     0,
      depthTest:   false,
      depthWrite:  false,
    });
    const mesh = new THREE.Mesh(geo, mat);
    mesh.position.z = -0.1; // just in front of camera near plane
    mesh.renderOrder = 999;
    mesh.frustumCulled = false;
    mesh.name = 'VRFadeOverlay';
    return mesh;
  }

  fadeOut() {
    return this._animate(0, 1);
  }

  fadeIn() {
    return this._animate(1, 0);
  }

  _animate(from, to) {
    return new Promise(resolve => {
      const dur   = VR_CONFIG.FADE_DURATION * 1000; // ms
      const start = performance.now();
      const mat   = this._mesh.material;

      const tick = (now) => {
        const t = Math.min((now - start) / dur, 1);
        mat.opacity = from + (to - from) * t;
        if (t < 1) requestAnimationFrame(tick);
        else resolve();
      };

      requestAnimationFrame(tick);
    });
  }
}

/* ═══════════════════════════════════════════════════════════════════════════
   §8  VR UI PANEL
   Canvas-rendered floating panel attached to the left controller grip.
═══════════════════════════════════════════════════════════════════════════ */

class VRUIPanel {
  /**
   * @param {THREE.Object3D} attachTo  – left controller grip
   * @param {object}         callbacks – { onExit, onNext, onPrev, onToggleMove, onToggleMinimap, onRecenter }
   */
  constructor(attachTo, callbacks) {
    this.attachTo  = attachTo;
    this.callbacks = callbacks;

    this._buttons = [
      { label: '✕  Exit VR',         key: 'onExit' },
      { label: '→  Next Room',        key: 'onNext' },
      { label: '←  Prev Room',        key: 'onPrev' },
      { label: '⇆  Toggle Move',      key: 'onToggleMove' },
      { label: '⊟  Minimap',          key: 'onToggleMinimap' },
      { label: '⊕  Recenter',         key: 'onRecenter' },
    ];

    this._hoveredIdx = -1;
    this._raycaster  = new THREE.Raycaster();

    this._buildPanel();
  }

  _buildPanel() {
    const W = 512, H = 320;
    this._canvas = document.createElement('canvas');
    this._canvas.width  = W;
    this._canvas.height = H;
    this._ctx = this._canvas.getContext('2d');

    this._texture = new THREE.CanvasTexture(this._canvas);

    const geo = new THREE.PlaneGeometry(VR_CONFIG.UI_WIDTH, VR_CONFIG.UI_HEIGHT);
    const mat = new THREE.MeshBasicMaterial({
      map:         this._texture,
      transparent: true,
      side:        THREE.DoubleSide,
    });
    this.mesh = new THREE.Mesh(geo, mat);
    this.mesh.name = 'VRUIPanel';

    // Position above/in-front of controller
    this.mesh.position.set(0, 0.12, -0.05);
    this.mesh.rotation.x = -Math.PI / 6;

    this.attachTo.add(this.mesh);
    this._render();
  }

  _render(hoveredIdx = -1) {
    const ctx = this._ctx;
    const W = this._canvas.width, H = this._canvas.height;

    // Background
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = 'rgba(0,10,20,0.88)';
    this._roundRect(ctx, 0, 0, W, H, 24);
    ctx.fill();

    // Border
    ctx.strokeStyle = '#00ffcc';
    ctx.lineWidth = 3;
    this._roundRect(ctx, 2, 2, W - 4, H - 4, 22);
    ctx.stroke();

    // Title
    ctx.fillStyle = '#00ffcc';
    ctx.font = 'bold 28px monospace';
    ctx.textAlign = 'center';
    ctx.fillText('FloorViz VR', W / 2, 40);

    // Divider
    ctx.strokeStyle = 'rgba(0,255,204,0.3)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(20, 54); ctx.lineTo(W - 20, 54);
    ctx.stroke();

    // Buttons (2 columns × 3 rows)
    const cols = 2, rows = 3;
    const bW = (W - 48) / cols;
    const bH = (H - 80) / rows;
    const pad = 8;

    this._buttons.forEach((btn, i) => {
      const col = i % cols;
      const row = Math.floor(i / cols);
      const x = 16 + col * (bW + 8);
      const y = 64 + row * (bH + 4);

      const hovered = (i === hoveredIdx);
      ctx.fillStyle = hovered ? 'rgba(0,255,204,0.25)' : 'rgba(255,255,255,0.05)';
      this._roundRect(ctx, x, y, bW, bH - pad, 10);
      ctx.fill();

      ctx.fillStyle = hovered ? '#ffffff' : '#aaffee';
      ctx.font = `${hovered ? 'bold' : 'normal'} 22px monospace`;
      ctx.textAlign = 'center';
      ctx.fillText(btn.label, x + bW / 2, y + bH / 2 - pad / 2 + 4);

      // Store button bounds for hit testing (normalised UV)
      btn._bounds = { x, y, w: bW, h: bH - pad };
    });

    this._texture.needsUpdate = true;
  }

  _roundRect(ctx, x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.lineTo(x + w - r, y);
    ctx.quadraticCurveTo(x + w, y, x + w, y + r);
    ctx.lineTo(x + w, y + h - r);
    ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
    ctx.lineTo(x + r, y + h);
    ctx.quadraticCurveTo(x, y + h, x, y + h - r);
    ctx.lineTo(x, y + r);
    ctx.quadraticCurveTo(x, y, x + r, y);
    ctx.closePath();
  }

  /**
   * Test controller ray against panel, update hover + fire callbacks.
   * @param {THREE.Object3D} controller
   */
  testRay(controller) {
    const origin    = new THREE.Vector3();
    const direction = new THREE.Vector3();
    controller.getWorldPosition(origin);
    controller.getWorldDirection(direction).negate();

    this._raycaster.set(origin, direction);
    const hits = this._raycaster.intersectObject(this.mesh);

    if (hits.length === 0) {
      if (this._hoveredIdx !== -1) {
        this._hoveredIdx = -1;
        this._render(-1);
      }
      return -1;
    }

    const uv  = hits[0].uv;
    const px  = uv.x * this._canvas.width;
    const py  = (1 - uv.y) * this._canvas.height;

    let hovered = -1;
    this._buttons.forEach((btn, i) => {
      if (!btn._bounds) return;
      const b = btn._bounds;
      if (px >= b.x && px <= b.x + b.w && py >= b.y && py <= b.y + b.h) {
        hovered = i;
      }
    });

    if (hovered !== this._hoveredIdx) {
      this._hoveredIdx = hovered;
      this._render(hovered);
    }

    return hovered;
  }

  /**
   * Trigger the callback for the currently hovered button.
   */
  activateHovered() {
    if (this._hoveredIdx < 0) return;
    const btn = this._buttons[this._hoveredIdx];
    if (btn && this.callbacks[btn.key]) {
      this.callbacks[btn.key]();
    }
  }

  setVisible(v) {
    this.mesh.visible = v;
  }
}

/* ═══════════════════════════════════════════════════════════════════════════
   §9  SPATIAL AUDIO STUB
   Architecture ready for future positional room audio implementation.
═══════════════════════════════════════════════════════════════════════════ */

class VRAudio {
  constructor() {
    this._listener  = null;
    this._roomSources = new Map(); // roomId → THREE.PositionalAudio
    this._ready = false;
  }

  /**
   * Call after user gesture (required for AudioContext).
   * @param {THREE.Camera} camera
   */
  init(camera) {
    // Three.js audio listener attaches to the camera
    this._listener = new THREE.AudioListener();
    camera.add(this._listener);
    this._ready = true;
    console.log('[VRAudio] Audio listener initialised. Ready for room audio sources.');
  }

  /**
   * Register a positional audio source for a room.
   * @param {string}         roomId
   * @param {THREE.Vector3}  position   – room centroid
   * @param {AudioBuffer}    buffer     – decoded audio
   * @param {object}         opts       – { refDistance, rolloffFactor, loop }
   */
  addRoomSource(roomId, position, buffer, opts = {}) {
    if (!this._ready) return;

    const src = new THREE.PositionalAudio(this._listener);
    src.setBuffer(buffer);
    src.setRefDistance(opts.refDistance    ?? 2);
    src.setRolloffFactor(opts.rolloffFactor ?? 2);
    src.setLoop(opts.loop ?? true);
    src.position.copy(position);

    // Sources are added to the scene as standalone objects
    this._roomSources.set(roomId, src);

    // Caller is responsible for adding to scene:
    // scene.add(src); src.play();
    return src;
  }

  /**
   * Remove and stop a room source.
   */
  removeRoomSource(roomId) {
    const src = this._roomSources.get(roomId);
    if (src) { src.stop(); this._roomSources.delete(roomId); }
  }

  dispose() {
    this._roomSources.forEach(src => src.stop());
    this._roomSources.clear();
  }
}

/* ═══════════════════════════════════════════════════════════════════════════
   §10  PERFORMANCE OPTIMIZER
   Applies VR-specific render settings; toggles shadow/pixel-ratio.
═══════════════════════════════════════════════════════════════════════════ */

class VRPerformance {
  /**
   * @param {THREE.WebGLRenderer} renderer
   * @param {THREE.Scene}         scene
   */
  constructor(renderer, scene) {
    this.renderer = renderer;
    this.scene    = scene;

    this._shadowsWereEnabled = renderer.shadowMap.enabled;
    this._prevPixelRatio     = renderer.getPixelRatio();
  }

  /** Call when entering VR. */
  enterVR() {
    const r = this.renderer;

    // Reduce pixel ratio to ease GPU; Quest runtime handles supersampling
    r.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));

    // Soft shadows are expensive; switch to basic
    r.shadowMap.type    = THREE.BasicShadowMap;
    r.shadowMap.enabled = true;

    // Reduce shadow map resolution on all lights
    this.scene.traverse(obj => {
      if (obj.isLight && obj.shadow) {
        obj.shadow.mapSize.set(
          VR_CONFIG.SHADOW_MAP_SIZE,
          VR_CONFIG.SHADOW_MAP_SIZE
        );
        obj.shadow.map?.dispose();
        obj.shadow.map = null;
      }
    });

    // Instanced mesh frustum culling is on by default — ensure it stays on
    this.scene.traverse(obj => {
      if (obj.isInstancedMesh) obj.frustumCulled = true;
    });

    console.log('[VRPerformance] VR render optimisations applied.');
  }

  /** Restore desktop settings when exiting VR. */
  exitVR() {
    const r = this.renderer;
    r.setPixelRatio(this._prevPixelRatio);
    r.shadowMap.type    = THREE.PCFSoftShadowMap;
    r.shadowMap.enabled = this._shadowsWereEnabled;
  }
}

/* ═══════════════════════════════════════════════════════════════════════════
   §11  VR TOUR ADAPTER
   Bridges FloorVizApp's existing tour system to VR context.
═══════════════════════════════════════════════════════════════════════════ */

class VRTourAdapter {
  /**
   * @param {FloorVizApp} app
   * @param {PlayerRig}   rig
   * @param {VRFade}      fade
   */
  constructor(app, rig, fade) {
    this.app  = app;
    this.rig  = rig;
    this.fade = fade;

    // Gaze hotspot meshes for room selection in VR
    this._hotspots = [];
    this._gazeRaycaster = new THREE.Raycaster();
    this._gazeTimer     = 0;
    this._gazeTarget    = null;
    this.GAZE_DWELL     = 2.0; // seconds to activate hotspot
  }

  /**
   * Build visible hotspot spheres from room metadata.
   * Call after renderModel().
   * @param {object[]} rooms  – app.model.rooms
   * @param {THREE.Scene} scene
   */
  buildHotspots(rooms, scene) {
    this._hotspots.forEach(h => scene.remove(h));
    this._hotspots = [];

    // Use tour room data (_tourRooms) if available — it has wall-cleared eyeX/eyeZ positions
    const tourRooms = window._tourRooms || [];

    rooms.forEach((room, i) => {
      // Prefer the tour's nudged eye position so hotspots match teleport targets
      const tr  = tourRooms[i];
      const cx  = tr?.eyeX ?? room.centroid?.[0] ?? room.position?.x ?? 0;
      const cz  = tr?.eyeZ ?? room.centroid?.[1] ?? room.position?.z ?? 0;
      const cy  = VR_CONFIG.PLAYER_HEIGHT - 0.3; // slightly below eye level

      const geo = new THREE.SphereGeometry(0.12, 16, 16);
      const mat = new THREE.MeshStandardMaterial({
        color:       0x00ffcc,
        emissive:    0x003322,
        roughness:   0.3,
        metalness:   0.6,
        transparent: true,
        opacity:     0.85,
      });
      const mesh = new THREE.Mesh(geo, mat);
      mesh.position.set(cx, cy, cz);
      mesh.name = `VRHotspot_${room.id ?? i}`;
      mesh.userData = {
        roomId:   room.id ?? i,
        roomName: room.label ?? room.name ?? `Room ${i}`,
        eyeX: cx, eyeZ: cz,  // store teleport target
      };

      scene.add(mesh);
      this._hotspots.push(mesh);
    });
  }

  /**
   * Update gaze selection. Call from animation loop in VR.
   * @param {THREE.Camera} camera
   * @param {number}       delta   seconds
   */
  updateGaze(camera, delta) {
    // Cast ray from camera forward
    this._gazeRaycaster.setFromCamera({ x: 0, y: 0 }, camera);
    const hits = this._gazeRaycaster.intersectObjects(this._hotspots);

    if (hits.length > 0) {
      const target = hits[0].object;
      if (this._gazeTarget !== target) {
        this._gazeTarget = target;
        this._gazeTimer  = 0;
      } else {
        this._gazeTimer += delta;
        // Visual feedback – pulse scale
        const t = this._gazeTimer / this.GAZE_DWELL;
        target.scale.setScalar(1 + 0.3 * Math.sin(t * Math.PI));

        if (this._gazeTimer >= this.GAZE_DWELL) {
          this._activateHotspot(target);
        }
      }
    } else {
      if (this._gazeTarget) this._gazeTarget.scale.setScalar(1);
      this._gazeTarget = null;
      this._gazeTimer  = 0;
    }
  }

  /**
   * Fly to room in VR (teleport + fade instead of orbit-camera tween).
   * @param {THREE.Mesh} hotspotMesh
   */
  async _activateHotspot(hotspotMesh) {
    this._gazeTimer  = 0;
    this._gazeTarget = null;

    const { eyeX, eyeZ, roomId } = hotspotMesh.userData;

    await this.fade.fadeOut();

    // Teleport to the room's wall-cleared eye position (stored on userData)
    const tx = eyeX ?? hotspotMesh.position.x;
    const tz = eyeZ ?? hotspotMesh.position.z;
    this.rig.teleportTo(new THREE.Vector3(tx, this.rig.position.y, tz));

    // Sync desktop tour index so minimap / room labels stay consistent
    try { this.app.flyToRoom?.(roomId); } catch {}

    await this.fade.fadeIn();
  }

  setVisible(v) {
    this._hotspots.forEach(h => { h.visible = v; });
  }
}

/* ═══════════════════════════════════════════════════════════════════════════
   §12  MAIN VR MIXIN
   Apply to FloorVizApp instance with FloorVizVR.mixin(app).
═══════════════════════════════════════════════════════════════════════════ */

const FloorVizVR = {

  /**
   * Extend an existing FloorVizApp instance with VR capabilities.
   * @param {FloorVizApp} app
   */
  mixin(app) {
    Object.assign(app, FloorVizVR._methods);
    app._vr = {}; // namespace for all VR state
    console.log('[FloorVizVR] Mixin applied to FloorVizApp.');
  },

  _methods: {

    /* ── §12.1  INIT VR ───────────────────────────────────────────────── */

    /**
     * Entry point. Call after renderer and scene are ready.
     * Replaces the manual "Enter VR" button handler.
     */
    async initVR() {
      const supported = await VRCompat.isSupported();
      if (!supported) {
        VRCompat.showUnsupportedBanner();
        // Also mark the page button as unsupported
        const btn = document.getElementById('enterVR');
        if (btn) { btn.disabled = true; btn.title = 'WebXR immersive-vr not supported on this device'; }
        console.warn('[FloorVizVR] WebXR immersive-vr not supported.');
        return;
      }

      const app      = this;
      const renderer = this.renderer;
      const scene    = this.scene;
      const camera   = this.camera;

      // ── Enable XR on renderer
      renderer.xr.enabled = true;
      renderer.xr.setFramebufferScaleFactor(1.0);

      // ── Create the Three.js VRButton (hidden — our #enterVR drives it).
      //    VRButton is available as window.VRButton via the inline shim in index_v9.html.
      let _vrInternalBtn = document.getElementById('_three_vr_btn');
      if (!_vrInternalBtn) {
        _vrInternalBtn = VRButton.createButton(renderer);
        _vrInternalBtn.style.display = 'none';
      }

      // ── Enable and wire our custom #enterVR button
      const enterBtn = document.getElementById('enterVR');
      if (enterBtn) {
        enterBtn.disabled = false;
        // Remove any previous listener to avoid double-wiring
        const newBtn = enterBtn.cloneNode(true);
        enterBtn.parentNode.replaceChild(newBtn, enterBtn);
        newBtn.addEventListener('click', () => {
          if (renderer.xr.getSession()) {
            renderer.xr.getSession().end();
          } else {
            _vrInternalBtn.click();
          }
        });
      }

      // ── Build subsystems
      const rig        = new PlayerRig(renderer, scene, camera);
      const collision  = new VRCollision(scene);
      const fade       = new VRFade(camera);
      const teleport   = new VRTeleport(scene, rig, collision, fade);
      const smooth     = new VRSmoothLocomotion(rig, camera, collision);
      const perf       = new VRPerformance(renderer, scene);
      const tourAdapt  = new VRTourAdapter(app, rig, fade);
      const audio      = new VRAudio();

      // ── Store state
      app._vr = {
        rig, collision, fade, teleport, smooth, perf, tourAdapt, audio,
        movementMode:   'teleport',  // 'teleport' | 'smooth'
        inSession:      false,
        uiPanel:        null,        // built after controller connects
        controllers:    rig.controllers,
        teleportActive: [false, false],
      };

      // ── Controller event wiring
      app._setupVRControllers();

      // ── Session lifecycle
      renderer.xr.addEventListener('sessionstart', () => app._onVRSessionStart());
      renderer.xr.addEventListener('sessionend',   () => app._onVRSessionEnd());

      console.log('[FloorVizVR] VR subsystems initialised (r128, CDN mode).');
    },

    /* ── §12.2  SETUP CONTROLLERS ────────────────────────────────────── */

    _setupVRControllers() {
      const app = this;
      const vr  = app._vr;

      vr.controllers.forEach(({ controller, grip }, i) => {

        // ── Trigger pressed (start teleport aim OR confirm smooth)
        controller.addEventListener('selectstart', () => {
          if (vr.movementMode === 'teleport') {
            vr.teleportActive[i] = true;
          }
        });

        // ── Trigger released (execute teleport OR UI confirm)
        controller.addEventListener('selectend', () => {
          if (vr.movementMode === 'teleport' && vr.teleportActive[i]) {
            vr.teleportActive[i] = false;
            vr.teleport.execute();
          }
        });

        // ── Squeeze pressed (toggle VR UI panel on right controller)
        controller.addEventListener('squeezestart', () => {
          if (i === 1 && vr.uiPanel) {
            const vis = !vr.uiPanel.mesh.visible;
            vr.uiPanel.setVisible(vis);
          }
        });

        // ── Squeeze released (activate UI button)
        controller.addEventListener('squeezeend', () => {
          if (i === 0 && vr.uiPanel?.mesh.visible) {
            vr.uiPanel.activateHovered();
          }
        });

        // ── Controller connected — attach UI panel to left grip
        controller.addEventListener('connected', (ev) => {
          console.log(`[FloorVizVR] Controller ${i} connected:`, ev.data.targetRayMode);

          if (i === 0 && !vr.uiPanel) {
            vr.uiPanel = new VRUIPanel(grip, {
              onExit:          () => app._exitVR(),
              onNext:          () => app.nextRoom?.(),
              onPrev:          () => app.previousRoom?.(),
              onToggleMove:    () => app._toggleVRMovementMode(),
              onToggleMinimap: () => app._toggleMinimap?.(),
              onRecenter:      () => vr.rig.recenter(),
            });
            vr.uiPanel.setVisible(false);
          }
        });

        controller.addEventListener('disconnected', () => {
          console.log(`[FloorVizVR] Controller ${i} disconnected.`);
        });
      });
    },

    /* ── §12.3  SESSION LIFECYCLE ─────────────────────────────────────── */

    _onVRSessionStart() {
      const vr = this._vr;
      vr.inSession = true;
      vr.perf.enterVR();

      // Build tour hotspots + spatial data from loaded model
      if (this.model?.rooms) {
        vr.tourAdapt.buildHotspots(this.model.rooms, this.scene);
        vr.collision.rebuild(this.scene);
        vr.teleport.rebuild();
      }

      // Place player at first room's eye position (already set in PlayerRig constructor,
      // but re-apply here in case model loaded after rig was built)
      const firstRoom = window._tourRooms?.[0];
      if (firstRoom) {
        vr.rig.rig.position.set(firstRoom.eyeX ?? firstRoom.cx, 0, firstRoom.eyeZ ?? firstRoom.cz);
        // Face toward room centroid so user starts looking into the room
        const dx = firstRoom.cx - (firstRoom.eyeX ?? firstRoom.cx);
        const dz = firstRoom.cz - (firstRoom.eyeZ ?? firstRoom.cz);
        if (Math.abs(dx) > 0.01 || Math.abs(dz) > 0.01) {
          vr.rig.rig.rotation.y = Math.atan2(dx, dz);
        }
      }

      // Suspend manual orbit controls
      vr._orbitSuspended = true;

      // Hide 2D UI panels
      document.getElementById('ui-overlay')?.classList.add('vr-session-hidden');
      document.getElementById('tour-minimap')?.classList.add('vr-session-hidden');

      // Initialise audio
      vr.audio.init(this.camera);

      console.log('[FloorVizVR] VR session started — player spawned at first room.');
    },

    _onVRSessionEnd() {
      const vr = this._vr;
      vr.inSession = false;
      vr.perf.exitVR();

      // Re-enable manual orbit controls
      vr._orbitSuspended = false;

      // Restore 2D UI panels
      document.getElementById('ui-overlay')?.classList.remove('vr-session-hidden');
      document.getElementById('tour-minimap')?.classList.remove('vr-session-hidden');

      // Cancel any active teleport
      vr.teleport.cancel();

      console.log('[FloorVizVR] VR session ended.');
    },

    _exitVR() {
      const session = this.renderer.xr.getSession();
      if (session) session.end();
    },

    /* ── §12.4  MOVEMENT MODE TOGGLE ──────────────────────────────────── */

    _toggleVRMovementMode() {
      const vr = this._vr;
      vr.movementMode = vr.movementMode === 'teleport' ? 'smooth' : 'teleport';

      // Hide teleport arc when switching to smooth
      if (vr.movementMode === 'smooth') vr.teleport.cancel();

      console.log(`[FloorVizVR] Movement mode: ${vr.movementMode}`);
    },

    /* ── §12.5  ANIMATION LOOP INTEGRATION ───────────────────────────── */

    /**
     * Call this from your existing animate() loop INSTEAD of your current
     * renderer.render(scene, camera) call:
     *
     *   function animate() {
     *     renderer.setAnimationLoop(app.vrAnimationLoop.bind(app));
     *   }
     *
     * Or patch your existing loop:
     *
     *   function animate(time, frame) {
     *     ...existing non-VR updates...
     *     app.handleVRFrame(time, frame);
     *     renderer.render(scene, camera);
     *   }
     */
    handleVRFrame(time, frame) {
      const vr = this._vr;
      if (!vr || !vr.inSession) return;

      const delta    = this._vrClock?.getDelta?.() ?? 0.016;
      const session  = this.renderer.xr.getSession?.();
      const gamepads = session
        ? [...session.inputSources].map(s => s.gamepad)
        : [];

      // ── Teleport arc update
      if (vr.movementMode === 'teleport') {
        vr.controllers.forEach(({ controller }, i) => {
          if (vr.teleportActive[i]) {
            vr.teleport.update(controller);
          }
        });
      }

      // ── Smooth locomotion
      if (vr.movementMode === 'smooth') {
        vr.smooth.update(gamepads, delta);
      }

      // ── Gaze-based hotspot selection
      vr.tourAdapt.updateGaze(this.camera, delta);

      // ── UI panel ray hover update (left controller → UI on left grip)
      if (vr.uiPanel?.mesh.visible) {
        // Right controller (index 1) points at left-grip panel
        const rightCtrl = vr.controllers[1]?.controller;
        if (rightCtrl) vr.uiPanel.testRay(rightCtrl);
      }
    },

    /**
     * Full replacement animation loop for WebXR.
     * Use renderer.setAnimationLoop(app.vrAnimationLoop.bind(app))
     */
    vrAnimationLoop(time, frame) {
      if (!this._vrClock) this._vrClock = new THREE.Clock();

      const delta = this._vrClock.getDelta();

      // Run existing app update logic if present
      this._updateDesktopSystems?.(delta);

      // VR frame logic
      this.handleVRFrame(time, frame);

      // Render
      this.renderer.render(this.scene, this.camera);
    },

    /* ── §12.6  POST-MODEL-LOAD HOOK ──────────────────────────────────── */

    /**
     * Call this after renderModel() so VR systems pick up new geometry.
     * Patch your existing renderModel():
     *
     *   const _orig = app.renderModel.bind(app);
     *   app.renderModel = async (json) => {
     *     await _orig(json);
     *     app.onModelReady();
     *   };
     */
    onModelReady() {
      const vr = this._vr;
      if (!vr) return;

      vr.collision.rebuild(this.scene);
      vr.teleport.rebuild();

      if (this.model?.rooms) {
        vr.tourAdapt.buildHotspots(this.model.rooms, this.scene);
      }

      console.log('[FloorVizVR] VR subsystems updated after model load.');
    },
  },
};

/* ═══════════════════════════════════════════════════════════════════════════
   §13  EXPORT
═══════════════════════════════════════════════════════════════════════════ */

// ES module export — comment out if using plain <script> tag
// export { FloorVizVR, VR_CONFIG, VRCompat, PlayerRig, VRCollision,
//          VRTeleport, VRSmoothLocomotion, VRFade, VRUIPanel, VRAudio, VRPerformance, VRTourAdapter };

// UMD-compatible global exposure for vanilla JS
if (typeof window !== 'undefined') {
  window.FloorVizVR       = FloorVizVR;
  window.VR_CONFIG        = VR_CONFIG;
  window.VRCompat         = VRCompat;
  window.PlayerRig        = PlayerRig;
  window.VRCollision      = VRCollision;
  window.VRTeleport       = VRTeleport;
  window.VRSmoothLocomotion = VRSmoothLocomotion;
  window.VRFade           = VRFade;
  window.VRUIPanel        = VRUIPanel;
  window.VRAudio          = VRAudio;
  window.VRPerformance    = VRPerformance;
  window.VRTourAdapter    = VRTourAdapter;
}
