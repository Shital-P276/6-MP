# FloorViz — VR System README
### What it is, what was built, how to run it, and how to set up the Vive Pro 2

---

## Table of Contents
1. [What FloorViz Does](#1-what-floorviz-does)
2. [What Was Built (Version History)](#2-what-was-built)
3. [Bug Fixes in v11](#3-bug-fixes-in-v11)
4. [How the VR System Works](#4-how-the-vr-system-works)
5. [Controller Mapping (Gamepad API)](#5-controller-mapping)
6. [Running the Project Locally](#6-running-locally)
7. [Vive Pro 2 Setup Guide](#7-vive-pro-2-setup)
8. [Demo Options Without Base Stations](#8-demo-without-base-stations)
9. [Architecture Reference](#9-architecture-reference)

---

## 1. What FloorViz Does

FloorViz converts 2D floor plan files (DXF, PNG, PDF) into interactive 3D buildings you can walk through in a browser.

**Core pipeline:**
```
Floor plan file → FastAPI backend → 3D JSON model → Three.js viewer → VR walkthrough
```

**Features:**
- Upload DXF / PNG / PDF floor plan
- Automatic wall, room, door, window detection
- Blueprint and Realistic render modes with material editor
- Virtual tour system (fly between rooms)
- Full VR mode — works in browser with headset OR desktop mouse-look fallback
- Bluetooth gamepad/controller support (no base stations needed for input)

---

## 2. What Was Built

### v7 → v8 (VR Integration)
Added the complete VR system as a self-contained module onto the existing viewer:

| Component | What it does |
|---|---|
| `PlayerRig` | XR camera + controller hierarchy; moves player through world |
| `VRCollision` | AABB wall bounding boxes; prevents walking through walls, slides along surfaces |
| `VRTeleport` | Parabolic arc cast onto floor; glowing ring marker; fade + teleport on trigger release |
| `VRSmoothLocomotion` | Left joystick movement, right joystick 45° snap turns |
| `VRFade` | Black plane attached to camera; smooth fade-to-black on room transition |
| `VRUIPanel` | Canvas-rendered floating menu on left controller grip |
| `VRAudio` | Spatial audio architecture stub (ready for future room audio) |
| `VRPerformance` | Reduces shadow map resolution and pixel ratio inside VR for Quest frame rate |
| `VRTourAdapter` | Bridges existing tour system; gaze-based hotspot navigation in VR |

### v8 → v10 (VR System Rewrite)
Rewrote the VR module from scratch as `FloorViz VR System v2` to:
- Fix user-gesture timing issue with WebXR `requestSession` (must be called synchronously inside click handler)
- Add proper desktop fallback (mouse-drag look, no headset required)
- Add WebXR session management with `local-floor` reference space
- Add Bluetooth gamepad support via Gamepad API (Vive wands, Xbox, PS, any controller)
- Add in-headset PREV/NEXT nav overlay
- Add VR minimap
- Fix XR room navigation (moves scene position instead of camera, which is controlled by HMD)

### v10 → v11 (Bug Fixes — this version)
See section 3 below.

---

## 3. Bug Fixes in v11

### Bug 1: VR view only fills part of the screen
**Root cause:** The Three.js renderer canvas was sized correctly in pixels but its CSS `width/height` weren't set to `100%`, so in XR mode the browser's compositing layer clipped it.

**Fix:**
```css
#canvas-container canvas {
  display: block;
  width: 100% !important;
  height: 100% !important;
}
```
Also added a `_resizeForXR()` call inside `_vrRequestXRSession()` that forces `renderer.setSize(containerWidth, containerHeight)` before the XR session starts.

---

### Bug 2: Bluetooth notifications falling behind VR video feed
**Root cause:** `#toast-container` had `z-index: 1000`. The WebXR canvas compositor and its overlay elements sit at a higher stacking level in some browsers, pushing the toast behind.

**Fix:** Raised toast to `z-index: 99999` which sits above all page layers including the XR canvas mirror.

```css
#toast-container {
  z-index: 99999;
}
```

---

### Bug 3: Bluetooth controller not detected by the site
**Root cause (two parts):**

**Part A** — `_vrWatchGamepads` had `if (!VR.on) return` inside the `gamepadconnected` listener. This meant if the controller connected (and fired its event) before you clicked ENTER VR, the event was silently swallowed and the hint never updated.

**Part B** — The `gamepadconnected` event fires exactly once when the browser first sees a button press from a newly-connected gamepad. If your controller was already paired/connected before the page loaded, `navigator.getGamepads()` will return it, but the event won't fire again. So the page never knew it was there.

**Fix:**

1. Removed the `if (!VR.on) return` gate — the toast and hint now fire any time a controller connects, even before entering VR. This also lets you confirm Bluetooth is working.

2. Added `_vrScanExistingGamepads()` — called when you click ENTER VR. It calls `navigator.getGamepads()` immediately and shows a toast for any controller already connected. If none are visible yet (browser hasn't seen a button press), it prompts you to press any button on the controller.

```js
function _vrScanExistingGamepads() {
  const pads = navigator.getGamepads ? navigator.getGamepads() : [];
  // ... shows toast for each connected pad
  // ... if none found, prompts user to press a button
}
```

**Important browser quirk:** Even after Bluetooth pairing, `navigator.getGamepads()` returns `null` slots until the user physically presses at least one button on the controller. This is a browser security measure. So the flow is:
1. Pair controller via Windows Bluetooth
2. Open FloorViz in browser
3. Press any button on the controller → browser registers it → `gamepadconnected` fires → toast appears
4. Click ENTER VR → controller works immediately

---

## 4. How the VR System Works

### Two modes: Desktop and WebXR

**Desktop mode** (no headset):
- Camera stays at room eye-height (1.6m)
- Mouse drag on canvas = look left/right/up/down
- Arrow keys or click hotspots = change rooms
- Gamepad joystick = look; RB/LB = next/prev room
- Works in any browser, no extensions needed

**WebXR mode** (headset or Immersive Web Emulator):
- Room navigation works by *moving the scene* (offsetting `scene.position`), not the camera — because the HMD controls the camera
- `local-floor` reference space → Y=0 is floor, eye height comes from the headset
- In-headset PREV/NEXT buttons rendered as large HTML overlays on the canvas
- Gamepad still works for nav and look-offset

### Session lifecycle
```
Click ENTER VR
  └─ enterVR() [synchronous, inside click handler]
       ├─ Sets VR.on = true
       ├─ Shows HUD elements
       ├─ _vrCeiling(), _vrHotspots(), _vrPips() [build 3D elements]
       ├─ navigator.xr exists?
       │    YES → _vrRequestXRSession() [async, but called synchronously]
       │              └─ requestSession('immersive-vr') [must be inside gesture]
       │              └─ renderer.xr.setSession(session)
       │              └─ renderer.setAnimationLoop(_vrXRTick)
       │    NO  → _vrFly(0) [desktop fallback, rAF loop]
       └─ _vrScanExistingGamepads()
```

### Room navigation in XR
```
_vrXRNav(dir)
  └─ Calculates fFrom (current scene.position) and fTo (-room.eyeX, -room.eyeZ)
  └─ Sets VR.flying = true
  └─ _vrXRTick() lerps scene.position each frame (cubic ease)
  └─ HMD tracks head — FloorViz tracks rooms
```

---

## 5. Controller Mapping

### Any Bluetooth gamepad (Xbox, PS, generic)

| Action | Xbox Controller | PS Controller |
|---|---|---|
| Look left/right | Right stick X | Right stick X |
| Look up/down | Right stick Y | Right stick Y |
| Next room | RB (button 5) | R1 (button 5) |
| Previous room | LB (button 4) | L1 (button 4) |
| Next room (alt) | B button (btn 1) | Circle (btn 1) |
| Prev room (alt) | X button (btn 2) | Square (btn 2) |

### Vive Wand (Bluetooth only, no base stations)

| Action | Vive Wand |
|---|---|
| Look | Touchpad (axes 0, 1) |
| Next room | Touchpad right half + click |
| Previous room | Touchpad left half + click |
| Next room (alt) | Grip button |
| Prev room (alt) | Menu button |

### How to connect Vive Wands via Bluetooth (no SteamVR, no base stations)
1. Hold the **System button** (very bottom of wand) for 3 seconds → LED flashes rapidly
2. Windows Settings → Bluetooth & devices → Add device → Bluetooth
3. Select **"VIVE Controller MV"** (may appear twice for two wands)
4. Pair both wands
5. Open FloorViz in Edge/Chrome → press any button on the wand → toast appears confirming connection
6. Click ENTER VR → wands control look and room navigation

---

## 6. Running Locally

### Requirements
- Python 3.9+
- A modern browser (Edge, Chrome, Firefox)

### Start backend
```bash
cd your-project-folder
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### Serve frontend
You need a local HTTP server (opening the HTML file directly won't work for WebXR or API calls):

```bash
# Option A: Python
python -m http.server 3000

# Option B: Node
npx serve . -p 3000

# Option C: VS Code Live Server extension
# Right-click index_v11.html → Open with Live Server
```

Then open: `http://localhost:3000/index_v11.html`

### For the Immersive Web Emulator (desktop VR simulation)
1. Install the Chrome extension: [Immersive Web Emulator](https://chrome.google.com/webstore/detail/immersive-web-emulator/cgffilbpcibhmcfbgggfhfolhkfbhmik)
2. Open DevTools (F12) → find the **WebXR** tab
3. Click ENTER VR in FloorViz → the emulator takes over
4. Drag the headset gizmo in the DevTools panel to rotate the view

---

## 7. Vive Pro 2 Setup Guide

> ⚠️ **Without base stations:** The Vive Pro 2 has **zero head tracking**. It uses SteamVR Lighthouse tracking which requires the physical base station boxes. There is no inside-out fallback. The headset display still works (you can see it), but the view won't move when you move your head.

### What works without base stations
- Wearing the headset and seeing the VR view on the headset display
- Connecting the Vive wands via Bluetooth for joystick/button input
- Viewing FloorViz on the headset (static view, no head tracking)
- Navigating rooms using wand buttons

### Full setup (WITH base stations — if you borrow them)

#### Step 1: Install SteamVR
1. Install [Steam](https://store.steampowered.com/) → install SteamVR from the Steam store (free)
2. Plug in the Vive Link Box (the small black box with ports) via HDMI and USB to your laptop

#### Step 2: Mount and pair base stations
1. Place two Base Station 2.0 units on opposite corners of the room, above head height, angled down ~30°
2. Plug them into power — they start automatically (no pairing needed)
3. They need line-of-sight to each other and to the play area

#### Step 3: Pair headset and controllers
1. Open SteamVR → it detects the headset automatically via the Link Box
2. Hold System button on each wand for 3 seconds → SteamVR pairs them

#### Step 4: Room setup
1. SteamVR → Room Setup → Standing Only (simpler, good for demo)
2. Follow the on-screen instructions to define floor level

#### Step 5: Open FloorViz in the Vive browser
**Option A — Use the PC browser mirrored to headset:**
- The Vive Link Box mirrors your PC screen to the headset
- Open `http://localhost:3000/index_v11.html` in Edge/Chrome on your laptop
- Put on the headset — you see whatever is on your laptop screen
- This gives head tracking in WebXR mode (because SteamVR feeds tracking data to the browser)

**Option B — Use the built-in Vive browser:**
- Inside the headset, open the Vive home environment
- Use the browser app → navigate to `http://YOUR-PC-IP:3000/index_v11.html`
- Find your PC IP: `ipconfig` in Command Prompt → look for IPv4 Address (e.g. 192.168.1.5)
- Both PC and headset must be on the same WiFi network

#### Step 6: Enter VR
- In FloorViz, process a floor plan → click ENTER VR
- SteamVR feeds WebXR tracking data → full head tracking works
- Controller triggers/touchpads work for navigation

### Laptop compatibility check
The Vive Pro 2 requires a decent GPU. Minimum:
- NVIDIA GTX 1060 or AMD RX 480 or better
- USB 3.0 port
- DisplayPort or HDMI 1.4+

Run the [SteamVR Performance Test](https://store.steampowered.com/app/323910/SteamVR_Performance_Test/) to check before demo day.

---

## 8. Demo Without Base Stations

### Best option: Immersive Web Emulator on a big monitor
1. Connect laptop to TV/projector via HDMI
2. Open FloorViz in Chrome with the Immersive Web Emulator extension installed
3. Process your floor plan → click ENTER VR → emulator activates
4. Open DevTools (F12) → WebXR tab → drag the headset gizmo to rotate view live
5. Audience sees the first-person walkthrough on the big screen
6. Connect an Xbox/PS controller via Bluetooth → use joystick to look, buttons to change rooms

This looks genuinely impressive on a large display and has zero hardware risk.

### Alternative: Vive wands as presenters (no head tracking)
1. Pair Vive wands via Bluetooth (see Section 5)
2. Enter VR in desktop mode (FloorViz falls back to mouse-look)
3. Use wand touchpad to look around, buttons to change rooms
4. Person wearing headset sees the static view; you control it from the wands

---

## 9. Architecture Reference

### File structure
```
index_v11.html          ← Everything (single file — no build step)
  ├── CSS styles         lines 7–1070 approx
  ├── HTML body          lines 765–985
  ├── Three.js CDN       line 985
  ├── Main viewer JS     lines 986–2410
  │     ├── initThree()
  │     ├── buildModel()
  │     ├── Tour system
  │     └── Material editor
  └── VR System JS       lines 3208–3976
        ├── VR state (VR object)
        ├── enterVR()
        ├── _vrRequestXRSession()
        ├── _vrXRTick() / _vrTick()
        ├── _vrGamepadTick()
        ├── _vrScanExistingGamepads()
        ├── vrNav() / _vrFly() / _vrXRNav()
        ├── _vrLook()
        └── exitVR()
```

### Key globals
| Variable | Type | Purpose |
|---|---|---|
| `VR` | Object | All VR state (on, rooms, idx, flying, xrSession, xrMode…) |
| `VR.rooms` | Array | Copy of `window._vrRooms` — room centroids and eye positions |
| `GP` | Object | Gamepad config (deadzone, look speed, nav cooldown) |
| `FLY` | Number | Room-to-room fly animation speed (0.03 = ~1 second) |
| `EYE` | Number | Camera eye height in metres (1.6) |
| `scene.position` | Vector3 | In XR mode, rooms are navigated by shifting the scene, not the camera |

### VR rooms data format
```js
window._vrRooms = [
  {
    cx: 5.2,        // room centroid X (world space)
    cz: 3.8,        // room centroid Z
    eyeX: 5.0,      // camera eye position X (may be offset from centroid to avoid walls)
    eyeZ: 3.6,      // camera eye position Z
    label: 'Living Room',
    area: 18.4,
  },
  // ...
]
```

This is populated by `_buildTourRooms()` in the main tour system and read by `enterVR()`.
