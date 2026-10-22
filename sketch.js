// ============================================================
//  Light Chaser · 手势聚光灯
//  p5.js + ml5.js HandPose（只识别单手）
//
//  交互：
//   1. 手掌移动         → 光斑跟随手腕（低通平滑 + 缓动，防抖动）
//   2. 手靠近/远离镜头  → z 深度映射光斑半径
//   3. 握拳             → 熄灭聚光灯（防抖）
//      摊开手掌         → 重新点亮
//   4. 点击按钮开启摄像头（满足浏览器安全策略）
//   5. 右下角 150×150 摄像头实时预览窗口（便于观察摄像头画面）
//
//  优化点：
//   - landmark 低通平滑 + 缓动跟随，抑制识别抖动
//   - 光斑目标值只在 updateLightFromHands() 中计算一次
//   - 半径平方预计算，范围判定不做 sqrt
//   - 光斑外碎片用极暗灰色绘制，亮灯/关灯之间平滑过渡
//   - 握拳 / 摊掌都需连续多帧才切换，中间状态不触发
//   - 长时间无手自动熄灯
//
//  【翻转一致性修正】
//   预览 <video>（#cam-preview）、模型输入、landmark 坐标三者统一为
//   “原始非镜像帧”：
//     · 预览 <video> 去掉 CSS 镜像（由 index.html 同步修改）
//     · ml5.handPose({ flipped: false }) —— 模型输入不再内部翻转
//     · landmark 坐标即原始视频像素坐标
//   只在画布映射的最后一步做一次水平镜像（x → 1 - x），
//   让画面符合“照镜子”直觉，避免多处翻转互相打架导致识别失败。
// ============================================================

let handPose = null;
let video = null;
let hands = [];

// ---- 运行状态 ----
let started = false;   // 是否已成功启动摄像头 + 模型
let poseReady = false; // HandPose 模型是否就绪

// ---- 光斑显示值 ----
let lightOn = true;
let lightX = 320;
let lightY = 240;
let lightRadius = 120;
let lightPower = 1; // 0~1 光照强度，用于开关灯的平滑过渡

// 目标值（由手势计算得到），实际值做缓动跟随
let targetX = 320;
let targetY = 240;
let targetRadius = 120;

// 缓动系数（0~1，越大越"跟手"）
const EASE_POS = 0.2;
const EASE_RADIUS = 0.1;
const EASE_POWER = 0.12;

// landmark 低通平滑系数（0~1，越小越平滑）
const SMOOTH_XY = 0.45;
const SMOOTH_Z = 0.1;

// 平滑后的手腕坐标缓存
let smoothWristX = 0;
let smoothWristY = 0;
let smoothWristZ = 0;
let hasSmooth = false;

// ---- z 深度映射范围（手靠近镜头 → z 更小 → 光斑更大）----
// 运行时可用键盘 [ ] 微调，见 keyPressed()
let zNear = -0.12;
let zFar = 0.1;
const RADIUS_MAX = 360;
const RADIUS_MIN = 60;

// ---- 碎片系统 ----
const FRAG_COUNT = 130; // 碎片数量（性能与观感平衡）
let fragments = [];

// ---- 手势防抖 ----
let fistCounter = 0; // 连续判定为握拳的帧数
let openCounter = 0; // 连续判定为摊掌的帧数
const FIST_FRAMES = 4; // 连续 4 帧握拳才关灯
const OPEN_FRAMES = 4; // 连续 4 帧摊掌才开灯
const NO_HAND_FRAMES = 45; // 约 0.75s 无手后自动熄灯
let noHandCounter = 0;

// ---- DOM ----
let hudEl, hudHandEl, hudLightEl, overlayEl, startBtnEl, camPreviewEl;
let hudTimer = 0;

// ============================================================
//  setup：只做准备，不碰摄像头（等用户点击）
// ============================================================
function setup() {
  createCanvas(windowWidth, windowHeight);
  pixelDensity(1); // 限制像素密度，避免高分屏下过载、提升帧率

  lightX = targetX = width / 2;
  lightY = targetY = height / 2;

  initFragments();
  cacheDomElements();

  // 绑定"开启摄像头"按钮（必须由用户点击，才会弹出授权）
  startBtnEl.addEventListener("click", startCamera, { once: true });
}

// ============================================================
//  DOM 缓存
// ============================================================
function cacheDomElements() {
  hudEl = document.getElementById("hud");
  hudHandEl = document.getElementById("hudHand");
  hudLightEl = document.getElementById("hudLight");
    overlayEl = document.getElementById("overlay");
  startBtnEl = document.getElementById("startBtn");
  camPreviewEl = document.getElementById("cam-preview"); // 右下角 150×150 实时预览
}

// ============================================================
//  摄像头 + 模型启动（必须在用户点击后调用）
// ============================================================
async function startCamera() {
  startBtnEl.disabled = true;
  startBtnEl.textContent = "正在请求摄像头…";

  // ---------- 1. 申请摄像头（触发浏览器授权弹窗）----------
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480, facingMode: "user" },
      audio: false,
    });
  } catch (err) {
    console.error("[Light Chaser] 摄像头启用失败：", err);
    startBtnEl.disabled = false;
    startBtnEl.textContent = "重新尝试开启摄像头";
    alert(
      "无法访问摄像头：\n" +
        (err && err.message ? err.message : err) +
        "\n\n请确认：\n" +
        "1. 页面通过 http://127.0.0.1:5500 打开（Live Server）\n" +
        "2. 浏览器地址栏左侧已允许摄像头权限"
    );
    return;
  }

  // 交给 p5 管理 video，然后隐藏（只作为检测输入，不显示画面）
  video = createCapture(stream);
  video.size(640, 480);
  video.hide();

    // 右下角 150×150 摄像头实时预览（方便观察摄像头拍到的画面）
  if (camPreviewEl) {
    camPreviewEl.srcObject = stream;
    camPreviewEl.play().catch(() => {});
    camPreviewEl.classList.add("visible");
  }

  startBtnEl.textContent = "正在加载手势模型…（首次约 10s）";

    // ---------- 2. HandPose 模型 ----------
  try {
    // flipped: false → 模型直接使用摄像头原始（非镜像）帧，
    // 与 #cam-preview 的 srcObject 画面完全一致，保证识别与预览匹配。
    // 镜像效果统一在 updateLightFromHands() 的画布映射中处理。
    handPose = await ml5.handPose({
      maxHands: 1, // 只识别单手，省算力
      flipped: false,
      modelType: "lite", // lite 模型更快
    });

    await waitForVideo(); // 等视频真正出帧再开始检测
    handPose.detectStart(video, gotHands);

    poseReady = true;
    started = true;

    // 淡出遮罩，显示 HUD
    overlayEl.classList.add("hidden");
    hudEl.classList.add("visible");
  } catch (err) {
    console.error("[Light Chaser] HandPose 初始化失败：", err);
    startBtnEl.disabled = false;
    startBtnEl.textContent = "重新尝试开启摄像头";
    alert(
      "手势模型加载失败，请检查网络（模型需从 CDN 下载）。\n" +
        (err && err.message ? err.message : err)
    );
  }
}

/** 等待视频元数据 + 首帧就绪 */
function waitForVideo(maxMs = 6000) {
  return new Promise((resolve) => {
    const t0 = performance.now();
    const check = () => {
      const el = video && video.elt;
      if ((el && el.readyState >= 2 && el.videoWidth > 0) || performance.now() - t0 > maxMs) {
        resolve();
      } else {
        setTimeout(check, 60);
      }
    };
    check();
  });
}

function draw() {
  background(0);

  // ---------- 1. 从手势更新目标值（只算目标，不直接改显示值） ----------
  if (started) updateLightFromHands();

  // ---------- 2. 缓动跟随：位置 / 半径 / 光强 ----------
  lightX = lerp(lightX, targetX, EASE_POS);
  lightY = lerp(lightY, targetY, EASE_POS);
  lightRadius = lerp(lightRadius, targetRadius, EASE_RADIUS);
  lightPower = lerp(lightPower, lightOn ? 1 : 0, EASE_POWER);

  // ---------- 3. 绘制碎片 + 光斑（ADD 叠加出光感） ----------
  blendMode(ADD);
  drawFragments();
  if (lightPower > 0.01) drawLight(lightPower);
  blendMode(BLEND);

  drawVignette();
  updateHUD();
}

// ============================================================
//  从手势更新目标值
// ============================================================
function updateLightFromHands() {
  if (!hands || hands.length === 0) {
    // 没有手：保持上一次位置；长时间无手则自动熄灯
    noHandCounter++;
    fistCounter = 0;
    openCounter = 0;
    if (noHandCounter > NO_HAND_FRAMES) lightOn = false;
    return;
  }

  noHandCounter = 0;

  const lm = hands[0].landmarks; // 21 个关键点
  if (!lm || lm.length < 21) return;

  const wrist = lm[0];

  // ---------- 低通平滑，抑制 landmark 抖动 ----------
  if (!hasSmooth) {
    smoothWristX = wrist.x;
    smoothWristY = wrist.y;
    smoothWristZ = wrist.z || 0;
    hasSmooth = true;
  } else {
    smoothWristX += (wrist.x - smoothWristX) * SMOOTH_XY;
    smoothWristY += (wrist.y - smoothWristY) * SMOOTH_XY;
    smoothWristZ += ((wrist.z || 0) - smoothWristZ) * SMOOTH_Z;
  }

    // ---------- 位置：视频坐标 → 画布坐标（cover 映射，不变形）----------
  // handPose 使用 flipped:false，landmark 坐标为原始（非镜像）视频像素坐标。
  // 这里做一次水平镜像（x → 1 - x），使光斑移动方向符合“照镜子”直觉，
  // 且与去掉 CSS 镜像的预览窗画面一一对应。
  const vw = 640;
  const vh = 480;
  const scale = Math.max(width / vw, height / vh);
  const offsetX = (width - vw * scale) / 2;
  const offsetY = (height - vh * scale) / 2;

  const mirroredX = vw - smoothWristX; // 镜像后的视频 x（像素）

  targetX = clamp(mirroredX * scale + offsetX, 0, width);
  targetY = clamp(smoothWristY * scale + offsetY, 0, height);

  // ---------- 半径：z 深度映射（z 越小越靠近镜头 → 光斑越大）----------
  const zNorm = constrain(map(smoothWristZ, zNear, zFar, 1, 0), 0, 1);
  const eased = zNorm * zNorm * (3 - 2 * zNorm); // smoothstep，中段更敏感
  targetRadius = lerp(RADIUS_MIN, RADIUS_MAX, eased);

  // ---------- 手势：握拳关灯 / 摊掌开灯（带防抖）----------
  const folded = countFoldedFingers(lm);

  if (folded >= 4) {
    fistCounter++;
    openCounter = 0;
    if (fistCounter >= FIST_FRAMES) lightOn = false;
  } else if (folded <= 1) {
    openCounter++;
    fistCounter = 0;
    if (openCounter >= OPEN_FRAMES) lightOn = true;
  } else {
    // 中间状态（半握）：不切换，避免抖动误判
    fistCounter = 0;
    openCounter = 0;
  }
}

/**
 * 统计收起的手指数量（食指 / 中指 / 无名指 / 小指）
 * 判定：指尖到手腕的距离小于 PIP 到手腕的距离 → 该手指弯曲
 * 相比单纯比较 y 坐标，这种方法对手的旋转 / 倾斜更鲁棒
 */
function countFoldedFingers(lm) {
  const tips = [8, 12, 16, 20];
    const pips = [6, 10, 14, 18];
  const wrist = lm[0];
  let folded = 0;

  for (let i = 0; i < 4; i++) {
    const tip = lm[tips[i]];
    const pip = lm[pips[i]];
    if (!tip || !pip) continue;
    const dTip = dist2(tip, wrist);
    const dPip = dist2(pip, wrist);
    // 指尖被"折"回手掌方向 → 到手腕距离小于 PIP
    if (dTip < dPip * 1.05) folded++;
  }
  return folded;
}

function dist2(a, b) {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return dx * dx + dy * dy;
}

// ============================================================
//  碎片初始化
// ============================================================
function initFragments() {
  fragments = [];
  for (let i = 0; i < FRAG_COUNT; i++) {
    fragments.push({
      x: random(width),
      y: random(height),
      vx: random(-0.28, 0.28),      // 缓慢漂移速度
      vy: random(-0.28, 0.28),
      size: random(1.2, 4.2),       // 碎片大小
      baseAlpha: random(140, 255),  // 基础亮度
      phase: random(TWO_PI),        // 闪烁相位（错开呼吸节奏）
      tint: random(-1, 1),          // 轻微色偏
    });
  }
}

// ============================================================
//  绘制碎片
//  - 光斑内：被点亮，随光晕呼吸闪烁，略微变大
//  - 光斑外：暗灰色，安静沉在黑暗里
// ============================================================
function drawFragments() {
  noStroke();

  const r = lightRadius;
  const r2 = r * r;              // 预计算半径平方，省去每帧 sqrt
  const t = frameCount * 0.045;  // 呼吸时间
  const on = lightPower;

  for (let i = 0; i < fragments.length; i++) {
    const f = fragments[i];

    // --- 漂移 ---
    f.x += f.vx;
    f.y += f.vy;
    // 边缘循环，避免出屏后消失
    if (f.x < -20) f.x += width + 40;
    if (f.x > width + 20) f.x -= width + 40;
    if (f.y < -20) f.y += height + 40;
    if (f.y > height + 20) f.y -= height + 40;

    const dx = f.x - lightX;
    const dy = f.y - lightY;
    const d2 = dx * dx + dy * dy;

    if (d2 > r2) {
      // ---- 光斑外：暗灰色，随开/关灯极轻微过渡 ----
      const a = lerp(10, 32, on);
      fill(96, 94, 86, a);
      circle(f.x, f.y, f.size);
      continue;
    }

    // ---- 光斑内：被照亮 ----
    const d = Math.sqrt(d2) / r;   // 0=中心，1=边缘
    const falloff = 1 - d;
    const soft = falloff * falloff; // 平方衰减，光晕更柔和

    // 光斑内的"呼吸"闪烁
    const twinkle = 0.72 + 0.28 * Math.sin(t + f.phase + d * 3.0);

    const alpha = f.baseAlpha * soft * twinkle * on;
    if (alpha < 1) continue;

    // 被光晕照亮的碎片略微变大
    const s = f.size * (1 + soft * 0.8);

    // 色温：中心偏暖白，边缘偏琥珀
    const cr = 255;
    const cg = lerp(232, 250, soft) + f.tint * 4;
    const cb = lerp(186, 234, soft);

    fill(cr, cg, cb, alpha);
    circle(f.x, f.y, s);

    // 少量较大碎片加一圈微光，增强体积感
    if (s > 3.2 && soft > 0.6) {
      fill(cr, cg, cb, alpha * 0.1);
      circle(f.x, f.y, s * 3.2);
    }
  }
}

// ============================================================
//  绘制柔光层：多层半透明同心圆模拟径向光晕（无需 shader，性能好）
//  power：0~1 光照强度，开关灯平滑过渡
// ============================================================
function drawLight(power) {
  noStroke();
  const cx = lightX;
  const cy = lightY;
  const r = lightRadius;

  fill(255, 240, 200, 10 * power);
  circle(cx, cy, r * 3.0);

  fill(255, 244, 208, 16 * power);
  circle(cx, cy, r * 2.0);

  fill(255, 248, 220, 22 * power);
  circle(cx, cy, r * 1.35);

  fill(255, 250, 232, 30 * power);
  circle(cx, cy, r * 0.85);

  fill(255, 252, 240, 40 * power);
  circle(cx, cy, r * 0.45);
}

// ============================================================
//  暗角：四周压暗，突出光斑
// ============================================================
function drawVignette() {
  noFill();
  const big = Math.max(width, height);
  for (let i = 0; i < 3; i++) {
    const rr = big * (0.62 + i * 0.18);
    stroke(0, 22);
    strokeWeight(big * 0.22);
    circle(width / 2, height / 2, rr * 2);
  }
  noStroke();
}

// ============================================================
//  HUD 状态提示（节流，减少 DOM 操作）
// ============================================================
function updateHUD() {
  hudTimer = (hudTimer + 1) % 12;
  if (hudTimer !== 0 || !started) return;

  const hasHand = hands && hands.length > 0;
  hudHandEl.textContent = hasHand ? "● 已检测到手" : "○ 未检测到手";
  hudHandEl.style.color = hasHand ? "#cfe8b0" : "#7d7768";

  hudLightEl.textContent = lightOn ? "● 灯光：开" : "● 灯光：关";
  hudLightEl.className = lightOn ? "on" : "off";
}

// ============================================================
//  工具
// ============================================================
function clamp(v, lo, hi) {
  return v < lo ? lo : v > hi ? hi : v;
}

// ml5 回调（在检测线程中触发）
function gotHands(results) {
  hands = results || [];
}

// 窗口尺寸变化：同步画布并重建碎片，避免碎片跑出可视区
function windowResized() {
  resizeCanvas(windowWidth, windowHeight);
  initFragments();
  lightX = targetX = width / 2;
  lightY = targetY = height / 2;
  hasSmooth = false;
}

// ============================================================
//  键盘调试
//   [ / ]  微调 z 深度映射范围（校准远近灵敏度）
//   SPACE  手动切换灯光
// ============================================================
function keyPressed() {
  // 避免空格键触发页面滚动
  if (key === " ") {
    lightOn = !lightOn;
    return false;
  }

  if (key === "[") {
    zNear -= 0.01;
    zFar -= 0.01;
    console.log("z 范围:", zNear.toFixed(2), "→", zFar.toFixed(2));
  } else if (key === "]") {
    zNear += 0.01;
    zFar += 0.01;
    console.log("z 范围:", zNear.toFixed(2), "→", zFar.toFixed(2));
  }
}
