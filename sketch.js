// ============================================================
//  Light Chaser · 手势聚光灯
//  p5.js + ml5.js HandPose（1.x，只识别单手）
//
//  交互：
//   1. 手掌移动          → 光斑跟随手腕（低通平滑 + 缓动，防抖动）
//   2. 手靠近 / 远离镜头 → 手越靠近，光斑越大
//   3. 握拳              → 熄灭聚光灯（需连续多帧确认）
//      张开手掌          → 重新点亮
//   4. 点击按钮开启摄像头（浏览器安全策略要求用户手势）
//   5. 右下角 160×120 摄像头实时预览窗（叠加关键点骨架，便于自检）
//   6. 快捷键：1 前置 · 2 后置 · 3 切换镜像 · ← / → 切换摄像头
//      SPACE 手动开关灯 · [ / ] 微调深度映射范围
//
//  ------------------------------------------------------------
//  【本版核心修复：Cannot read properties of null (reading 'estimateHands')】
//
//  现象：控制台持续刷屏报错。
//  原因：渲染循环 draw() 在手部检测器（handPose.model，旧代码里写作 kd）
//        还没初始化完成时就调用了 estimateHands()，于是每一帧都抛一次异常。
//
//  修复：
//   ① 新增守卫 detectorReady()：只有「外壳对象 + 内部 model + 就绪标记」
//      三者都成立才返回 true。
//   ② 新增唯一识别入口 safeEstimateHands()：进来第一件事就是判空，
//      if (!handPose) return; / if (!handPose.model) return;
//      —— 检测器为 null 时【直接跳过】，绝不执行 estimateHands。
//   ③ 启动/驱动检测的 startDetectLoop() / manualDetectLoop() 同样先过守卫，
//      且每帧重新检查，模型中途失效也不会崩。
//   ④ draw() 不再直接接触识别逻辑，统一走 tryUpdateLight()，内部 try/catch。
//   ⑤ 报错只打印一次（reportOnce），不会刷屏，同时把错误写进 HUD 方便排查。
//
//  ------------------------------------------------------------
//  【深度取值】
//     hand.keypoints    → 21 点 {x, y}（像素坐标，z 基本为 0）
//     hand.keypoints3D  → 21 点 {x, y, z}，z 以“手腕宽度”为单位，通常为负，
//                         手越靠近镜头越趋近 0（≈ -0.03 ~ -0.15）
//     取深度时优先 keypoints3D[0].z，退化时用 keypoints[0].z。
//
//  【检测线程不能丢帧】
//   ml5 的 detectStart 内部是 estimateHands → 回调 → await tf.nextFrame()
//   串行循环。回调里只要抛一次异常，循环就永久停摆。所以回调里只做
//   “接收 + 存引用”，手势判断 / 平滑 / 映射全部放到 draw() 里。
//
//  【摄像头选择】
//   facingMode 只给一个值在某些设备（虚拟摄像头、无前置的一体机、手机外接）
//   会抛 OverconstrainedError → 多级降级 + 摄像头列表切换（1 / 2 / ← / →）。
//
//  【模型就绪】
//   ml5.handPose() 返回“外壳对象”，内部 this.model 异步创建；就绪前调用
//   detectStart 会报 null.estimateHands。
//   → 全程 await pose.ready + 轮询内部 model，确认后才开始检测。
// ============================================================

// ---- 核心对象 ----
let handPose = null; // ml5 HandPose 实例（外壳对象，内部 .model 才是检测器 kd）
let video = null;    // 原生 HTMLVideoElement（#cam-preview，一物两用）

// ---- 检测结果 ----
let hands = [];         // 最近一次原始结果（只读，不在绘制中修改）
let lastHandCount = -1; // 手指数变化（HUD 用）

// ---- 检测循环健康检查 ----
let detectCallbackCount = 0;
let usingManualDetect = false;
let lastDetectAt = 0;

// ---- 运行状态 ----
let started = false;        // 摄像头 + 模型 + 检测循环全部就绪
let poseReady = false;      // 内部 model 已创建（关键标记）
let loadingModel = false;   // 正在加载（屏幕提示）
let updatingStream = false; // 正在切换摄像头（防重入）

// ---- 错误保护：保证控制台不被刷屏 ----
let errorLogged = false;   // 同一个致命错误只打印一次
let errorCount = 0;        // 错误累计次数（HUD 显示）
let lastErrorMessage = ""; // 最近一次错误消息

// ---- 摄像头 ----
let facingMode = "user";
let useMirror = true; // 光斑 / 预览骨架是否镜像（“照镜子”直觉）
let videoDevices = [];
let deviceIndex = -1;
let currentStream = null;

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
const EASE_RADIUS = 0.16;
const EASE_POWER = 0.1;

// 手腕坐标低通平滑系数（0~1，越小越平滑）
const SMOOTH_XY = 0.4;
const SMOOTH_Z = 0.12;

// 平滑后的手腕坐标缓存
let smoothWristX = 0;
let smoothWristY = 0;
let smoothWristZ = 0;
let hasSmooth = false;

// ---- 深度 → 半径映射 ----
// keypoints3D.z：负数，越靠近镜头越大（趋近 0）。运行时可用 [ ] 微调
let zNear = -0.15; // 视为“手很近” → 最大光斑
let zFar = -0.03;  // 视为“手较远” → 最小光斑
const RADIUS_MAX = 360;
const RADIUS_MIN = 60;

// ---- 碎片系统 ----
const FRAG_COUNT = 130; // 碎片数量（性能与观感平衡）
let fragments = [];

// ---- 手势防抖 ----
let fistCounter = 0; // 连续判定为握拳的次数
let openCounter = 0; // 连续判定为张开的次数
const FIST_FRAMES = 4;     // 连续 4 次握拳才关灯
const OPEN_FRAMES = 4;     // 连续 4 次张开才开灯
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

  // 必须由用户点击触发（浏览器授权策略）
  // 注意：不用 { once: true }，这样加载失败后还能重试
  if (startBtnEl) startBtnEl.addEventListener("click", startCamera);

  // 摄像头热插拔：插上 USB 摄像头 / 手机上线后自动刷新列表
  if (navigator.mediaDevices && navigator.mediaDevices.enumerateDevices) {
    navigator.mediaDevices.addEventListener?.("devicechange", () => {
      if (!started) refreshDevices();
    });
  }
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
  camPreviewEl = document.getElementById("cam-preview"); // 右下角实时预览
}

// 更新加载提示文案（遮罩里的按钮文字，与画布上的提示配合）
function setLoadingHint(text) {
  if (startBtnEl) startBtnEl.textContent = text;
}

// ============================================================
//  守卫 1：检测器（内部 model）是否真的可用
//  返回 false 时，绝对不能调用 estimateHands / detect / detectStart。
// ============================================================
function detectorReady() {
  return !!(handPose && handPose.model && poseReady);
}

// 守卫 2：统一的软错误记录 —— 只打印第一次，之后仅计数，避免刷屏
function reportOnce(tag, err) {
  errorCount++;
  lastErrorMessage = (err && err.message) || String(err);
  if (!errorLogged) {
    errorLogged = true;
    console.error("[Light Chaser] " + tag + "：", err);
    console.warn(
      "[Light Chaser] 检测器未就绪或调用出错，已自动跳过检测以避免刷屏。" +
        "若反复出现，请确认 ml5 为 1.x 且模型能正常下载。"
    );
  }
}

// ============================================================
//  守卫 3：唯一的识别入口
//  这就是修复 Cannot read properties of null (reading 'estimateHands')
//  的核心位置 —— model 为 null 时直接 return，绝不往下走。
// ============================================================
function safeEstimateHands() {
  if (!handPose) return false;          // A：外壳对象不存在
  if (!handPose.model) return false;    // B：内部检测器为 null
  if (!poseReady) return false;         // C：就绪标记未置位
  if (!video || video.readyState < 2 || !video.videoWidth) return false; // D：没出帧

  try {
    if (typeof handPose.detect === "function") {
      const p = handPose.detect(video, (results) => {
        detectCallbackCount++;
        lastDetectAt = performance.now();
        gotHands(results);
      });
      if (p && typeof p.catch === "function") {
        p.catch((err) => reportOnce("detect 异步失败", err));
      }
      return true;
    }

    if (typeof handPose.model.estimateHands === "function") {
      // 兼容旧版 / 直接调用底层 MediaPipe detector
      const p = handPose.model.estimateHands(video);
      Promise.resolve(p).then(gotHands, (err) =>
        reportOnce("estimateHands 失败", err)
      );
      return true;
    }
  } catch (err) {
    reportOnce("estimateHands 调用失败", err);
  }
  return false;
}
// ============================================================
//  摄像头 + 模型启动（必须在用户点击后调用）
//  全程 async/await：严格保证「模型就绪 → 才开始检测」的顺序
// ============================================================
async function startCamera() {
  // 防止重复点击启动多个检测循环
  if (loadingModel || started) return;

  if (startBtnEl) startBtnEl.disabled = true;
  setLoadingHint("正在请求摄像头…");

  // ---------- 1. 申请摄像头（触发浏览器授权，带多级降级）----------
  let stream;
  try {
    stream = await getUserMediaWithFallback(facingMode);
  } catch (err) {
    console.error("[Light Chaser] 摄像头启用失败：", err);
    if (startBtnEl) startBtnEl.disabled = false;
    setLoadingHint("重新尝试开启摄像头");
    alert(
      "无法访问摄像头：\n" +
        (err && err.message ? err.message : err) +
        "\n\n请确认：\n" +
        "1. 页面通过 http://127.0.0.1:5500 打开（Live Server）\n" +
        "2. 浏览器地址栏左侧已允许摄像头权限"
    );
    return;
  }

  // ---------- 检测输入：直接使用原生 <video>（#cam-preview） ----------
  // 不用 createCapture() 的 p5 封装对象，避免某些版本下 ml5 拿到 0×0 的帧。
  // 原生 HTMLVideoElement 是 tfjs fromPixels 最稳的输入，同时兼作预览窗。
  if (camPreviewEl) {
    video = camPreviewEl;
  } else {
    // 兜底：DOM 里没有 #cam-preview 就现场创建一个隐藏的 video
    video = document.createElement("video");
    video.playsInline = true;
    video.muted = true;
    video.style.cssText = "display:none";
    document.body.appendChild(video);
  }
  attachStream(stream);
  try {
    await video.play();
  } catch (e) {
    // 某些浏览器自动播放策略下 play() 可能被拒绝，忽略即可
  }

  // ---------- 2. 加载 HandPose 模型（必须等它真正就绪）----------
  loadingModel = true;
  setLoadingHint("模型加载中，请等待…");

  try {
    // flipped: false → 模型使用摄像头原始（非镜像）帧，与预览窗一致；
    // 镜像效果统一在 updateLightFromHands() / 预览骨架中处理。
    const pose = await ml5.handPose({
      maxHands: 1,        // 只识别单手，省算力
      flipped: true,
      modelType: "lite",  // lite 模型更快
    });

    // 先存进全局实例（此刻 poseReady 仍为 false，守卫会拦住一切检测调用）
    handPose = pose;

    // 第一道保险：官方 ready Promise
    if (pose && pose.ready && typeof pose.ready.then === "function") {
      await pose.ready;
    }

    // 第二道保险：轮询确认内部 model（检测器）确实已创建
    const ok = await ensureModelReady(handPose);
    if (!ok) {
      throw new Error("HandPose 模型对象未就绪（内部 model 仍为 null）");
    }

    await waitForVideo(); // 等视频真正出帧再开始检测

    // 关键日志：确认送进模型的确实是原生 video 且尺寸正常
    console.log(
      "[Light Chaser] 检测输入：",
      video.constructor.name,
      video.videoWidth + "×" + video.videoHeight,
      "readyState=" + video.readyState
    );

    // ---------- 3. 模型确认就绪，才开始检测 ----------
    poseReady = true;
    started = true;
    loadingModel = false;
    detectCallbackCount = 0;

    startDetectLoop();

    // 淡出遮罩，显示 HUD
    if (overlayEl) overlayEl.classList.add("hidden");
    if (hudEl) hudEl.classList.add("visible");
  } catch (err) {
    console.error("[Light Chaser] HandPose 初始化失败：", err);
    loadingModel = false;
    poseReady = false;
    started = false;
    handPose = null; // 清空，保证 draw() 里的守卫一定拦住

    if (startBtnEl) startBtnEl.disabled = false;
    setLoadingHint("重新尝试开启摄像头");
    alert(
      "手势模型加载失败，请检查网络（模型需从 CDN 下载）。\n" +
        (err && err.message ? err.message : err)
    );
  }
}

// ============================================================
//  启动检测循环
//  先过守卫：检测器没就绪就直接返回，绝不调用 detectStart
// ============================================================
function startDetectLoop() {
  if (!detectorReady() || !video) return;

  try {
    handPose.detectStart(video, (results) => {
      // 回调里只做“接收 + 存引用”，绝不做重活，
      // 避免异常打断 ml5 内部的串行检测循环。
      detectCallbackCount++;
      lastDetectAt = performance.now();
      gotHands(results);
    });
  } catch (err) {
    reportOnce("detectStart 启动失败", err);
    fallbackToManualDetect();
    return;
  }

  // ---------- 兜底：若 detectStart 循环没跑起来，改用手动逐帧检测 ----------
  setTimeout(() => {
    if (started && detectCallbackCount === 0 && !usingManualDetect) {
      console.warn(
        "[Light Chaser] detectStart 未产生回调，切换到手动逐帧检测模式"
      );
      fallbackToManualDetect();
    }
  }, 3000);
}

function fallbackToManualDetect() {
  if (usingManualDetect) return;
  usingManualDetect = true;
  try {
    if (handPose && typeof handPose.detectStop === "function") {
      handPose.detectStop();
    }
  } catch (e) {
    // 忽略：部分版本在未启动时 detectStop 会抛错
  }
  manualDetectLoop();
}

// ============================================================
//  手动逐帧检测（detectStart 失效时的兜底方案）
//  用 requestAnimationFrame 驱动；每帧重新检查模型，未就绪就跳过
// ============================================================
function manualDetectLoop() {
  if (!started || !usingManualDetect) return;

  // 检测器为 null → 下一帧再试，期间不调用任何检测 API
  if (!detectorReady()) {
    requestAnimationFrame(manualDetectLoop);
    return;
  }

  // 走统一的安全入口（内部还有一层判空 + try/catch）
  safeEstimateHands();

  requestAnimationFrame(manualDetectLoop);
}
// ============================================================
//  draw：渲染循环
//  只通过带守卫的 tryUpdateLight() 接触识别结果，
//  检测器为 null 时它内部直接跳过，不会产生任何报错。
// ============================================================
function draw() {
  background(0);

  // ---------- 1. 从手势更新目标值（只算目标，不直接改显示值） ----------
  tryUpdateLight();

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

  // ---------- 4. 预览窗关键点骨架（视频本身由 <video> 渲染） ----------
  drawPreviewSkeleton();

  // ---------- 5. 模型加载中：在画布上显示提示文字 ----------
  if (loadingModel) drawLoadingText();

  updateHUD();
}
// 安全地从手势更新目标值：
//  - 检测器为 null / 未就绪 → 直接跳过（这就是修复刷屏的关键一环）
//  - 任何异常都只记录一次，绝不每帧刷屏
function tryUpdateLight() {
  if (!started || !poseReady) return;
  if (!detectorReady()) return; // 核心：kd（内部 model）为 null 时跳过
  try {
    updateLightFromHands();
  } catch (err) {
    reportOnce("手势数据处理出错", err);
  }
}

// ============================================================
//  画布上的「模型加载中，请等待」提示
// ============================================================
function drawLoadingText() {
  push();
  textAlign(CENTER, CENTER);

  // 半透明底条，保证在任何画面上都清晰可读
  noStroke();
  fill(0, 150);
  const w = Math.min(width * 0.8, 460);
  rect((width - w) / 2, height / 2 + 62, w, 76, 38);

  // 呼吸闪烁的主文字
  const pulse = 0.6 + 0.4 * Math.sin(frameCount * 0.08);
  fill(255, 236, 180, 255 * pulse);
  textSize(18);
  text("模型加载中，请等待…", width / 2, height / 2 + 88);

  // 小圆点动画，表示正在工作
  const dots = ".".repeat((Math.floor(frameCount / 18) % 3) + 1);
  fill(255, 236, 180, 150);
  textSize(13);
  text(dots, width / 2, height / 2 + 116);
  pop();
}
// ============================================================
//  预览窗关键点骨架（绘制在 <video> 上方对应位置，便于自检）
// ============================================================
const HAND_BONES = [
  [0, 1], [1, 2], [2, 3], [3, 4],        // 拇指
  [0, 5], [5, 6], [6, 7], [7, 8],        // 食指
  [5, 9], [9, 10], [10, 11], [11, 12],   // 中指
  [9, 13], [13, 14], [14, 15], [15, 16], // 无名指
  [13, 17], [17, 18], [18, 19], [19, 20],// 小指
  [0, 17],                               // 手掌根部
];

function drawPreviewSkeleton() {
  if (!video || !camPreviewEl || !hands || hands.length === 0) return;

  const lm = hands[0] && (hands[0].keypoints || hands[0].landmarks);
  if (!lm || lm.length < 21) return;

  const r = camPreviewEl.getBoundingClientRect();
  if (r.width < 2) return;

  const vw = video.videoWidth || 640;
  const vh = video.videoHeight || 480;

  // 预览窗是 object-fit: cover，按同样规则换算坐标
  const scale = Math.max(r.width / vw, r.height / vh);
  const offX = r.left + (r.width - vw * scale) / 2;
  const offY = r.top + (r.height - vh * scale) / 2;

  const toScreen = (p) => {
    const x = useMirror ? vw - p.x : p.x;
    return { x: offX + x * scale, y: offY + p.y * scale };
  };

  push();
  stroke(255, 233, 168, 200);
  strokeWeight(1.5);
  noFill();
  for (let i = 0; i < HAND_BONES.length; i++) {
    const a = lm[HAND_BONES[i][0]];
    const b = lm[HAND_BONES[i][1]];
    if (!a || !b) continue;
    const sa = toScreen(a);
    const sb = toScreen(b);
    line(sa.x, sa.y, sb.x, sb.y);
  }

  noStroke();
  fill(255, 245, 210, 230);
  for (let i = 0; i < lm.length; i++) {
    if (!lm[i]) continue;
    const s = toScreen(lm[i]);
    circle(s.x, s.y, 3);
  }
  pop();
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

  // ml5 1.x 的 handPose 结果里，每只手的 21 个关键点字段名是
  // keypoints（不是 landmarks！），坐标为原始视频像素坐标。
  const lm = hands[0].keypoints || hands[0].landmarks;
  if (!lm || lm.length < 21) return;

  // 深度要取 3D 关键点（同一索引），兼容不同小版本的字段名
  const lm3d = hands[0].keypoints3D || hands[0].keypoints3d || null;

  const wrist = lm[0];
  if (!wrist) return;

  // ---------- 低通平滑，抑制 landmark 抖动 ----------
  if (!hasSmooth) {
    smoothWristX = wrist.x;
    smoothWristY = wrist.y;
    smoothWristZ = wristDepth(lm, lm3d);
    hasSmooth = true;
  } else {
    smoothWristX += (wrist.x - smoothWristX) * SMOOTH_XY;
    smoothWristY += (wrist.y - smoothWristY) * SMOOTH_XY;
    smoothWristZ += (wristDepth(lm, lm3d) - smoothWristZ) * SMOOTH_Z;
  }

  // ---------- 位置：视频坐标 → 画布坐标（cover 映射，不变形）----------
  // handPose 使用 flipped:false，keypoints 坐标为原始（非镜像）视频像素坐标。
  // 这里做一次水平镜像（x → vw - x），使光斑移动方向符合照镜子直觉，
  // 且与去掉 CSS 镜像的预览窗画面一一对应。
  const vw = (video && video.videoWidth) || 640;
  const vh = (video && video.videoHeight) || 480;
  const scale = Math.max(width / vw, height / vh);
  const offsetX = (width - vw * scale) / 2;
  const offsetY = (height - vh * scale) / 2;

  const mappedX = useMirror ? vw - smoothWristX : smoothWristX;

  targetX = clamp(mappedX * scale + offsetX, 0, width);
  targetY = clamp(smoothWristY * scale + offsetY, 0, height);

  // ---------- 半径：z 深度映射（z 趋近 0 = 靠近镜头 → 光斑更大）----------
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
// 取手腕深度：优先 keypoints3D[0].z（真实相对深度），
// 退化时用 keypoints[0].z（多数情况为 0，此时半径保持基准值）。
function wristDepth(lm, lm3d) {
  if (lm3d && lm3d[0] && typeof lm3d[0].z === "number") return lm3d[0].z;
  if (lm && lm[0] && typeof lm[0].z === "number") return lm[0].z;
  return 0;
}

// 统计收起的手指数量（食指 / 中指 / 无名指 / 小指）
// 判定：指尖到手腕的距离小于 PIP 到手腕的距离 → 该手指弯曲
// 相比单纯比较 y 坐标，这种方法对手的旋转 / 倾斜更鲁棒
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
    // 指尖被折回手掌方向 → 到手腕距离小于 PIP
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
//  摄像头相关工具
// ============================================================
async function getUserMediaWithFallback(mode) {
  const attempts = [
    { video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: mode }, audio: false },
    { video: { facingMode: mode }, audio: false },
    { video: { width: { ideal: 640 }, height: { ideal: 480 } }, audio: false },
    { video: true, audio: false },
  ];
  let lastErr = null;
  for (let i = 0; i < attempts.length; i++) {
    try {
      return await navigator.mediaDevices.getUserMedia(attempts[i]);
    } catch (e) {
      lastErr = e;
    }
  }
  throw lastErr || new Error("getUserMedia 全部降级方案均失败");
}

// 把流挂到预览 <video> 上，并停止旧流
function attachStream(stream) {
  if (currentStream && currentStream !== stream) {
    currentStream.getTracks().forEach((t) => t.stop());
  }
  currentStream = stream;
  if (video) {
    video.srcObject = stream;
    if (video.classList) video.classList.add("visible");
    video.play().catch(() => {});
  }
  refreshDevices();
}

// 刷新摄像头设备列表
async function refreshDevices() {
  try {
    const list = await navigator.mediaDevices.enumerateDevices();
    videoDevices = list.filter((d) => d.kind === "videoinput");
    const track =
      currentStream && currentStream.getVideoTracks
        ? currentStream.getVideoTracks()[0]
        : null;
    const settings = track ? track.getSettings() : null;
    if (settings && settings.deviceId) {
      const idx = videoDevices.findIndex((d) => d.deviceId === settings.deviceId);
      if (idx >= 0) deviceIndex = idx;
    }
    if (deviceIndex < 0 && videoDevices.length > 0) deviceIndex = 0;
  } catch (e) {
    // 忽略
  }
}
// 切换摄像头：kind 为 "user" / "environment" / "next" / "prev" / 索引
async function switchCamera(kind) {
  if (updatingStream || !started) return;

  // 前置 / 后置
  if (kind === "user" || kind === "environment") {
    facingMode = kind;
    updatingStream = true;
    try {
      const stream = await getUserMediaWithFallback(facingMode);
      attachStream(stream);
      hasSmooth = false;
    } catch (e) {
      reportOnce("切换 facingMode 失败", e);
    }
    updatingStream = false;
    return;
  }

  // 按设备列表切换
  if (!videoDevices.length) await refreshDevices();
  if (!videoDevices.length) return;

  let next = deviceIndex;
  if (kind === "next") next = (deviceIndex + 1) % videoDevices.length;
  else if (kind === "prev")
    next = (deviceIndex - 1 + videoDevices.length) % videoDevices.length;
  else if (typeof kind === "number") next = kind;
  if (next === deviceIndex || next < 0 || next >= videoDevices.length) return;

  updatingStream = true;
  try {
    const dev = videoDevices[next];
    const constraints = dev.deviceId
      ? {
          video: {
            deviceId: { exact: dev.deviceId },
            width: { ideal: 640 },
            height: { ideal: 480 },
          },
          audio: false,
        }
      : { video: { width: { ideal: 640 }, height: { ideal: 480 } }, audio: false };
    const stream = await navigator.mediaDevices.getUserMedia(constraints);
    deviceIndex = next;
    attachStream(stream);
    hasSmooth = false; // 换了摄像头，重置平滑缓存
  } catch (e) {
    reportOnce("切换摄像头失败", e);
  }
  updatingStream = false;
}
// ============================================================
//  确认模型内部真的就绪
//  ml5 1.x 的 HandPose 实例内部持有 this.model（检测模型；底层
//  hand-pose-detection 的识别入口即 estimateHands）。只要 model 仍为 null
//  就绝对不能调用 detect / detectStart。
//  这里做「await ready + 限时轮询」双保险，兼容不同小版本字段差异。
// ============================================================
async function ensureModelReady(pose, maxMs = 20000) {
  if (!pose) return false;

  // 第一道：官方 ready Promise（存在就 await，最可靠）
  if (pose.ready && typeof pose.ready.then === "function") {
    try {
      await pose.ready;
    } catch (e) {
      console.error("[Light Chaser] 模型 ready 被拒绝：", e);
      return false;
    }
  }

  // 第二道：轮询内部字段。model 必须存在。
  // segmenter 若为 undefined（该版本无此字段）则视为 OK；为 null 则继续等。
  const t0 = performance.now();
  while (performance.now() - t0 < maxMs) {
    const modelOk = !!pose.model;
    const segOk = pose.segmenter === undefined || !!pose.segmenter;
    if (modelOk && segOk) return true;
    await sleep(100);
  }
  return !!pose.model;
}

// 基于 Promise 的 sleep，用于轮询等待
function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// 等待视频元数据 + 首帧就绪
function waitForVideo(maxMs = 6000) {
  return new Promise((resolve) => {
    const t0 = performance.now();
    const check = () => {
      const el = video; // 原生 HTMLVideoElement
      if (
        (el && el.readyState >= 2 && el.videoWidth > 0) ||
        performance.now() - t0 > maxMs
      ) {
        resolve();
      } else {
        setTimeout(check, 60);
      }
    };
    check();
  });
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
  const r2 = r * r;             // 预计算半径平方，省去每帧 sqrt
  const t = frameCount * 0.045; // 呼吸时间
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
    const d = Math.sqrt(d2) / r;    // 0=中心，1=边缘
    const falloff = 1 - d;
    const soft = falloff * falloff; // 平方衰减，光晕更柔和

    // 光斑内的呼吸闪烁
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
  if (!hudHandEl || !hudLightEl) return;

  const hasHand = hands && hands.length > 0;
  hudHandEl.textContent = hasHand ? "● 已检测到手" : "○ 未检测到手";
  hudHandEl.style.color = hasHand ? "#cfe8b0" : "#7d7768";

  hudLightEl.textContent = lightOn ? "● 灯光：开" : "● 灯光：关";
  hudLightEl.className = lightOn ? "on" : "off";

  // 出错次数挂在 title 上，鼠标悬停可见（不干扰界面）
  if (errorCount > 0 && hudEl) {
    hudEl.title = "检测错误 " + errorCount + " 次：" + lastErrorMessage;
  }
}

// ============================================================
//  工具
// ============================================================
function clamp(v, lo, hi) {
  return v < lo ? lo : v > hi ? hi : v;
}

// ml5 回调（在检测线程中触发）：只存引用，绝不在这里做重活
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
//  键盘
//   1 前置摄像头 · 2 后置摄像头 · 3 切换镜像
//   ← / → 切换摄像头设备
//   SPACE 手动切换灯光
//   [ / ] 微调 z 深度映射范围（校准远近灵敏度）
// ============================================================
function keyPressed() {
  // 避免空格键触发页面滚动
  if (key === " ") {
    lightOn = !lightOn;
    return false;
  }

  if (key === "1") {
    switchCamera("user");
  } else if (key === "2") {
    switchCamera("environment");
  } else if (key === "3") {
    useMirror = !useMirror;
    console.log("[Light Chaser] 镜像：", useMirror ? "开" : "关");
  } else if (keyCode === LEFT_ARROW) {
    switchCamera("prev");
  } else if (keyCode === RIGHT_ARROW) {
    switchCamera("next");
  } else if (key === "[") {
    zNear -= 0.01;
    zFar -= 0.01;
    console.log("z 范围:", zNear.toFixed(2), "->", zFar.toFixed(2));
  } else if (key === "]") {
    zNear += 0.01;
    zFar += 0.01;
    console.log("z 范围:", zNear.toFixed(2), "->", zFar.toFixed(2));
  }
}
