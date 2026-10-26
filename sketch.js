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
//   5. 右下角 160×120 摄像头实时预览窗（带关键点骨架，便于自检）
//   6. 快捷键：1 前置 · 2 后置 · 3 切换镜像 · ← / → 切换摄像头
//
//  ------------------------------------------------------------
//  【本版重点修复 1：深度取值】
//   ml5 1.x 的结果结构（源码 addKeypoints 已确认）：
//     hand.keypoints    → 长度 21 的数组，每项 {x, y, name}（像素坐标）
//     hand.keypoints3D  → 长度 21 的数组，z 以“手腕宽度”为单位，
//                         通常为负，手越靠近镜头越趋近 0（≈ -0.03 ~ -0.15）
//     hand.wrist / hand.index_finger_tip … → 兼容别名（指向同一个点）
//   旧代码取 keypoints[0].z（像素层没有真实深度）→ 半径恒定 / 乱跳。
//   现在改用 keypoints3D[0].z，并校正了映射方向。
//
//  【本版重点修复 2：检测线程不能丢帧】
//   ml5 的 detectStart 内部是 `estimateHands → 回调 → await tf.nextFrame()`
//   串行循环。回调里只要抛一次异常，整个循环就永久停摆（症状：识别到
//   一次手之后再也不更新，或者一帧都不出）。现在回调里只做“接收 + 存
//   引用”，手势判断 / 平滑 / 映射全部放到 draw() 里，并加 try/catch 兜底。
//
//  【本版重点修复 3：摄像头选择】
//   facingMode 只给一个值 "user" 在某些设备（虚拟摄像头、无前置的一体机、
//   手机外接）会抛 OverconstrainedError，表现为“摄像头启动失败”。
//   → 多级降级 + 摄像头列表切换（1 / 2 / ← / →）。
//
//  【本版重点修复 4：模型就绪】
//   ml5.handPose() 返回“外壳对象”，内部 this.model 异步创建；就绪前调用
//   detectStart 会报 “Cannot read properties of null (reading 'estimateHands')”。
//   → 全程 await pose.ready + 轮询内部 model，确认后才开始检测。
// ============================================================

let handPose = null;
let video = null;

// ---- 检测结果 ----
let hands = [];         // 最近一次原始结果（只读，不在绘制中修改）
let lastHandCount = -1; // 手指数变化（HUD 用）

// ---- 检测循环健康检查 ----
let detectCallbackCount = 0;
let usingManualDetect = false;
let lastDetectAt = 0;

// ---- 运行状态 ----
let started = false;      // 摄像头 + 模型 + 检测循环全部就绪
let poseReady = false;    // 内部 model 已创建
let loadingModel = false; // 正在加载（屏幕提示）

// ---- 摄像头 ----
let facingMode = "user";
let useMirror = true; // 光斑 / 预览是否镜像（“照镜子”直觉）
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
