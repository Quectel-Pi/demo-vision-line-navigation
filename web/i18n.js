(function () {
  "use strict";

  const STORAGE_KEY = "line-follow-ui-language";
  const DEFAULT_LANGUAGE = "en";
  const EN = {
    // Header and navigation
    "视觉循迹小车 · 调试台": "Vision Line-Following Car · Debug Console",
    "视觉循迹小车 · Web 调试台": "Vision Line-Following Car · Web Console",
    "相机画面与导航控制。进入页面时仅预览，不会自动驱动车轮。": "Camera view and navigation control. The page opens in preview only and never drives the wheels automatically.",
    "Config · 参数设置 →": "Config · Settings →",
    "Config · 视觉与循迹参数": "Config · Vision & Tracking Parameters",
    "保存后应用到小车；修改参数会停车，需手动重新启动导航。": "Changes are applied to the car after saving; changing parameters stops the car and requires a manual restart of navigation.",
    "← 返回主控制页": "← Back to Control",

    // Main sections and buttons
    "实时相机与识线叠加": "Live Camera & Line Overlay",
    "识别与运行状态": "Recognition & Runtime Status",
    "启动循迹": "Start Tracking",
    "停止电机": "Stop Motors",
    "摄像头回预设位": "Home Camera",
    "超声波避障": "Ultrasonic Avoidance",
    "下载本轮记录": "Download This Run",
    "正在连接小车…": "Connecting to the car…",
    "相机实时画面": "Live camera view",
    "画面连接中…": "Connecting video…",
    "画面连接已建立 · 断线自动重连": "Video connected · automatic reconnect",
    "画面连接中断，正在自动重连；旧画面不代表当前现场。": "Video disconnected; reconnecting automatically. The old image does not represent the current scene.",
    "页面在后台，返回后重新连接画面。": "The page is in the background; the video will reconnect when you return.",
    "状态连接中…": "Connecting status…",
    "状态连接中断，旧数据已失效；正在自动重连。": "Status connection lost; old data is invalid. Reconnecting automatically.",
    "默认关闭，正在读取设置…": "Disabled by default; loading settings…",
    "默认关闭。开启后，${preSlowCm} cm 起降速，距离 ≤ ${stopDistanceCm} cm 或测距不可用时停车。切换会停止当前行驶，设置自动保存。": "Disabled by default. When enabled, the car slows at ${preSlowCm} cm and stops at ≤ ${stopDistanceCm} cm or when distance is unavailable. Switching stops the current run and saves automatically.",
    "默认关闭，关闭时不执行避障转弯寻线。开启后，${preSlowCm} cm 起降速，距离 ≤ ${stopDistanceCm} cm 停车、原地左转约90°后前进寻线，未找到再原地右转寻线；找到稳定黑线后继续循迹，失败则停车。切换会停车，需重新点击导航，设置自动保存。": "Disabled by default; no avoidance turn/search runs while disabled. When enabled, the car slows at ${preSlowCm} cm, stops at ≤ ${stopDistanceCm} cm, turns left about 90° and searches forward; if no line is found it turns right and searches again. A stable line hands control back to tracking; failure stops the car. Switching stops the car, requires pressing Start again, and saves automatically.",
    "左：原图、ROI、扫描候选和选中路径；中：黑色阈值二值图及扫描质量；右：地面坐标图（仅有实测标定时）。扫描线按近端黄 / 中段紫 / 远端青区分，橙色为候选区间，绿色为 VALID 路径，紫色拟合线为图像坐标结果。": "Left: original image, ROI, scan candidates and selected path. Middle: black-threshold binary image and scan quality. Right: ground-coordinate plot (only with measured calibration). Scanlines are colored near=yellow, middle=purple, far=cyan; orange marks candidate intervals, green marks the VALID path, and the purple fit line is in image coordinates.",
    "IMU 反馈未启用。": "IMU feedback is not enabled.",
    "直角弯：未启用": "Right-angle turns: disabled",
    "本机测试记录：未启用": "Onboard test recording: disabled",
    "摄像头云台自由调节（点击展开）": "Camera gimbal free adjustment (click to expand)",
    "二维云台手动拖动": "2-axis gimbal manual drag",
    "拖动控制二维云台": "Drag to control the 2-axis gimbal",
    "反转水平": "Invert pan",
    "反转俯仰": "Invert tilt",
    "拖动控制云台；松开鼠标或手指后停止发送。方向相反时勾选对应反转项。移动云台会使地面标定失效，需重新确认后才能启动循迹。": "Drag to control the gimbal; releasing the mouse or finger stops sending commands. Tick the matching invert option if the direction is reversed. Moving the gimbal invalidates ground calibration; reconfirm it before starting tracking.",
    "正在拖动；靠近软限位时目标会自动限幅。机械限位未经验证，请勿长时间顶住端点。": "Dragging; the target is clamped near soft limits. Mechanical limits are unverified; do not hold against an endpoint for long.",
    "角度按参考程序线性映射（水平偏置 +15°，俯仰偏置 0°），括号内保留 PWM 脉宽。它是命令角估算，不是舵机角度反馈。车轮未使能时可拖动；方向相反可勾选反转。": "Angles are linearly mapped from the reference program (pan offset +15°, tilt offset 0°), with PWM pulse width in parentheses. These are estimated command angles, not servo feedback. Dragging is available while the wheels are disabled; tick invert if the direction is reversed.",
    "水平 —° · 俯仰 —°": "Pan —° · Tilt —°",
    "已发送（无角度反馈）": "Sent (no angle feedback)",
    "回位中": "Homing",
    "待回位": "Awaiting home",
    "手动位置（无角度反馈）": "Manual position (no angle feedback)",
    "未启用": "Disabled",
    "未就绪": "Not ready",
    "未运行": "Not running",
    "未知": "Unknown",
    "未检出": "Not detected",
    "未标定": "Uncalibrated",
    "未通过": "Not passed",
    "连接中…": "Connecting…",
    "未启用/无数据": "Disabled / no data",
    "最后命令为零（无反馈）": "Last command zero (no feedback)",
    "非零命令 / 状态未知": "Non-zero command / state unknown",
    "暗色候选 · 未确认": "Dim candidate · unconfirmed",
    "准备就绪": "Ready",
    "正常循迹": "Normal tracking",
    "直线": "Straight",
    "转向": "Turning",
    "接近弯心": "Approaching corner",
    "确认出口": "Confirming exit",
    "确认 L 弯": "Confirming L turn",
    "等待 L 弯": "Waiting for L turn",
    "出口已捕获": "Exit captured",
    "画面向右弯": "Curve to the right",
    "画面向左弯": "Curve to the left",
    "纯视觉 / 未启用": "Vision only / disabled",
    "纯视觉模式，IMU 采样和反馈均未启用。": "Vision-only mode; IMU sampling and feedback are disabled.",
    "仅观测：IMU 不修正电机输出。短窗口用于短时变化；累计角不是绝对航向。": "Observe only: the IMU does not correct motor output. The short window shows short-term change; cumulative angle is not an absolute heading.",
    "联合控制：视觉决定方向，IMU 反馈角速度；累计角不是绝对航向。": "Combined control: vision sets direction and the IMU feeds back yaw rate; cumulative angle is not an absolute heading.",

    // Runtime states
    "正在停车": "Stopping",
    "停车未确认": "Stop not confirmed",
    "仅预览": "Preview only",
    "等待黑线": "Waiting for line",
    "确认黑线": "Acquiring line",
    "循迹运行中": "Tracking",
    "丢线停车": "Line lost (stopped)",
    "障碍停车": "Obstacle stop",
    "等待距离数据": "Waiting for distance",
    "相机错误": "Camera error",
    "运行错误": "Runtime error",
    "已停止": "Stopped",
    "需地面标定": "Ground calibration required",
    "轨迹不确定": "Path uncertain",
    "重新确认轨迹": "Reacquiring path",
    "近端路径丢失停车": "Near path lost (stopped)",
    "直角弯接近": "Corner approach",
    "直角弯转向": "Corner turn",
    "直角弯出口确认": "Corner exit confirm",
    "等待弯心画面": "Waiting for corner view",
    "直角弯退出停车": "Corner abort (stopped)",
    "等待 IMU": "Waiting for IMU",
    "IMU 异常": "IMU error",
    "停车标志确认": "Stop marker confirm",
    "接近停车标志": "Approaching stop marker",
    "停车点对位": "Stop alignment",
    "已到停车标志": "Stop marker reached",
    "停车定位失败": "Stop positioning failed",
    "控制失败": "Control failure",
    "请求超时": "Request timeout",
    "记录错误": "Recording error",
    "正在记录": "Recording",
    "本轮已保存": "Run saved",
    "记录达到存储上限": "Recording storage limit reached",
    "正在保存结束记录": "Saving end record",
    "有效": "Valid",
    "不确定": "Uncertain",
    "丢线": "Lost",
    "通过": "Pass",
    "未就绪 / —": "Not ready / —",
    "联合控制 / READY": "Combined control / READY",
    "直角弯：等待 L 弯": "Right-angle turns: waiting for L turn",
    "例如：[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]": "Example: [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]",

    // Metric labels
    "电机命令状态（软件记录，无轮速反馈）": "Motor command state (software log, no wheel-speed feedback)",
    "电池电压": "Battery voltage",
    "运行状态": "Runtime state",
    "相机 / 帧率": "Camera / FPS",
    "云台回位状态": "Gimbal home state",
    "近端黑线": "Near line",
    "远端黑线": "Far line",
    "选中路径连续性": "Selected path continuity",
    "横向偏差": "Lateral error",
    "弯道预瞄输入（归一化）": "Curve preview input (normalized)",
    "远 / 中 / 近候选扫描行": "Far / mid / near candidate scanlines",
    "近端路径质量（行数 / 置信度）": "Near path quality (rows / confidence)",
    "车头参考点横向偏差（cm，右正）": "Front-reference lateral error (cm, right positive)",
    "最近有效观测距离（cm）": "Nearest valid observation (cm)",
    "车头参考点方向误差（°）": "Front-reference heading error (°)",
    "地面路径状态 / 外推距离": "Ground path state / extrapolation",
    "方向角补偿（归一化）": "Heading compensation (normalized)",
    "视觉转向输出（归一化）": "Vision steering output (normalized)",
    "最终转向输出（归一化）": "Final steering output (normalized)",
    "拟合轨迹中点（x, y）": "Fitted path midpoint (x, y)",
    "前进分量占空比（命令值，无速度反馈）": "Forward duty (command, no speed feedback)",
    "转向分量占空比（命令值，无速度反馈）": "Turn duty (command, no speed feedback)",
    "四轮占空比请求 FL / FR / RL / RR": "Wheel duty request FL / FR / RL / RR",
    "前方距离": "Forward distance",
    "巡线探头 1 / 2 / 3 / 4（1=低反射）": "Line sensors 1 / 2 / 3 / 4 (1=low reflection)",
    "前方方向（图像角度）": "Forward heading (image angle)",
    "弯向 / 图像曲率": "Turn direction / image curvature",
    "弯道速度倍率（软件命令比例）": "Curve speed scale (software command)",
    "中心线拟合 RMS": "Centerline fit RMS",
    "IMU 模式 / 数据状态": "IMU mode / data state",
    "陀螺仪 XYZ（°/s，未减零偏）": "Gyro XYZ (°/s, raw)",
    "实际转向角速度（°/s，左正右负）": "Actual yaw rate (°/s, left positive)",
    "期望角速度（°/s）": "Target yaw rate (°/s)",
    "角速度误差（°/s）": "Yaw-rate error (°/s)",
    "IMU 转向修正（归一化）": "IMU steering correction (normalized)",
    "短窗口相对转角（°）": "Short-window relative angle (°)",
    "本段累计相对角（°，仅诊断）": "Segment cumulative angle (°, diagnostics only)",
    "IMU 数据年龄": "IMU data age",
    "校准期加速度模长（g，尺度待核实）": "Calibration acceleration magnitude (g, scale unverified)",
    "视觉转向趋势（归一化，左正右负）": "Vision turn hint (normalized, left positive)",

    // Config page
    "视觉与循迹参数": "Vision & Tracking Parameters",
    "近端循迹与丢线保护": "Near Tracking & Line-Loss Protection",
    "黑线识别": "Line Detection",
    "控制参数": "Control Parameters",
    "二维云台 PWM 回位": "2-axis Gimbal PWM Home",
    "弯道速度控制（默认关闭）": "Curve Speed Control (disabled by default)",
    "方向和曲率诊断（仅显示）": "Heading & Curvature Diagnostics (display only)",
    "实际落地标定与胶带宽度范围": "Ground Calibration & Tape Width Range",
    "调试可视化（正式运行可关闭）": "Debug Visualization (can be disabled in normal runs)",
    "保存并应用参数": "Save & Apply",
    "正在保存…": "Saving…",
    "宽度": "Width",
    "高度": "Height",
    "平滑系数": "Smoothing factor",
    "灰度阈值": "Grayscale threshold",
    "请求超时": "Request timeout",
    "基础占空比": "Base duty",
    "扫描线数量": "Scanline count",
    "最低置信度": "Minimum confidence",
    "中心跳变惩罚": "Center-jump penalty",
    "宽度变化惩罚": "Width-change penalty",
    "帧间轨迹确认": "Inter-frame path confirmation",
    "弯道目标倍率": "Curve target scale",
    "最大横向斜率": "Maximum lateral slope",
    "直线速度倍率": "Straight speed scale",
    "稳定识线帧数": "Stable-line frames",
    "速度倍率上限": "Speed scale upper limit",
    "速度倍率下限": "Speed scale lower limit",
    "最大转向占空比": "Maximum turn duty",
    "最少有效扫描线": "Minimum valid scanlines",
    "移动时间（秒）": "Move time (s)",
    "稳定等待（秒）": "Settle wait (s)",
    "丢失后不确定帧数": "Uncertain frames after loss",
    "宽度匹配评分权重": "Width-match score weight",
    "扫描线与路径评分": "Scanline & path scoring",
    "扫描覆盖评分权重": "Scan-coverage score weight",
    "相邻宽度变化容差": "Adjacent width-change tolerance",
    "路径平滑评分权重": "Path-smoothness score weight",
    "宽度范围外附加容差": "Extra tolerance outside width range",
    "近端最少有效扫描线": "Minimum valid near scanlines",
    "仅近端可见时速度倍率": "Speed scale when only near is visible",
    "俯仰目标脉宽（μs）": "Tilt target pulse (μs)",
    "大偏差阈值（归一化）": "Large-error threshold (normalized)",
    "水平目标脉宽（μs）": "Pan target pulse (μs)",
    "俯仰舵机 PWM 通道": "Tilt servo PWM channel",
    "减速变化率（倍率/秒）": "Deceleration rate (scale/s)",
    "加速变化率（倍率/秒）": "Acceleration rate (scale/s)",
    "宽度样本参考值（mm）": "Tape width sample reference (mm)",
    "服务启动时回到指定位置": "Return to preset position on service start",
    "水平舵机 PWM 通道": "Pan servo PWM channel",
    "允许胶带最大宽度（mm）": "Maximum tape width (mm)",
    "允许胶带最小宽度（mm）": "Minimum tape width (mm)",
    "直线判定阈值（px⁻¹）": "Straightness threshold (px⁻¹)",
    "远端预瞄权重（0=关闭）": "Far preview weight (0=off)",
    "大偏差转向增益（1=关闭）": "Large-error steering gain (1=off)",
    "方向前馈最大贡献（归一化）": "Maximum heading-feedforward contribution (normalized)",
    "近端连续丢失后解除使能帧数": "Frames before disarming after continuous near loss",
    "强弯曲率标定阈值（px⁻¹）": "Strong-curvature threshold (px⁻¹)",
    "方向前馈增益 Kθ（每弧度）": "Heading-feedforward gain Kθ (per radian)",
    "近端 ROI（主要循迹区域）": "Near ROI (primary tracking area)",
    "远端 ROI（辅助观察区域）": "Far ROI (auxiliary observation area)",
    "PID 微分滤波 α（1=关闭）": "PID derivative filter α (1=off)",
    "PID 输出平滑 α（1=关闭）": "PID output smoothing α (1=off)",
    "最大中心变化速度（画面宽度/秒）": "Maximum center-change rate (image width/s)",
    "启用弯道速度调节（实车标定前保持关闭）": "Enable curve speed control (keep off before vehicle calibration)",
    "确认小车处于实际落地高度，且云台固定在循迹角度": "Confirm the car is at its actual ground height and the gimbal is fixed at the tracking angle",
    "地面梯形 ROI 四角（归一化 [x,y]，从远端左角顺时针）": "Ground trapezoid ROI corners (normalized [x,y], clockwise from far-left)",
    "自定义扫描线位置（0 到 1，从远端到近端；留空使用均匀分布）": "Custom scanline positions (0 to 1, far to near; leave blank for uniform spacing)",
    "胶带宽度样本（y_norm 和 width_norm=像素宽度/图像宽度，至少远/中/近三行并覆盖 ROI 全深度）": "Tape width samples (y_norm and width_norm = pixel width / image width; at least far/mid/near and covering the full ROI depth)",
    "远端用于预瞄；远端暂时不可见时，近端轨迹仍有效则降速跟随。近端轨迹连续无效达到设定帧数后停车并解除使能，避免线路恢复时自动重启。": "Far path is used for preview. If the far path is temporarily invisible but the near path remains valid, the car follows at reduced speed. After the configured number of consecutive invalid near frames, the car stops and disarms to avoid automatic restart when the line returns.",
    "宽度样本按参考值标定；运行时按画面位置换算宽度区间，并按相邻扫描行估计的局部斜率修正弯道斜切造成的表观变宽。额外容差不宜设得过大。": "Width samples are calibrated against the reference. At runtime, the width interval is converted by image position and corrected for apparent widening caused by curved-tape obliquity using the local slope estimated from adjacent scanlines. Do not set the extra tolerance too large.",
    "倍率只缩放前进分量，转向分量保留；为避免原地反转，转向量不会超过当前前进量。它不是实测车速，图像曲率阈值需用实际落地画面标定。识线不确定时仍按安全逻辑停车。": "The scale only affects the forward component; steering is preserved. To avoid in-place reversal, steering never exceeds the current forward command. This is not measured speed; the image-curvature threshold must be calibrated on the actual ground image. The car still stops under the safety logic when line detection is uncertain.",
    "参考云台程序的俯仰软件范围为 -40° 至 +90°（-40°≈1944 μs，+90°=500 μs）；这不是机械限位反馈。请确认机构无干涉，并核对 Model A PWM 通道。未确认前保持关闭；回位按钮仅在电机未使能时可用。": "The reference gimbal program uses a tilt software range of -40° to +90° (-40°≈1944 μs, +90°=500 μs); this is not mechanical-limit feedback. Confirm there is no mechanical interference and verify the Model A PWM channels. Keep it off until confirmed; the Home button works only while the motors are disabled.",
    "勾选后请点击下方“保存并应用参数”，确认才会生效。": "After ticking, click “Save & Apply” below for the change to take effect.",
    "参数已保存；如果已启动，将停车并重新确认黑线。": "Parameters saved; if running, the car will stop and reacquire the line.",
    "保存失败：${error.message}": "Save failed: ${error.message}",
    "读取参数失败：${error.message}，正在自动重试。": "Failed to read parameters: ${error.message}; retrying automatically."
  };

  const PATTERNS = [
    [/^状态更新：(.+) · 每秒自动读取$/, "Status: $1 · auto-refresh every second"],
    [/^数据过期（(.+)s）$/, "Data stale ($1s)"],
    [/^水平 (.+)（(.+) μs）· 俯仰 (.+)（(.+) μs）$/, "Pan $1 ($2 μs) · Tilt $3 ($4 μs)"],
    [/^无法启动：(.+)$/, "Unable to start: $1"],
    [/^云台回位失败：(.+)$/, "Gimbal homing failed: $1"],
    [/^云台移动失败：(.+)$/, "Gimbal move failed: $1"],
    [/^停止请求失败：(.+)$/, "Stop request failed: $1"],
    [/^避障设置失败：(.+)$/, "Failed to update obstacle avoidance: $1"],
    [/^读取配置失败：(.+)，正在重试。$/, "Failed to read configuration: $1; retrying."],
    [/^停车未确认：(.+)。请立即物理断电。$/, "Stop not confirmed: $1. Physically disconnect power immediately."],
    [/^IMU 数据不可用：(.+)。电机运动已锁定。$/, "IMU data unavailable: $1. Motor motion is locked."],
    [/^(.+)。电机状态未知，请立即物理断电。$/, "$1. Motor state unknown; physically disconnect power immediately."],
    [/^状态更新：(.+)$/, "Status: $1"],
    [/^地面 ROI 与 (.+)–(.+)mm 胶带宽度范围已标定。当前仅预览，电机未使能；改变云台角度或安装高度后需重新标定。$/, "Ground ROI and tape width range $1–$2 mm are calibrated. Preview only; motors are disabled. Recalibrate after changing the gimbal angle or mounting height."],
    [/^地面 ROI \/ 胶带宽度范围尚未标定，电机保持锁止。 /, "Ground ROI / tape width range is not calibrated; motors stay locked. "],
    [/^小车本机记录：/, "Onboard recording: "],
    [/^部分记录已达上限：/, "Some recordings reached their limit: "],
    [/^左转 (.+)°$/, "Left turn $1°"],
    [/^右转 (.+)°$/, "Right turn $1°"],
    [/^默认关闭，关闭时不执行避障转弯寻线。开启后，([0-9.]+) cm 起降速，距离 ≤ ([0-9.]+) cm 停车、原地左转约90°后前进寻线，未找到再原地右转寻线；找到稳定黑线后继续循迹，失败则停车。切换会停车，需重新点击导航，设置自动保存。$/, "Disabled by default; no avoidance turn/search runs while disabled. When enabled, the car slows at $1 cm and stops at distance ≤ $2 cm, turns left about 90° and searches forward; if no line is found, it turns right and searches again. A stable line hands control back to tracking; failure stops the car. Switching stops the car, requires pressing Start again, and saves automatically."]
  ];

  const PHRASES = [
    ["未通过", "Not passed"],
    ["未就绪", "Not ready"],
    ["空间 有效", "Spatial valid"],
    ["空间 不确定", "Spatial uncertain"],
    ["时序 有效", "Temporal valid"],
    ["时序 不确定", "Temporal uncertain"],
    ["置信度", "confidence"],
    ["有效扫描线", "valid scanlines"],
    ["重捕获", "reacquired"],
    ["胶带范围", "tape range"],
    ["近端标定像素范围", "near calibrated pixel range"],
    ["条数据", "data rows"],
    ["张原图", "raw images"],
    ["无预测", "no prediction"],
    ["联合控制", "Combined control"],
    ["画面向左弯", "Curve to the left"],
    ["画面向右弯", "Curve to the right"],
    ["预览", "Preview"],
    ["直角弯：", "Right-angle turns: "],
    ["等待 L 弯", "waiting for L turn"],
    ["通过", "Pass"],
    ["启用", "Enabled"],
    ["目标", "target"],
    ["当前", "current"],
    ["例如：", "Example: "],
    ["默认关闭，", "Disabled by default. "],
    ["开启后，", "When enabled, "],
    ["距离 ≤ ", "distance ≤ "],
    ["停车、", "stop, "],
    ["状态更新：", "Status: "],
    ["每秒自动读取", "auto-refresh every second"],
    ["数据过期", "Data stale"],
    ["水平", "Pan"],
    ["俯仰", "Tilt"],
    ["正在自动重连", "reconnecting automatically"],
    ["旧数据已失效", "old data is invalid"],
    ["电机未使能", "motors disabled"],
    ["电机保持锁止", "motors stay locked"],
    ["需重新标定", "recalibration required"],
    ["请立即物理断电", "physically disconnect power immediately"],
    ["未找到稳定黑线", "no stable line found"],
    ["找到稳定黑线后继续循迹", "a stable line hands control back to tracking"],
    ["失败则停车", "failure stops the car"],
    ["原地左转约90°后前进寻线", "turn left about 90° and search forward"],
    ["未找到再原地右转寻线", "if not found, turn right and search again"],
    ["测距不可用时停车", "stop when distance is unavailable"],
    ["起降速", "begin slowing"],
    ["设置自动保存", "settings save automatically"],
    ["切换会停车", "switching stops the car"],
    ["需重新点击导航", "press Start again"],
    ["关闭时不执行避障转弯寻线", "no avoidance turn/search runs while disabled"],
    ["正在重试", "retrying"],
    ["画面连接中断", "Video disconnected"],
    ["画面连接已建立", "Video connected"],
    ["断线自动重连", "automatic reconnect"],
    ["旧画面不代表当前现场", "the old image does not represent the current scene"],
    ["页面在后台，返回后重新连接画面", "The page is in the background; the video reconnects when you return"],
    ["云台移动失败", "Gimbal move failed"],
    ["云台回位失败", "Gimbal homing failed"],
    ["无法启动", "Unable to start"],
    ["停止请求失败", "Stop request failed"],
    ["避障设置失败", "Failed to update obstacle avoidance"],
    ["读取配置失败", "Failed to read configuration"],
    ["停车未确认", "Stop not confirmed"],
    ["IMU 数据不可用", "IMU data unavailable"],
    ["电机运动已锁定", "motor motion is locked"],
    ["电机状态未知", "motor state unknown"],
    ["已发送", "Sent"],
    ["无角度反馈", "no angle feedback"],
    ["手动位置", "Manual position"],
    ["最后命令为零", "Last command zero"],
    ["无反馈", "no feedback"],
    ["非零命令", "Non-zero command"],
    ["状态未知", "state unknown"],
    ["暗色候选", "Dim candidate"],
    ["未确认", "unconfirmed"],
    ["准备就绪", "Ready"],
    ["正常循迹", "Normal tracking"],
    ["接近弯心", "Approaching corner"],
    ["确认出口", "Confirming exit"],
    ["出口已捕获", "Exit captured"],
    ["接近停车标志", "Approaching stop marker"],
    ["已到停车标志", "Stop marker reached"],
    ["停车定位失败", "Stop positioning failed"],
    ["等待弯心画面", "Waiting for corner view"],
    ["直角弯出口确认", "Corner exit confirm"],
    ["直角弯退出停车", "Corner abort (stopped)"],
    ["近端路径丢失停车", "Near path lost (stopped)"],
    ["记录达到存储上限", "Recording storage limit reached"],
    ["正在保存结束记录", "Saving end record"]
  ];

  let language = DEFAULT_LANGUAGE;
  let observer = null;
  let applying = false;
  let titleOrigin = null;
  const textOrigins = new WeakMap();
  const attrOrigins = new WeakMap();
  const ATTRS = ["placeholder", "title", "aria-label", "alt"];

  function hasChinese(text) {
    return /[\u3400-\u9fff]/.test(text);
  }

  function translate(text) {
    if (!text || !hasChinese(text)) return text;
    const key = text.trim();
    if (EN[key]) return text.replace(key, EN[key]);
    const spatial = key.match(/^空间 (有效|不确定) \/ 时序 (有效|不确定) · 置信度 ([0-9.]+) · 有效扫描线 ([0-9]+\/[0-9]+) · 重捕获 ([0-9]+) · 胶带范围 ([^·]+?) · 近端标定像素范围 ([^。]+)。 ?(.*)$/);
    if (spatial) {
      const state = value => value === "有效" ? "valid" : "uncertain";
      return `Spatial ${state(spatial[1])} / temporal ${state(spatial[2])} · confidence ${spatial[3]} · valid scanlines ${spatial[4]} · reacquired ${spatial[5]} · tape range ${spatial[6].trim()} · near calibrated pixel range ${spatial[7].trim()}. ${spatial[8]}`;
    }
    const record = key.match(/^(?:小车本机记录：|Onboard recording: )(.+?) · ([0-9]+) 条数据 \/ ([0-9]+) 张原图(.*)$/);
    if (record) {
      const state = EN[record[1].trim()] || record[1].trim();
      return `Onboard recording: ${state} · ${record[2]} data rows / ${record[3]} raw images${record[4]}`;
    }
    for (const [pattern, replacement] of PATTERNS) {
      if (pattern.test(key)) return key.replace(pattern, replacement);
    }
    let out = key;
    for (const [zh, en] of PHRASES) {
      if (out.includes(zh)) out = out.split(zh).join(en);
    }
    if (out !== key) {
      out = out.replace(/。/g, ". ").replace(/；/g, "; ").replace(/，/g, ", ");
      out = out.replace(/\s+/g, " ").trim();
    }
    return out === key ? text : text.replace(key, out);
  }

  function applyTextNode(node) {
    if (!node || !node.nodeValue || !node.nodeValue.trim()) return;
    if (node.parentElement && node.parentElement.closest("script, style, textarea, pre, code, [data-i18n-skip]")) return;
    if (language === "zh") {
      const original = textOrigins.get(node);
      if (original !== undefined && node.nodeValue !== original) node.nodeValue = original;
      return;
    }
    if (!hasChinese(node.nodeValue)) return;
    if (!textOrigins.has(node)) textOrigins.set(node, node.nodeValue);
    const translated = translate(node.nodeValue);
    if (translated !== node.nodeValue) node.nodeValue = translated;
  }

  function applyAttributes(element) {
    if (!element || element.closest("script, style, [data-i18n-skip]")) return;
    for (const attr of ATTRS) {
      if (!element.hasAttribute(attr)) continue;
      const value = element.getAttribute(attr);
      if (!value || !hasChinese(value)) continue;
      let store = attrOrigins.get(element);
      if (!store) {
        store = {};
        attrOrigins.set(element, store);
      }
      if (language === "zh") {
        if (store[attr] !== undefined && value !== store[attr]) element.setAttribute(attr, store[attr]);
        continue;
      }
      if (store[attr] === undefined) store[attr] = value;
      const translated = translate(value);
      if (translated !== value) element.setAttribute(attr, translated);
    }
  }

  function walk(root) {
    if (!root) return;
    if (root.nodeType === Node.TEXT_NODE) {
      applyTextNode(root);
      return;
    }
    if (root.nodeType !== Node.ELEMENT_NODE && root.nodeType !== Node.DOCUMENT_NODE) return;
    applyAttributes(root);
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT | NodeFilter.SHOW_ELEMENT);
    let node = walker.nextNode();
    while (node) {
      if (node.nodeType === Node.TEXT_NODE) applyTextNode(node);
      else applyAttributes(node);
      node = walker.nextNode();
    }
  }

  function updateToggle() {
    document.querySelectorAll("[data-language-toggle]").forEach((button) => {
      button.textContent = language === "en" ? "Chinese" : "English";
      button.setAttribute("aria-label", language === "en" ? "Switch to Chinese" : "Switch to English");
      button.setAttribute("title", language === "en" ? "Switch to Chinese" : "Switch to English");
      button.setAttribute("lang", language === "en" ? "en" : "en");
    });
  }

  function applyLanguage(next, persist) {
    language = next === "zh" ? "zh" : "en";
    if (persist) localStorage.setItem(STORAGE_KEY, language);
    document.documentElement.lang = language === "zh" ? "zh-CN" : "en";
    applying = true;
    if (observer) observer.disconnect();
    if (document.title) {
      if (language === "zh") {
        if (titleOrigin) document.title = titleOrigin;
      } else if (hasChinese(document.title)) {
        if (titleOrigin === null) titleOrigin = document.title;
        document.title = translate(titleOrigin);
      }
    }
    walk(document.body);
    updateToggle();
    applying = false;
    if (observer) {
      observer.observe(document.body, { subtree: true, childList: true, characterData: true, attributes: true, attributeFilter: ATTRS });
    }
  }

  function toggleLanguage() {
    applyLanguage(language === "en" ? "zh" : "en", true);
  }

  function install() {
    if (document.querySelector("[data-language-toggle]") === null) {
      const holder = document.querySelector("header nav") || document.querySelector("header") || document.body;
      const button = document.createElement("button");
      button.type = "button";
      button.className = "i18n-toggle";
      button.setAttribute("data-language-toggle", "");
      button.style.cssText = "margin-left:12px;padding:6px 10px;border:1px solid #3b4b61;border-radius:6px;background:#1d2a3a;color:#dce8f7;font:inherit;cursor:pointer;";
      button.addEventListener("click", toggleLanguage);
      holder.appendChild(button);
    }
    document.querySelectorAll("[data-language-toggle]").forEach((button) => {
      if (!button.dataset.bound) {
        button.dataset.bound = "1";
        button.addEventListener("click", toggleLanguage);
      }
    });
    const saved = localStorage.getItem(STORAGE_KEY);
    applyLanguage(saved === "zh" ? "zh" : DEFAULT_LANGUAGE, false);
    observer = new MutationObserver((mutations) => {
      if (applying) return;
      applying = true;
      for (const mutation of mutations) {
        if (mutation.type === "characterData") applyTextNode(mutation.target);
        else if (mutation.type === "attributes") applyAttributes(mutation.target);
        else {
          for (const node of mutation.addedNodes) walk(node);
        }
      }
      applying = false;
    });
    observer.observe(document.body, { subtree: true, childList: true, characterData: true, attributes: true, attributeFilter: ATTRS });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", install, { once: true });
  } else {
    install();
  }
})();
