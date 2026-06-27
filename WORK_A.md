# SafetyCommander — A 的工作（Agent 推理 + 闭环编排 + 演示层）

> 范围：A 负责"会推理的 agent 大脑 + watch→decide→act→report 闭环 + 演示"。
> 感知层（YOLO 检测，B/4090）见 [PERCEPTION.md](PERCEPTION.md)，A 已为其接好接口。

**一句话定位**：VLM 读一份可编辑的安全规程，对工厂画面/视频推理出风险等级（必须引用条款），
按等级自动执行动作，合成班次交接报告——全程自主，且能**现场证明是模型在推理、不是代码硬编码**。

---

## 1. 架构（A 的部分）

```
   静态帧 frames/  ─┐                      ┌─ 视频 demo_clips/（多机位）
                    ▼                      ▼
   ┌──────────────── 自主闭环（每帧 / 每个时序窗口）──────────────┐
   │ ① WATCH 读画面 + 规程 + 可选 YOLO 事实                         │
   │ ② DECIDE  vlm_judge → Qwen3-VL 依规程推理 → 结构化判断+引用条款 │
   │ ③ ACT     dispatch → 按风险等级路由动作（不判风险）            │
   │ ④ REPORT  ShiftReport 累积 → 班末 markdown 交接单              │
   └──────────────────────────────────────────────────────────────┘
        持久化 logs/*.json   ·   reports/handoff_*.md   ·   大屏实时
   人改 safety_policy.txt → 行为即变（现场反杀 demo）
```

---

## 2. 已完成模块（逐个）

| 文件 | 职责 | 关键内容 |
|---|---|---|
| [config.py](config.py) | 配置 | 从 `.env` 读 VLM endpoint/key/model；路径；`load_policy()`；perception/annotated 目录 |
| [safety_policy.txt](safety_policy.txt) | **规程（唯一事实来源）** | 8 节：PPE / 叉车 / 火烟 / 泼洒 / 护栏 LOTO / 限制行为 / 区域豁免 / **风险分级 none→critical**；可被运营经理随时编辑 |
| [vlm_judge.py](vlm_judge.py) | **大脑** | `judge_frame()` 单帧；`judge_clip()` 多帧时序；强制结构化 JSON + 容错解析 + JSON-mode 兜底；grounding 防幻觉；`temperature=0` 可复现；`perception=` 融合 YOLO 事实 |
| [actions.py](actions.py) | 分级动作 | `create_safety_log / notify_supervisor / assign_corrective_action / escalate_incident / flag_area`；`dispatch()` **只按 VLM 给的等级路由，零风险判断** |
| [shift_report.py](shift_report.py) | 交接报告 | 累积事件 → `generate_handoff()` markdown（违规/near-miss/未关闭工单/按类型分布/全量日志） |
| [main.py](main.py) | 编排 | `run_shift()` 静态帧闭环；`run_video()`/`run_videos()` 单片 & **多机位整班**时序闭环；`sample_windows()` |
| [dashboard.py](dashboard.py) + [templates/index.html](templates/index.html) | 大屏 | Flask 实时仪表盘（轮询）：当前帧/判断/**引用条款**/触发动作/滚动 feed/班末报告；`SC_VIDEO=` 视频模式；YOLO 画框叠加位 |
| [demo_policy_flip.py](demo_policy_flip.py) | **climax** | 同一帧判两次，唯一差别是规程加一行 → 判断翻转、引用新条款、零改代码 |
| [extract_frames.py](extract_frames.py) | 工具 | 用 imageio-ffmpeg（自带 ffmpeg）从视频抽帧 |

---

## 3. 关键能力 & 实测结果（都对真实 endpoint 跑过）

- **单帧判断**：结构化 JSON，~5s/帧，引用具体条款。
- **视频时序判断**：多帧一次调用（4 帧 ~660 tokens）。铁证——同一超载叉车**单帧判 NONE、视频判 HIGH**，模型原话载荷前倾"在几帧间逐渐加大"。
- **多机位整班**：对 8 个真实 CCTV 片段跑出 **26 窗口 → 5 HIGH / 11 MEDIUM / 10 NONE，5 near-miss，11 未关闭工单**，near-miss 引用 2.1/2.2/2.4。
- **改规程翻转**（`demo_policy_flip.py`）：加 clause 2.6 → 同帧 **NONE→MEDIUM 并引用 2.6**，复现 **5/5**。
- **perception 融合**：喂入"0.8m"事实 → 模型自主升 **HIGH** 引用 2.1（仍由模型判，非硬编码）。

---

## 4. 命中评分

**"agent" 三要素（demo 必须三者都有）**
1. 推理模型 ✅ —— `vlm_judge` 读规程推理，引用条款。
2. 工作流工具 ✅（部分）—— 安全日志 / 通知 / 整改工单 / 升级 / 交接报告（Slack 待接）。
3. 自主 ✅ —— watch→decide→act→report 自己跑完，跨 8 机位。

**4 个评分问题**
1. 真实运营问题 ✅ 安全巡检+事件响应+交接。
2. **模型在推理而非开发者** ✅ —— 改规程现场反杀；`dispatch` 只路由、无 if-then 风险映射。
3. 端到端可信可落地 ✅ —— 真实 CCTV 视频 → 判断 → 动作 → 交接，大屏可演。
4. 敢给运营经理看 ✅ —— 零误报（grounding）+ 像真交接单的报告。

---

## 5. "没有硬编码风险"的证据（评分命门）
- 风险等级 100% 由 VLM 读 `safety_policy.txt` 推出，代码里**搜不到** `if hazard: risk=...`。
- [actions.py](actions.py) `dispatch()` 只把"模型给的等级"映射到动作集，是**动作路由**不是风险判断。
- 改一行规程 → 判断就变（`demo_policy_flip.py` 实证）。

---

## 6. 数据生命周期（一帧 / 一个时序窗口）
```
画面/窗口  →  encode(base64)  +  policy  +  shift_context  [+ YOLO facts]
          →  Qwen3-VL  →  {observation, hazard_type, risk_level, policy_clause,
                            reasoning, recommended_actions}
          →  dispatch(risk_level)  →  logs/*.json  +  通知/工单/升级
          →  ShiftReport.add  →  班末 reports/handoff_*.md  +  大屏
```

---

## 7. 怎么跑（命令清单）
```bash
pip install -r requirements.txt
cp .env.example .env          # 填 VLLM_KEY

python main.py                              # 静态帧整班
python main.py demo_clips/                  # 视频：8 机位整班（时序）
python demo_policy_flip.py                  # 改规程翻转 climax（~10s）

python dashboard.py                                  # 大屏：静态帧
SC_VIDEO=demo_clips python dashboard.py              # 大屏：8 机位视频整班
# http://localhost:8000
```

---

## 8. 还剩的 A 待办（非阻塞）
- [ ] **Slack 集成**：`notify_supervisor` 接真 incoming webhook（需要一个 URL）。
- [ ] **交接报告打磨成"真交接单"格式**（运营经理最吃，命中评分#4）。
- [ ] **pitch 5 页 + 3 分钟讲稿**：视频整班 → 改规程 climax → 交接报告。

> 兜底：纯 VLM 路径（单帧 + 视频）已全部稳定、可复现；改规程 climax 计过时（~10s）。
