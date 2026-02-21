#!/bin/bash
# =============================================================================
# S3PO-GS 调参自动运行脚本
# 用法（在 Docker 容器 5d0fdd8cb92d 内运行）：
#   docker exec -it 5d0fdd8cb92d bash -c "cd /workspace && bash run_tune.sh"
# 或后台运行（推荐，断开终端也不会中断）：
#   docker exec -d 5d0fdd8cb92d bash -c "cd /workspace && nohup bash run_tune.sh &"
# 查看进度：
#   docker exec 5d0fdd8cb92d tail -f /workspace/tune_logs/run_tune_master.log
# =============================================================================

GPU_ID=2
LOG_DIR="/workspace/tune_logs"
SUMMARY_FILE="${LOG_DIR}/summary.csv"
MASTER_LOG="${LOG_DIR}/run_tune_master.log"
RESULT_DATASET_DIR="/workspace/results/stereo_seq_easy_stereo_seq_easy"

mkdir -p "$LOG_DIR"

# 实验列表：名称 | 配置文件路径
declare -a EXPS=(
  "baseline|configs/mono/Stereo/Stereo_seq_easy/exp_decouple.yaml"
  "P1A_alpha090|configs/mono/Stereo/Stereo_seq_easy/tune/tune_P1A_alpha090.yaml"
  "P1B_alpha093|configs/mono/Stereo/Stereo_seq_easy/tune/tune_P1B_alpha093.yaml"
  "P1C_alpha095|configs/mono/Stereo/Stereo_seq_easy/tune/tune_P1C_alpha095.yaml"
  "P2A_decouple41|configs/mono/Stereo/Stereo_seq_easy/tune/tune_P2A_decouple41.yaml"
  "P2B_decouple51|configs/mono/Stereo/Stereo_seq_easy/tune/tune_P2B_decouple51.yaml"
  "P3A_pw5|configs/mono/Stereo/Stereo_seq_easy/tune/tune_P3A_pw5.yaml"
  "P3B_ws10_pw5|configs/mono/Stereo/Stereo_seq_easy/tune/tune_P3B_ws10_pw5.yaml"
  "P4A_ns50|configs/mono/Stereo/Stereo_seq_easy/tune/tune_P4A_ns50.yaml"
  "P4B_ns80|configs/mono/Stereo/Stereo_seq_easy/tune/tune_P4B_ns80.yaml"
  "P5A_flow01_int10|configs/mono/Stereo/Stereo_seq_easy/tune/tune_P5A_flow01_int10.yaml"
  "P5B_flow02_int10|configs/mono/Stereo/Stereo_seq_easy/tune/tune_P5B_flow02_int10.yaml"
)

TOTAL=${#EXPS[@]}

# -----------------------------------------------------------------------------
# 日志函数
# -----------------------------------------------------------------------------
log() {
  local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
  echo "$msg"
  echo "$msg" >> "$MASTER_LOG"
}

# -----------------------------------------------------------------------------
# 从结果目录提取指标
# 参数：$1 = 实验名称, $2 = 该次运行的结果目录路径
# -----------------------------------------------------------------------------
extract_metrics() {
  local exp_name="$1"
  local run_dir="$2"

  if [ -z "$run_dir" ] || [ ! -d "$run_dir" ]; then
    log "  [WARN] 未找到本次运行的结果目录"
    echo "${exp_name},N/A,N/A,N/A,N/A,N/A" >> "$SUMMARY_FILE"
    return
  fi

  local stats_file="${run_dir}/plot/stats_final.json"
  local psnr_file="${run_dir}/psnr/after_opt/final_result.json"

  local rmse="N/A" psnr="N/A" ssim="N/A" lpips="N/A"

  if [ -f "$stats_file" ]; then
    rmse=$(python3 -c "import json; d=json.load(open('${stats_file}')); print(d.get('rmse','N/A'))" 2>/dev/null || echo "N/A")
  fi

  if [ -f "$psnr_file" ]; then
    psnr=$(python3 -c "import json; d=json.load(open('${psnr_file}')); print(d.get('mean_psnr','N/A'))" 2>/dev/null || echo "N/A")
    ssim=$(python3 -c "import json; d=json.load(open('${psnr_file}')); print(d.get('mean_ssim','N/A'))" 2>/dev/null || echo "N/A")
    lpips=$(python3 -c "import json; d=json.load(open('${psnr_file}')); print(d.get('mean_lpips','N/A'))" 2>/dev/null || echo "N/A")
  fi

  log "  结果: RMSE=${rmse}, PSNR=${psnr}, SSIM=${ssim}, LPIPS=${lpips}"
  log "  目录: ${run_dir}"
  echo "${exp_name},${rmse},${psnr},${ssim},${lpips},${run_dir}" >> "$SUMMARY_FILE"
}

# =============================================================================
# 主流程
# =============================================================================
log "=========================================="
log "S3PO-GS 调参实验 开始"
log "GPU: ${GPU_ID} | 实验数: ${TOTAL}"
log "=========================================="

# 写 CSV 表头
echo "experiment,rmse_ate,mean_psnr,mean_ssim,mean_lpips,result_dir" > "$SUMMARY_FILE"

FAILED=0
SUCCEEDED=0

for i in "${!EXPS[@]}"; do
  IFS='|' read -r EXP_NAME CONFIG_PATH <<< "${EXPS[$i]}"
  IDX=$((i + 1))

  log "------------------------------------------"
  log "[${IDX}/${TOTAL}] 实验: ${EXP_NAME}"
  log "  配置: ${CONFIG_PATH}"

  EXP_LOG="${LOG_DIR}/${EXP_NAME}.log"

  # 记录运行前已有的时间戳目录集合
  BEFORE_DIRS=$(ls -d "${RESULT_DATASET_DIR}"/20*/ 2>/dev/null | sort)

  # 运行 SLAM
  CUDA_VISIBLE_DEVICES=${GPU_ID} python3 slam.py --config "${CONFIG_PATH}" \
    > "${EXP_LOG}" 2>&1
  EXIT_CODE=$?

  if [ $EXIT_CODE -ne 0 ]; then
    log "  [FAIL] 退出码=${EXIT_CODE}，查看日志: ${EXP_LOG}"
    echo "${EXP_NAME},FAIL,FAIL,FAIL,FAIL,FAIL" >> "$SUMMARY_FILE"
    FAILED=$((FAILED + 1))
    # 不退出，继续下一个实验
    continue
  fi

  log "  [OK] 运行完成"
  SUCCEEDED=$((SUCCEEDED + 1))

  # 通过对比运行前后的目录找到本次新增的结果目录
  AFTER_DIRS=$(ls -d "${RESULT_DATASET_DIR}"/20*/ 2>/dev/null | sort)
  NEW_DIR=$(comm -13 <(echo "$BEFORE_DIRS") <(echo "$AFTER_DIRS") | tail -1)

  # 如果对比法失败，回退到最新目录
  if [ -z "$NEW_DIR" ]; then
    NEW_DIR=$(ls -td "${RESULT_DATASET_DIR}"/20*/ 2>/dev/null | head -1)
  fi

  # 去掉尾部斜杠
  NEW_DIR="${NEW_DIR%/}"

  extract_metrics "${EXP_NAME}" "${NEW_DIR}"
done

log "=========================================="
log "全部实验完成！成功 ${SUCCEEDED}/${TOTAL}，失败 ${FAILED}/${TOTAL}"
log "=========================================="
log ""
log "结果汇总表："

# 打印汇总表格
column -t -s',' "$SUMMARY_FILE" 2>/dev/null | tee -a "$MASTER_LOG" || cat "$SUMMARY_FILE" | tee -a "$MASTER_LOG"

log ""
log "汇总 CSV: ${SUMMARY_FILE}"
log "各实验日志: ${LOG_DIR}/<实验名>.log"
log "主日志:     ${MASTER_LOG}"
