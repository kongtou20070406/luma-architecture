#!/bin/bash
# wait_and_sanity.sh
# 后台监控训练进程，结束后自动跑 sanity_check.py
#
# 用法:
#   nohup bash wait_and_sanity.sh <train_pid> <label> > <out.log> 2>&1 &
# 例子:
#   nohup bash wait_and_sanity.sh 1026187 v19_epoch1 > /tmp/sanity_wait.log 2>&1 &

set -e

TRAIN_PID=${1:-}
LABEL=${2:-sanity}

if [ -z "$TRAIN_PID" ]; then
    echo "用法: $0 <train_pid> <label>"
    exit 1
fi

PYTHON="/home/kt/ai/.venvs/luma-global/bin/python"
ROOT="/home/kt/ai/luma-architecture"
SANITY_SCRIPT="$ROOT/minimind/scripts/sanity_check.py"
CKPT_DIR="$ROOT/minimind/artifacts/checkpoints"
OUT_DIR="$ROOT/minimind/artifacts/sanity_check"
mkdir -p "$OUT_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT_JSON="$OUT_DIR/${LABEL}_${TIMESTAMP}.json"

echo "================================================================"
echo "  wait_and_sanity.sh"
echo "  监控 PID: $TRAIN_PID"
echo "  label: $LABEL"
echo "  输出: $OUT_JSON"
echo "  时间: $(date)"
echo "================================================================"

# 等待训练进程结束
echo ""
echo "等待训练进程 $TRAIN_PID 结束..."
while kill -0 $TRAIN_PID 2>/dev/null; do
    sleep 60
done
echo "训练进程 $TRAIN_PID 已结束 ($(date))"

# 给 GPU 和 checkpoint 一点时间落盘
sleep 10

# 找最新 checkpoint
LATEST_CKPT=$(ls -t "$CKPT_DIR"/phase6_step*.pt 2>/dev/null | head -1)
if [ -z "$LATEST_CKPT" ]; then
    echo "ERROR: 没找到 checkpoint!"
    exit 1
fi
echo "使用 checkpoint: $LATEST_CKPT"

# 等 GPU 彻底空闲
echo ""
echo "等 GPU 空闲..."
for i in $(seq 1 30); do
    GPU_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 || echo 0)
    if [ "$GPU_MEM" -lt 5000 ]; then
        echo "GPU 已释放 (used=${GPU_MEM} MiB)"
        break
    fi
    sleep 10
done

# 运行 sanity check
echo ""
echo "================================================================"
echo "  启动 sanity_check.py"
echo "================================================================"
cd "$ROOT/minimind"
$PYTHON "$SANITY_SCRIPT" \
    --checkpoint "$LATEST_CKPT" \
    --output "$OUT_JSON" \
    --tests all 2>&1 | tee "$OUT_DIR/${LABEL}_${TIMESTAMP}.log"

echo ""
echo "================================================================"
echo "  sanity_check 完成 ($(date))"
echo "  结果: $OUT_JSON"
echo "================================================================"
