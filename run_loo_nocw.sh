#!/bin/bash
# Same leave-one-out as run_loo.sh but WITHOUT class weighting (④ removed).
cd /root/autodl-tmp/BU-Mamba
mkdir -p logs_loo_nocw

EPOCHS=40
BATCH=16
WORKERS=8

run_one () {
  name=$1; prefix=$2; testdir=$3
  out=output_loo_nocw_${name}

  echo "===================== TRAIN exclude=${prefix} (no class-weight) -> test=${name} ====================="
  python multitask/train.py \
    --data-root dataset/multitask_all \
    --exclude-prefix "${prefix}_" \
    --batch-size "$BATCH" --epochs "$EPOCHS" --num-workers "$WORKERS" \
    --k-folds 5 --fold 0 --no-tb \
    --output-dir "$out" 2>&1 | tee "logs_loo_nocw/train_${name}.log"

  echo "===================== EVAL CLASSIFICATION  ${name} ====================="
  python eval_multitask.py --weights "$out/best_cls.pth" \
    --test-sets "dataset/$testdir" --num-workers "$WORKERS" 2>&1 | tee "logs_loo_nocw/eval_cls_${name}.log"

  if [ "$name" != "priv" ]; then
    echo "===================== EVAL SEGMENTATION  ${name} ====================="
    python eval_seg.py --weights "$out/best_seg.pth" 2>&1 | tee "logs_loo_nocw/eval_seg_${name}.log"
  fi
}

run_one busi busi Dataset_BUSI_bm
run_one priv priv private_date
run_one uc   uc   Dataset_BUS_UC_bm
run_one uclm uclm Dataset_BUS_UCLM_bm

echo "===================== ALL DONE (no class-weight) ====================="
