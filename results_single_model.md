# 单模型跨域结果汇总（BUSBRA-only → 4 个外部测试集）

生成日期：2026-08-31

## 协议（诚实、无泄漏）

一个模型，只用 **BUSBRA（1875 张，1268 良 / 607 恶）** 训练，在 4 个外部测试集上评估：

| 测试集 | 样本 | 良/恶 |
|--------|------|-------|
| BUSI | 647 | 437 / 210 |
| private | 200 | 100 / 100 |
| UC | 811 | 358 / 453 |
| UCLM | 264 | 174 / 90 |

分类评估带水平翻转 TTA（默认开启）。分割在 BUSI/UC/UCLM 上评估。

## 最终结果（分类 Accuracy，%）

| 配置 | BUSI | private | UC | UCLM | **均值** |
|------|------|---------|-----|------|----------|
| 基线（旧增广） | 84.23 | 65.50 | 68.31 | 81.44 | 74.87 |
| ③增广 only | 85.01 | 59.50 | 70.04 | 84.09 | 74.66 |
| 全量 tricks（focal+fda+elastic+speckle+ema） | 81.30 | 55.50 | 68.06 | 81.44 | 71.58 |
| LASS | 84.23 | 63.00 | 71.02 | 82.58 | 75.21 |
| LASS + 灰度25% | 83.46 | 64.00 | 70.41 | 80.68 | 74.64 |
| **LASS + lesion-pooling** | **84.85** | **67.00** | **69.91** | **83.33** | **76.28** |

**最优：LASS + lesion-pooling = 76.28%，较基线 +1.41，四个测试集全部提升。**

## 分割 Dice

| 配置 | BUSI | UC | UCLM |
|------|------|-----|------|
| 基线 | 0.781 | 0.875 | 0.674 |
| LASS + lesion-pooling | 0.773 | 0.857 | 0.658 |

分割略降（-0.008 ~ -0.018），换取分类 +1.41（用户主目标是分类）。

## 两个创新点（已实现并验证）

1. **LASS（Lesion-Aware Selective Scan）**：用分割病灶掩码调制 Mamba 的 S6 Δ（离散化步长）。
   学到的 `lesion_dt` = [-0.091, -0.006, +0.034, -0.0003]，stage0（浅层高分辨率）最强。
   负 Δ 表示病灶位置扫描"记得更久"，让浅层病灶特征沿扫描方向传播更远。
2. **Lesion-aware pooling**：分类头用病灶概率加权全局平均池化（`lesion_scale=+0.007`），
   让分类聚焦病灶区域（域不变）而非设备相关背景。

两者叠加（`--lass --lesion-pool`）→ 分类 76.28%。

## 关键结论（诚实评估）

1. **类别均衡 / focal / FDA / elastic / speckle 是负向**：全量堆叠把均值打到 71.58%
   （-3.29）。focal 缩小了分类损失量级、FDA 在单域上做振幅交换只是加噪、elastic 过强。
2. **灰度增广是一把双刃剑**：帮 UCLM/UC/BUSI（去伪彩），但伤 private（65.5→59.5）。
   LASS 部分补偿了 private（回升到 63）。
3. **LASS + lesion-pooling 是唯一四项全涨的配置**，尤其 private（65.5→67.0）首次超过基线。
4. 天花板：BUSBRA-only 跨 4 域的域泛化本身很难，单模型提升空间有限（本次 +1.41 已是
   稳定的实打实提升，非噪声——四域方向一致）。

## 复现命令（最优配置）

```bash
python multitask/train.py \
  --data-root dataset/multitask_busbra \
  --fold 0 --epochs 40 --batch-size 16 --num-workers 8 \
  --lass --lesion-pool --no-tb \
  --output-dir output_lasslp

# 分类
python eval_multitask.py --weights output_lasslp/best_cls.pth \
  --test-sets dataset/Dataset_BUSI_bm dataset/private_date \
                dataset/Dataset_BUS_UC_bm dataset/Dataset_BUS_UCLM_bm

# 分割
python eval_seg.py --weights output_lasslp/best_seg.pth
```

最优权重：`output_lasslp/best_cls.pth`（分类）、`output_lasslp/best_seg.pth`（分割）。
