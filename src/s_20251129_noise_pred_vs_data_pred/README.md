# 予測対象を変えて生成品質の違いを確認

## 1. 概要

* 論文 **"Back to Basics: Let Denoising Generative Models Denoise" (He et al., 2025\)**
* 「高次元空間（大きなパッチサイズ）ではデータ直接予測（x-prediction）が有効であり、ノイズ予測（eps-prediction）は破綻する」ことを検証。
* この主張は、データは低次元多様体上に分布するが、ノイズはそうではない高次元上に分布することを根拠としている。
* CIFAR-10およびCelebAを用い、予測ターゲットが生成品質に与える影響を比較する。
* JiT (Just image Transformer) 

## 2. 実験

**CIFAR-10 (32 x 32) :**
  * **Pred Mode:** x (提案手法) / eps (従来手法) / v
  * **Patch Size:** 4 (低次元) / 32 (画像全体を1パッチ＝1トークン)  


**CelebA (128 x 128, Center Crop) :**
* **Pred Mode:** x (提案手法) / eps (従来手法) / v
* **Patch Size:** 8 (低次元) / 128 (画像全体を1パッチ＝1トークン) 

## 3. 結果と考察

* 全モード、パッチサイズでLossは収束した。
* 高パッチでは、x-predictionのみ鮮明。vはノイズが残り、epsは完全なノイズに見える。
* 低パッチでは、どのモードでも高パッチより鮮明だが、epsは独特な品質となった。
* パッチサイズ32の高次元入力において、ネットワークは複雑なノイズ分布(eps)を学習しきれないが、データ多様体(x)へのマッピングは可能であることが確認された。論文の主張通り、x-predの優位性が再現されたとみえる。
* 低パッチ、eps予測の生成品質は、パッチ境界が残っている。これは、空間相関が弱く、ランダム性の高い高周波なノイズを、パッチ境界をまたいで滑らかに予測するのは難しいためと考える。
* 低パッチ、eps予測の生成品質は、白飛び・黒飛びが目立つ。これは、逆拡散過程の初期に、推定ノイズからデータを計算する以下の式において、 $\epsilon_\theta$ にわずかでも誤差があると、分母の $\sqrt{\bar{\alpha}_t}$ が小さいため、その誤差が増幅されたためと考える。

$$
x_0 \approx \frac{x_t - \sqrt{1 - \bar{\alpha}_t} \epsilon_\theta(x_t, t)}{\sqrt{\bar{\alpha}_t}}
$$


| パッチサイズ 4 (上から x, v, eps) | パッチサイズ 32 (上から x, v, eps) |
| :---: | :---: |
| ![x-8](../../output/s_20251129_noise_pred_vs_data_pred/20251129_cifar10/x_4_sample_epoch_010.png) | ![x-128](../../output/s_20251129_noise_pred_vs_data_pred/20251129_cifar10/x_32_sample_epoch_010.png) |
| ![v-8](../../output/s_20251129_noise_pred_vs_data_pred/20251129_cifar10/v_4_sample_epoch_010.png) | ![v-128](../../output/s_20251129_noise_pred_vs_data_pred/20251129_cifar10/v_32_sample_epoch_010.png) |
| ![eps-8](../../output/s_20251129_noise_pred_vs_data_pred/20251129_cifar10/eps_4_sample_epoch_010.png) | ![eps-128](../../output/s_20251129_noise_pred_vs_data_pred/20251129_cifar10/eps_32_sample_epoch_010.png) |

| パッチサイズ 8 (上から x, v, eps) | パッチサイズ 128 (上から x, v, eps) |
| :---: | :---: |
| ![x-8](../../output/s_20251129_noise_pred_vs_data_pred/20251130_celeba/x_8_sample_epoch_010.png) | ![x-128](../../output/s_20251129_noise_pred_vs_data_pred/20251130_celeba/x_128_sample_epoch_010.png) |
| ![v-8](../../output/s_20251129_noise_pred_vs_data_pred/20251130_celeba/v_8_sample_epoch_010.png) | ![v-128](../../output/s_20251129_noise_pred_vs_data_pred/20251130_celeba/v_128_sample_epoch_010.png) |
| ![eps-8](../../output/s_20251129_noise_pred_vs_data_pred/20251130_celeba/eps_8_sample_epoch_010.png) | ![eps-128](../../output/s_20251129_noise_pred_vs_data_pred/20251130_celeba/eps_128_sample_epoch_010.png) |