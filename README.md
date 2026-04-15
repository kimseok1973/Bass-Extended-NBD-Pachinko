# NBD-Bass 統合モデル: パチンコ参加者動態の閾値効果補正

Bass 拡張モデル（3状態 ODE）に **NBD（Negative Binomial Distribution）** を組み合わせ、
「年1回以上プレイ」という閾値観測による歪みを補正したベイズ推定フレームワーク。

---

## 1. 動機

従来の `Pachinko_3State_P1tP2t` では「パチンコ参加人口」の年次推移を直接観測として用いたが、 $p_1$ と $p_2$ の識別性が不十分だった。

**仮説**: 「年1回以上」の閾値は、**真の離脱**と**低頻度化による閾値下落（NBD 閾値効果）**を区別できない。見かけの $p_2$ は両者の合成である。

---

## 2. 数理モデル全体像

### 2.1 個人レベルの遊戯頻度モデル（NBD）

各個人 $i$ の年間遊戯回数 $X_i$ は Poisson、強度 $\lambda_i$ は Gamma 分布:

$$X_i \mid \lambda_i \sim \text{Poisson}(\lambda_i T), \qquad \lambda_i \sim \text{Gamma}(r(t), \alpha(t))$$

ここで $T = 1$ 年、 $r$ は形状、 $\alpha$ は rate（逆スケール）。 $\lambda$ を周辺化した分布は **Negative Binomial Distribution (NBD)** となる:

$$P(X_i = k \mid r, \alpha) = \frac{\Gamma(r + k)}{\Gamma(r) \, k!} \left(\frac{\alpha}{\alpha + 1}\right)^r \left(\frac{1}{\alpha + 1}\right)^k$$

集団平均: $\mathbb{E}[\lambda] = r / \alpha$、分散 $\mathbb{V}[\lambda] = r / \alpha^2$。

### 2.2 ゼロ確率と閾値補正

「年1回以上プレイ = アクティブ」の確率:

$$P(X \geq 1 \mid r, \alpha) = 1 - P(X = 0 \mid r, \alpha) = 1 - \left(\frac{\alpha}{\alpha + 1}\right)^r$$

これが **NBD 閾値補正係数** で、Bass 動態と観測を結ぶ橋渡し。

### 2.3 ビン確率

頻度ビン $k_{\min} \le X < k_{\max}$ の確率（heavy 等は $k_{\max} = \infty$）:

$$P_{\text{bin}}(r, \alpha) = \sum_{k = k_{\min}}^{k_{\max} - 1} P(X = k \mid r, \alpha)$$

アクティブ条件付きシェア:

$$\hat{p}_{\text{bin}} = \frac{P_{\text{bin}}(r, \alpha)}{1 - P(X = 0 \mid r, \alpha)}$$

### 2.4 Bass 拡張 3状態 ODE

正規化状態 $s + a + x = 1$:

| 状態 | 意味 |
|---|---|
| $s(t)$ | 潜在層（未参加） |
| $a(t)$ | アクティブプレイヤー（Bass 内部状態） |
| $x(t)$ | 離脱者 |

時変ガウス摂動:

$$p_1(t) = p_{1,\text{base}} + A_1 \exp\!\left(-\frac{1}{2}\left(\frac{t - c_1}{w_1}\right)^2\right)$$

$$p_2(t) = p_{2,\text{base}} + A_2 \exp\!\left(-\frac{1}{2}\left(\frac{t - c_2}{w_2}\right)^2\right)$$

連立 ODE（ $q_2 = 0$ 固定）:

$$\frac{ds}{dt} = -\bigl(p_1(t) + q_1 a\bigr) s$$

$$\frac{da}{dt} = \bigl(p_1(t) + q_1 a\bigr) s + \alpha \bigl(p_1(t) + q_1 a\bigr) x - p_2(t) \, a$$

$$\frac{dx}{dt} = p_2(t) \, a - \alpha \bigl(p_1(t) + q_1 a\bigr) x$$

### 2.5 時変 NBD パラメータ

$2013$ 年中心の対数線形:

$$\log r(t) = r_0 + r_1 (t - t_{\text{2013}} )$$

$$\log \alpha_{\text{nbd}}(t) = a_0 + a_1 (t - t_{\text{2013}} )$$

### 2.6 二重尤度

**観測1: 参加人口（1994-2019、年次）**

$$A_{\text{obs}}(t) \sim \mathcal{N}\!\left(M \cdot a(t) \cdot \bigl[1 - (\tfrac{\alpha_{\text{nbd}}(t)}{\alpha_{\text{nbd}}(t) + 1})^{r(t)}\bigr], \; \sigma^2\right)$$

**観測2: 頻度分布（n=1500、9 年）**

5ビンの観測カウント: 
```math
\mathbf{n}_t = (n_{\text{heavy}}, n_{\text{middle}}, n_{\text{lmiddle}}, n_{\text{light}}, n_{\text{rare}})
``` 
$$\mathbf{n}_t \sim \text{Multinomial}\!\bigl(N_{\text{survey}} = 1500, \; \hat{\mathbf{p}}_{\text{bin}}(r(t), \alpha_{\text{nbd}}(t))\bigr)$$

### 2.7 ビン定義（時間で異なる）

| ビン | 2008-2012 | 2013- |
|---|---|---|
| heavy | $X \geq 104$ | $X \geq 104$ |
| middle | $48 \leq X < 104$ | $48 \leq X < 104$ |
| lmiddle | $12 \leq X < 48$ | $12 \leq X < 48$ |
| light | $5 \leq X < 12$ | $4 \leq X < 12$ |
| rare | $2 \leq X < 5$ | $1 \leq X < 4$ |

### 2.8 事前分布

$$
\begin{aligned}
p_{1,\text{base}}, p_{2,\text{base}} &\sim \text{Uniform}(0, 0.3) \\
A_1 &\sim \text{Uniform}(0, 0.5),\ A_2 \sim \text{Uniform}(0, 0.5) \\
c_1 &\sim \text{Uniform}(1994, 2005),\ c_2 \sim \text{Uniform}(2005, 2019) \\
w_1, w_2 &\sim \text{Uniform}(0.5, 5.0) \\
q_1 &\sim \text{Uniform}(0, 2),\ \alpha \sim \text{Uniform}(0, 1) \\
M &\sim \text{Uniform}(1500, 5000)\ [\text{万人}],\ \sigma \sim \text{Uniform}(20, 500) \\
r_0 &\sim \text{Uniform}(\log 0.05, \log 5),\ r_1 \sim \text{Uniform}(-0.2, 0.2) \\
a_0 &\sim \text{Uniform}(\log 0.001, \log 1),\ a_1 \sim \text{Uniform}(-0.3, 0.3)
\end{aligned}
$$

合計 **16 パラメータの同時ベイズ推定**。

---

## 3. ファイル構成

| ファイル | 内容 |
|---|---|
| `Pachinko_NBD_Bass.ipynb` | ステップ1: 各年独立に NBD$(r, \alpha)$ を MLE 推定、$M_{\text{latent}}(t)$ を逆算 |
| `Pachinko_NBD_Bass_3State.ipynb` | ステップ2: $M_{\text{latent}}(t)$ を観測として 3状態 Bass 推定 |
| `Pachinko_NBD_Bass_Joint.ipynb` | **最終版**: NBD-Bass 統合ベイズ（16 パラメータ同時推定） |

---

## 4. 主要結果（統合モデル）

### 4.1 Bass 部分

| パラメータ | 推定値 | 95% CI | 解釈 |
|---|---|---|---|
| **M** | 3846 万人 | [3180, 4620] | 総潜在市場（$\lambda_i = 0$ 含む） |
| **c₂** | **2014.03** | ±4 年 | スマホ期の構造変化点 |
| **p₂_base** | 0.156 | CI/med 0.69 ○ | 平常離脱率（識別改善） |
| A₂ | 0.149 | CI/med 1.15 △ | $c_2$ 付近の離脱ブースト |
| q₁ | 0.76 | CI/med 1.46 △ | 内因伝播 |
| α (Bass) | 0.24 | CI/med 1.49 △ | 再加入率 |

### 4.2 NBD 部分

| パラメータ | 推定値 | 95% CI | 結論 |
|---|---|---|---|
| $r(2013)$ | 0.369 | tight | heterogeneity |
| $r_1$ | 0.008 | $[-0.006, 0.022]$ | ゼロ含む → **不変** |
| $\alpha_{\text{nbd}}(2013)$ | 0.00566 | tight | rate scale |
| $a_1$ | $-0.022$ | $[-0.038, -0.006]$ | **有意減少 ✓** |

### 4.3 構造的発見

1. **$r$ は不変**: 頻度分布の形（heterogeneity）は 2008-2019 で変化なし
2. **$\alpha_{\text{nbd}}$ は有意に減少**: 年率 $-2.2\%$ log units
3. **$\mathbb{E}[\lambda] = r / \alpha$ は増加**: 2008年 ~59 回/年 → 2019年 ~74 回/年

数値例：
$$\alpha(2008) = e^{-5.17 - 0.022 \times (-5)} = e^{-5.06} \approx 0.0063$$
$$\alpha(2019) = e^{-5.17 - 0.022 \times 6} = e^{-5.30} \approx 0.0050$$
$$\mathbb{E}[\lambda](2008) = 0.369 / 0.0063 \approx 58.6, \quad \mathbb{E}[\lambda](2019) = 0.369 / 0.0050 \approx 73.8$$

4. 「**薄い層が抜けて、残った人は heavy/middle 中心の濃いプレイヤー**」という市場構造
5. **$c_2 = 2014$ はロバスト**: どのモデル（従来 Bass, NBD 単独, 統合）でも一致

### 4.4 閾値補正の定量

$P(X = 0)$ の事後中央値:
$$P(X = 0 \mid r(2013), \alpha(2013)) = (0.00566 / 1.00566)^{0.369} \approx 0.148$$
$$\therefore P(X \geq 1) \approx 0.852$$

つまり Bass のアクティブ状態 $a(t)$ の人々のうち、年1回閾値を越えるのは約 85%。観測参加人口は真の Bass アクティブの $0.85$ 倍に過小評価されている。

---

## 5. 識別性改善の定量評価

| パラメータ | NBD単独版 | 従来 q2=0版 | **統合版** | 改善 |
|---|---|---|---|---|
| p₂_base | 1.27 △ | 1.34 △ | **0.69 ○** | ✓ |
| A₂ | 3.33 ✗ | 2.40 ✗ | **1.15 △** | ✓ |
| σ | 1.64 △ | 0.69 ○ | **0.35 ○** | ✓ |

`CI/med` $= (q_{0.975} - q_{0.025}) / |q_{0.5}|$、$< 1$ で○、$< 2$ で△、$\geq 2$ で✗

---

## 6. 実装環境

- Julia 1.10
- Turing.jl（NUTS, target accept 0.9, max_depth 10, 1000×4 chains）
- DifferentialEquations.jl（Tsit5, abstol = reltol = $10^{-6}$）
- SpecialFunctions.jl（`loggamma`, ForwardDiff 互換）

---

## 7. データ出典

- パチンコ参加人口: レジャー白書（1994-2019 年次、千人単位）
- 頻度分布調査: サンプル数 $n = 1500$ の年次調査（2008-2020、5 ビン）

---

## 8. 一般的枠組みとしての意義

本手法は、**Bass 系モデルへ NBD 観測補正を組み込む一般的フレームワーク**として、以下の領域に拡張可能:

- フィットネスサブスク（月1回以上利用を閾値とする場合）
- SNS 利用（週1回以上アクティブユーザー）
- 遊戯・娯楽習慣全般
- 閾値観測を伴う行動データ全般

任意の閾値 $\tau$ について、観測される「アクティブ数」は:

$$A_{\text{obs}}(t) = M(t) \cdot a(t) \cdot \mathbb{P}\bigl(X \geq \tau \mid r(t), \alpha(t)\bigr)$$

この定式化により、潜在状態（Bass 動態）と観測層（NBD 閾値超過）を分離して推定できる。
