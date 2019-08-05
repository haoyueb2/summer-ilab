# Chap1. Practical Aspects of Deep Learning

## 1.1 Train/Dev/Test Sets

- 训练数据的划分
    - 训练集
    - 验证集（简单交叉验证集）
    - 测试集
- 在训练集上训练，尝试不同的模型框架，在验证集上评估这些模型，然后迭代并选出适用的模型。
- 在机器学习中，如果只有一个训练集和一个验证集，而没有独立的测试集，遇到这种情 况，训练集还被人们称为训练集，而验证集则被称为测试集，不过在实际应用中，人们只是 把测试集当成简单交叉验证集使用，并没有完全实现该术语的功能。

## 1.2 Bias/Variance 偏差/方差

![image-20190724144750131](assets/image-20190724144750131.png)

1. 如果给这个数据集拟合一条直线，可能得到一个逻辑回归拟合，但它并不能很好地拟合该数据，这是高偏差(**high bias**)的情况，我们称为“欠拟合"(**underfitting**)
2. 如果我们拟合一个非常复杂的分类器，比如深度神经网络或含有隐藏单元的神经网络，可能就非常适用于这个数据集，但是这看起来也不是一种很好的拟合方式分类器方差较高（**high variance**），数据过度拟合（**overfitting**）。
3. 在两者之间，可能还有一些像图中这样的，复杂程度适中，数据拟合适度的分类器，这个数据拟合看起来更加合理，我们称之为“适度拟合”（**just right**）是介于过度拟合和欠拟合中间的一类。

## 1.4 Regularization

- 深度学习中解决过拟合问题——高方差的两个解决方法：
    - 正则化
    - 准备更多数据
- 正则化有助于避免过度拟合，减少网络误差。
- $L2$ 范数：$\|x\|_{2}=\sqrt{\sum_{i} x_{i}^{2}}$
- 逻辑回归函数中加入正则化

$$
\min_{w,b}J(w,b),\quad w\in \R^{n_x},b\in \R
$$

$$
J(w,b)=\frac{1}{m}\sum_{i=1}^{m}L(\hat{y}^{(i)},y^{(i)})+\frac{\lambda}{2m}\| w\|_2 ^2
$$

$$
L2\ \ regulation:\|w\|_2^2=\sum_{y=1}^{n_x}w_j^2=w^Tw
$$

- 只正则化参数 $w$ : 因为 $w$  通常是一个高维参数矢量，已经可以表达高偏差问题，$w$ 可能包含有很多参数，我们不可能拟合所有参数，而 $b$ 只是单个数字。
- $L1$ 正则化：

$$
J(w,b)=\frac{1}{m}\sum_{i=1}^{m}L(\hat{y}^{(i)},y^{(i)})+\frac{\lambda}{2m}\| w\|_1=\frac{1}{m}\sum_{i=1}^{m}L(\hat{y}^{(i)},y^{(i)})+\frac{\lambda}{2m}\sum_{j=1}^{n_x}|w_j|
$$

- 如果用 $L1$ 正则化，那么 $w$ 最终会是稀疏的，也就是说 $w$ 向量中优很多 $0$，
- $\lambda$ 是正则化参数，通常用验证集或交叉验证集来配置这个参数，尝试各种各样的数据，寻找最好的参数，我们要考虑训练集之间的权衡，把参数设置为较小值，这样可以避免过拟合，所以 $\lambda$ 是个需要调整的超参。
- 弗罗贝尼乌斯范数 **Frobenius Norm**

$$
\| w^{[l]}\|_F^2=\sum_{i=1}^{n^{[l]}}\sum_{j=1}^{n^{[l-1]}}(w_{ij}^{[l]})
$$

- 使用 **Frobenius Norm** 实现梯度下降

    用 **backprop** 计算出 $dW$ 的值，**backprop** 会给出 $J$ 对 $W$ 的偏导数，实际上是 $W^{[l]}$，然后$W^{[l]}=W^{[l]}-\alpha dW^{[l]}$ 
    $$
    dW^{[l]}=(\text{from backprop})+\frac{\lambda}{m}w^{[l]},\ \frac{\partial J}{\partial w^{[l]}}=\partial w^{[l]}
    $$

    $$
    W^{[l]}=W^{[l]}-\alpha\left[(\text{from backprop})+\frac{\lambda}{m}W^{[l]} \right]\\=W^{[l]}-\frac{\alpha \lambda}{m}W^{[l]}-\alpha(\text{from backprop})
    $$

- $L2$ 正则化也称为"权重衰减”

## 1.5 Why Regularization Reduces Overfitting?

- 直观上理解就是如果正则化参数 $\lambda$ 设置得足够大，权重矩阵 𝑊 被设置为接近于 0 的值，即把多隐藏单元的权重设为 0，于是基本上消除了这些隐藏单元的许多影响。
- 但是 𝜆 会存在一个中间值，于是会有一个接近“Just Right”的中间状态。

## 1.6 Dropout Regularization 随机失活正则化

1. **Inverted Dropout** 反向随机失活
2. 