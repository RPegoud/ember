# RMS Norm

## Definition
$$
y_{i}= \frac{x_{i}}{\sqrt{ \epsilon + \frac{1}{n} \sum_{k}x_{k}^{2} }} * w_{i} 
$$
## Gradients
### ${} \partial x {}$
Since every $x_{i}$ contributes to every $y_{k}$ through the normalisation factor, we need to sum the contribution of every ${} k$. Let $o$ be an upstream gradient:
$$
dx_{i} = \frac{\partial o}{\partial x_{i}} = \sum_{k} \frac{\partial{o}}{\partial y_{k}} \frac{\partial y_{k}}{\partial x_{i}}\
$$
In the case where $i=k$, we have:
$$
\begin{align}
\left( \frac{u}{v} \right)^{'} &= \frac{u'v - uv'}{v^{2}} \\ \\
&\begin{cases}
u &= x_{i} \\
u' &= 1 \\
v &= \sqrt{ \frac{1}{n} \sum_{k}x_{k}^{2} } \\
v' &= \frac{1}{2} \left( \frac{1}{n}\sum_{k}x_{k}^{2} \right)^{-1/2} \left( \frac{2}{n}x_{i} \right)
\end{cases}
\end{align}
$$
$$
\begin{align} \\
\frac{\partial y_{i}}{\partial x_{i}} &= \frac{\partial}{\partial x_{i}} \frac{x_{i}}{\sqrt{ \epsilon  + \frac{1}{n}\sum_{k}x_{k}^{2}}}w_{i} \\
 \\
&= \left( \frac{\sqrt{{\frac{1}{n}\sum_{k}}x^{2}_{k} } - \frac{1}{2}\left( \frac{1}{n}\sum_{k}x_{k}^{2} \right)^{-1/2}\left( \frac{2}{n}x_{i} \right)x_{i} }{\frac{1}{n}\sum_{k}x_{k}^{2}} \right) w_{i} \\ 
&= \left( \frac{\sqrt{{\frac{1}{n}\sum_{k}}x^{2}_{k} } - \left( \frac{1}{n}\sum_{k}x_{k}^{2} \right)^{-1/2} \frac{1}{n} x_{i}^{2} }{\frac{1}{n}\sum_{k}x_{k}^{2}} \right) w_{i} \\
&= \left( \frac{\text{RMS} - \frac{1}{\text{RMS}} \frac{1}{n}x_{i}^{2}}{\text{RMS}^{2}} \right) w_{i} \\
&= \left( \frac{1}{\text{RMS}} - \frac{\frac{1}{\text{RMS}^{2}} \frac{1}{n}x_{i}^{2}} {\text{RMS}} \right) w_{i} \\ \\
&= \frac{w_{i}- \frac{1}{\text{RMS}^{2}}\cdot \frac{1}{n} \cdot x_{i}^{2}\cdot w_{i}}{\text{RMS}}
\end{align}
$$
If $k\neq i$, we have:
$$
\begin{align}
\frac{\partial y_{j}}{x_{i}} &= \left( \frac{- \frac{1}{2}\left( \frac{1}{n}\sum_{k}x_{k}^{2} \right)^{-1/2}\left( \frac{2}{n}x_{i} \right)x_{j}}{\frac{1}{n}\sum_{k}x_{k}^{2}} \right) w_{j} \\ \\
&= \frac{-\frac{1}{n}\cdot \frac{1}{\text{RMS}}\cdot x_{i}\cdot x_{j} \cdot w_{j}}{\text{RMS}^{2}}  \\ \\
&= \frac{-\frac{1}{\text{RMS}^{2}}\cdot \frac{1}{n} \cdot x_{i}\cdot x_{j} \cdot w_{j}}{\text{RMS}}
\end{align}
$$
Summing both expressions, we get:
$$
\begin{align}
\partial x_{i} &= \frac{\partial o}{\partial x_{i}} = \frac{\partial o}{\partial y_{i}} \frac{\partial y_{i}}{\partial x_{i}} + \sum_{j\neq i} \frac{\partial o}{\partial y_{j}} \frac{\partial y_{j}}{\partial x_{i}} \\ \\
&= \frac{1}{\text{RMS}}\left( \partial y_{i} \left[ w_{i} - \frac{1}{n} \cdot \frac{1}{\text{RMS}^{2}} \cdot x_{i}^{2} w_{i} \right]  + \sum_{j\neq i} \partial y_{j} \left[ -\frac{1}{\text{RMS}^{2}} \cdot \frac{1}{n} \cdot x_{i} \cdot x_{j} \cdot w_{j} \right]\right) \\
&= \frac{1}{\text{RMS}} \left[ w_{i}\cdot \partial y_{i}  -\sum_{j} \frac{1}{\text{RMS}^{2}} \cdot \frac{1}{n}  \cdot x_{i} \cdot x_{j} \cdot w_{j} \cdot \partial y_{j} \right] \color{blue} \tag{1} \\
&= \frac{1}{\text{RMS}} \left[ w_{i}\cdot \partial y_{i}  - \frac{1}{n \text{RMS}^{2}}  \cdot x_{i} \cdot \sum_{j} (x_{j} \cdot w_{j} \cdot \partial y_{j}) \right] 
\end{align}
$$
Here $\color{teal} (1)$ is obtained by noticing that for $i=j$ we have $x_{i}^{2}\cdot w_{i} = x_{i}\cdot x_{j} \cdot w_{j}$, therefore, we can pull this term inside the sum, making it a sum over all $j$.

And in vector notation:
1. $w_{i}\cdot \partial y_{i}$ is the element-wise multiplication of the gradient and weight vector at index $i$: $w \odot \partial y$
2. $S = \sum_{j} (x_{j} \cdot w_{j} \cdot \partial y_{j})$ sums the element-wise product of three items. This can be seen as the dot product between $x$ and the vector $(w \odot \partial y)$, resulting in a scalar value.
3. The remaining term is $\frac{1}{n \text{RMS}^{2}} \cdot x _{i} \cdot S$. Since $S$ is a scalar, this simply represents the vector $x$ scaled by $S$.
This yields the combined vector form:
$$
\partial x = \frac{1}{\text{RMS}} \left( w \odot \partial y - \frac{1}{n \text{RMS}^{2}} \langle \partial y \odot w, x \rangle \cdot x  \right)
$$

### $\partial w$
$$
dw_{i} = \frac{\partial o}{\partial y_{i}} \frac{\partial y_{i}}{\partial w_{i}}
$$
Each $w_{i}$ affects the output for a single $y_{i}$, therefore we don't need to sum over $k$:
$$
\begin{align}
\frac{\partial y_{i}}{\partial w_{i}} &= \frac{\partial}{\partial w_{i}} \frac{x_{i}}{\sqrt{ \epsilon + \frac{1}{n} \sum_{k}x_{k}^{2} }}\cdot w_{i} \\
&= \frac{x_{i}}{\sqrt{ \epsilon + \frac{1}{n} \sum_{k}x_{k}^{2} }}
\end{align}
$$
However, in a batched setting, the weight $w_{i}$ is shared across all tokens in the batch. The full gradient is therefore the sum of the gradients produced by every single token:
$$
\partial w_{i} = \sum_{\text{batch b}}\left(  \frac{\partial o^{(b)}}{\partial y_{i}^{(b)}} \cdot \frac{x_{i}^{(b)}}{\text{RMS}^{(b)}} \right)
$$
And in vector form:
$$
\partial w = \sum_{\text{batch b}}(\partial y \odot \hat{x})
$$
With $\hat{x} = \frac{x}{\text{RMS}}$.