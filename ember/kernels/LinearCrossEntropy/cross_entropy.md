## Definition
$$
\begin{align}
f(x, y, w) &= -\log\left( \frac{e^{ x\cdot w_{y} }}{\sum_{j}e^{ a(x_{j}w_{j}) }} \right) 
\end{align}
$$
## Gradients

$$
\begin{align}
\frac{ \partial L }{ \partial X } &=  \frac{ \partial L }{ \partial y } \frac{ \partial y }{ \partial x } = (\text{softmax}(XW) - Y_{\text{one-hot}}) \cdot W^{T} \\ \\
\frac{ \partial L }{ \partial W } &= \frac{ \partial L }{ \partial y } \frac{ \partial y }{ \partial w }   =X^{T} \cdot (\text{softmax}(XW)-Y_{\text{one-hot}})
\end{align}
$$