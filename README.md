---
output: bookdown::html_document2
---

# K-Forrelation Dataset for Benchmarking Hybrid Classical-Quantum Classifiers

![image](.media/readme_cover.png)

*[Updated: Oct 13, 2022]*

*Note: All Rights Reserved - Unpublished research results*

## Table of Content
1. [Motivation](#motivation)
2. [Contributions](#contributions)
3. [Tools Used](#tools)
4. [K-Forrelation Classification Problem](#problem)
5. [Algorithms to generate the dataset](#algorithm)
6. [Guidelines for benchmarking & Generation](#guideline)

## **1. Motivation** <a id="motivation"></a>
Currently, most research on quantum classifiers (e.g., quantum support vector machine (QSVM), quantum neural networks (QNN)) employ simple datasets used in the classical machine learning literature (e.g., MNIST, Iris) to benchmark the performance of their variational quantum classifier (VQC), or the construction thereof. //{refs}//

 This project concerns the generation algorithm of a benchmark dataset based on the k-Forrelation problem, which is theoretically proved to be BQP-complete and solvable using QSVM and QNN [(Jager & Krems, 2022)](https://arxiv.org/abs/2207.05865). The k-Forrelation dataset is thus worth considering for use as an evaluation criterion for circuit design and selection algorithms.
 
 **Note:** Benchmarking of (classical) machine learning algorithms concern criteria such as model complexity, scalability, sample complexity, interpretability, ability to learn from data stream, performance, among others. It remains an open question what should be the criteria to benchmark quantum algorithms and confirm any hypothesis of performance improvement. This is because the matters of concern for quantum computing are different and include: number of qubits, noise, depth of circuit, trainability, among others. This research suggests some potential use for the k-Forrelation dataset, but leave further discussion on the proper benchmarking of quantum algorithms to future works

 ## **2. Contributions** <a id="contributions"></a>
* An algorithm to generate k-Forrelation datasets with high positive class threshold based on approximated Fourier Transform 
* An analysis of the properties of k-Forrelation datasets in classification task
* Guidelines for the generation of k-Forrelation datasets for benchmarking
* Suggestions and demonstration for potential uses of k-Forrelation datasets *(in progress)*

## **3. Tools Used** <a id="tools"></a>
* MATLAB (for development of sampling algorithm and dataset generation) 
* Python (for training composite-kernel SVM and quantum classifers on the datasets)
* Advanced Research Computing (ARC) Cluster to run all codes

## **4. K-Forrelation Classification Problem** <a id="problem"></a>
In 2014, [Aaronson and Ambainis](https://www.scottaaronson.com/papers/for.pdf) proved the maximal separation in query complexity for black-box model between quantum and classical computation. The study involved a property-testing problem called **Forrelation** (originally introduced by [Aaronson, 2009](https://arxiv.org/pdf/0910.4698.pdf)). In Forrelation, two Boolean functions are given and the task is to decide if one function is highly correlated with the Fourier transform of the other. **K-fold Forrelation (or, k-Forrelation)** is the heuristical generalization of Forrelation that applies to k > 2 Boolean functions.  

A review of Fourier analysis on the Boolean Cube can be found [here (de Wolf, 2008)](https://theoryofcomputing.org/articles/gs001/gs001.pdf). This section summarizes the mathematical concepts of discrete Fourier transform on the Boolean cube for the generation of a classification dataset.

### *4.1. Fourier transform on the Boolean cube and correlation*
Consider a function

$$\begin{align}
f : \{0,1\}^n \mapsto \mathbb{R}  
\end{align}$$

The value table $f$ can be represented as a vector in $\mathbb{R}^{2n}$:

$$ 
\begin{align}
    f \equiv \begin{bmatrix}
           f(0\cdots00) \\
           f(0\cdots01) \\
           \vdots \\
           f(1\cdots11)
         \end{bmatrix}
\end{align} 
$$

(we will later focus on functions that map only to Boolean values $-1$ or $+1$. Here, assuming mapping to real numbers without loss of generality)

Define a $2^n \ -$ dimensional function space over $\mathbb{R}$ with inner product:

$$
\begin{align}
\langle f,g \rangle \coloneqq \frac{1}{2^n} \sum_{x \in \{0,1\}^n}f(x)g(x) = \mathbb{E}[f(x)g(x)]  
\end{align}
$$

which defines the $l_2$-norm:

$$
\begin{align}
\Vert f \Vert_2 = \sqrt{\langle f,f \rangle} = \sqrt{\sum_{x \in \{0,1\}^n}f(x)^2}  = \sqrt{\mathbb{E}[f(x)^2] }
\end{align}
$$

Also, define the function $\chi_s(x)$ in the space as:

$$
\begin{align} 
\chi_s(x) = (-1)^{S \cdot x}\sqrt{2^n}
\end{align}
$$

where, $S \subseteq [n]=\{1,2,\dots,n\}$ with its characteristic vector $S \in \{0,1\}^n$. I.e., $S$ is the short-handed notation for a binary string of length $n$ with value 1 at the integer indices indicated by S. For example, $S = \{1,2,4\} \subseteq [5] \rightarrow S \equiv 1\ 1\ 0\ 1\ 0$.

<!---
Note: The definition of $\chi_s$ in Eqn. (5) includes a constant factor which makes the norm of $\chi_s$ not unity. This adjustment is mainly to reconcile with the form K-Forrelation introduced by Aaronson (2014)
--> 

$S \cdot x$ is the sum of the bit-wise product of binary string $S$ and $x$: 

$$
\begin{align} 
S \cdot x = \sum_{i=1}^ns_ix_i = \sum_{i \in S}x_i
\end{align}
$$

It can be shown that $\langle \chi_s,\chi_t\rangle = \delta_{st}$. Thus, the set of all $\chi_s$ forms an orthogonal basis of the function space.

Then, the Fourier Transform of any function $f(x)$ in the space can be defined as:

$$
\begin{align} 
\hat{f}(S) = \langle f, \chi_s \rangle & = \frac{1}{2^n} \sum_{x \in \{0,1\}^n}f(x)\chi_s(x) \\
\hat{f}(S) & = \frac{1}{\sqrt{2^n}} \sum_{x \in \{0,1\}^n} (-1)^{S \cdot x} f(x)
\end{align}
$$

Given two functions $f(x), \ g(x): \{0,1\}^n \mapsto \mathbb{R} $, computing the inner product of $f(x)$ and the Fourier transform of $g(x)$:

$$
\begin{align}
\langle f, \hat{g} \rangle &= \frac{1}{2^n} \sum_{x \in \{0,1\}^n}f(x)\hat{g}(x) \\
&= \frac{1}{2^n}\sum_{x \in \{0,1\}^n}f(x) \left( \frac{1}{\sqrt{2^n}} \sum_{y \in \{0,1\}^n} (-1)^{x \cdot y} g(y) \right) \\
& = \frac{1}{2^\frac{3n}{2}} \sum_{x,\ y \ \in \{0,1\}^n}f(x)(-1)^{x \cdot y}g(y) \\
\Phi_{f,g} & \coloneqq \frac{1}{2^\frac{3n}{2}} \sum_{x,\ y \ \in \{0,1\}^n}f(x)(-1)^{x \cdot y}g(y)
\end{align}
$$

$\Phi_{f,g}$ from Eqn. (12) is a measure of how correlated is $f(x)$ to the Fourier transform of $g(x)$.

The form of $\Phi$ can be heuristically generalize to $k$ functions, called k-Fold Forrelation or, k-Forrelation for short:

$$
\begin{align}
\boxed{
\Phi_{f_1,\dots,f_k} = \frac{1}{2^{n(k+1)/2}} \sum_{x_1,\dots,x_k \in \{0,1\}^n}f_1(x_1)(-1)^{x_1 \cdot x_2}f_2(x_2) \dots (-1)^{x_{k-1} \cdot x_k}f_k(x_k)
}
\end{align}
$$

### *4.2. k-Fold Forrelation Problem*

If we restrict the range of $f_k: \{0,1\}^n \mapsto \pm 1$ (instead of $\mathbb{R}$), $\Phi_{f_1,\dots,f_k}$ is precisely the amplitude with which the circuit shown in Fig. (1) returns $|0 \rangle ^{\otimes n}$ as its output (Aaronson, 2014)

![image](.media/kforr_circuit.png)
<p style="text-align: center;"> Figure 1. A quantum circuit that can be taken to define the k-fold Forrelation problem.  (Aaronson, 2014) </p>

Where $U_{f_k}$ maps each basis state $|x\rangle$ to $f_i(x)|x\rangle$

Since $\Phi$ represents amplitude of the zero-bitstring, we see that $|\Phi_{f_1,\dots,f_k}| \le 1$. 

**The decision problem:** Decide whether $|\Phi_{f_1,\dots,f_k}| \le \frac{1}{100}$ (no correlation) or $|\Phi_{f_1,\dots,f_k}| \ge \frac{3}{5}$ (high correlation) promised that one of them is the case. 

This decision problem was used to prove the maximal separation in query complexity between quantum and classical computation (Aaronson, 2014). Recently, Jager & Krems showed that variational quantum classifiers and quantum kernels have the theoretical expressiveness to solve the k-Forrelation problem (which is known to be PromiseBQP-complete). However, the authors used the exact quantum circuit that described the k-Forrelation problem as the feature map for both methods in the proof. 

The question remains to find a quantum classifier architecture capable of learning the k-Forrelation problem without the characteristic feature map (circuit in Fig. (1)) provided. This effort requires the generation of k-Forrelation datasets at various scales (length of input bitstring $n$, number of function $k$, and the positive class threshold $\mu$). Generating a balance dataset is difficult since the positive class is much rarer than the negative class, especially at larger $k$ and $n$. The current research studies the properties of and challenges in generating the k-Forrelation datasets, as well as propose an algorithm to sample balance dataset in high k. The generated binary datasets can potentially be used for benchmarking quantum classifiers due to their relevant to the quantum computing regime.

## **5. Algorithms to generate the datasets**

Boolean -1/+1

Conditions on the functions

Mean and variance of phi (general and policy space)

Random generation

Appromiated Fourier algorithm












