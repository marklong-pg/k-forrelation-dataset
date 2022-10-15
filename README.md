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

 This project concerns the generation algorithm of a benchmark dataset based on the k-Forrelation problem, which is theoretically proved to be BQP-complete and solvable using QSVM and QNN [(Jager, 2022)](https://arxiv.org/abs/2207.05865). The k-Forrelation dataset is thus worth considering for use as an evaluation criterion for circuit design and selection algorithms.
 
 **Note:** Benchmarking of (classical) machine learning algorithms concern criteria such as model complexity, scalability, sample complexity, interpretability, ability to learn from data stream, performance, among others. It remains an open question what should be the criteria to benchmark quantum algorithms and confirm any hypothesis of performance improvement. This is because the matters of concern for quantum computing are different and include: number of qubits, noise, depth of circuit, trainability, among others. This research suggests some potential use for the k-Forrelation dataset, but leave further discussion on the proper benchmarking of quantum algorithms to future works

 ## **2. Contributions** <a id="contributions"></a>
* An analysis of the properties of the k-Forrelation datasets in classification task
* An algorithm to generate k-Forrelation datasets with high positive class threshold
* Guidelines for the generation of k-Forrelation datasets for benchmarking
* Suggestions and demonstration for potential uses of k-Forrelation datasets

## **3. K-Forrelation Classification Problem** <a id="problem"></a>
In 2014, [Aaronson and Ambainis](https://www.scottaaronson.com/papers/for.pdf) proved the maximal separation in query complexity for black-box model between quantum and classical computation. The study involved a property-testing problem called **Forrelation** (originally introduced by [Aaronson, 2009](https://arxiv.org/pdf/0910.4698.pdf)). In Forrelation, two Boolean functions are given and the task is to decide if one function is highly correlated with the Fourier transform of the other. **K-fold Forrelation (or, k-Forrelation)** is the heuristical generalization of Forrelation that applies to k > 2 Boolean functions.  

A review of Fourier analysis on the Boolean Cube can be found [here (de Wolf, 2008)](https://theoryofcomputing.org/articles/gs001/gs001.pdf). This section summarizes the mathematical concepts of discrete Fourier transform on the Boolean cube for the generation of a classification dataset.

### *3.1. Fourier transform on the Boolean cube and correlation*
Consider a function 
$$
\begin{equation}
f : \{0,1\}^n \mapsto \R  
\end{equation} 
$$ 

The value table $f$ can be represented as a vector in $\R^{2n}$:
$$ 
\begin{equation}
    f \equiv \begin{bmatrix}
           f(0\cdots00) \\
           f(0\cdots01) \\
           \vdots \\
           f(1\cdots11)
         \end{bmatrix}
\end{equation} 
$$

(we will later focus on functions that map only to Boolean values $-1$ or $+1$. Here, assuming mapping to real numbers without loss of generality)

Define a $2^n \ -$ dimensional function space over $\R$ with inner product:

$$
\begin{equation}
\langle f,g \rangle \coloneqq \frac{1}{2^n} \sum_{x \in \{0,1\}^n}f(x)g(x) = \mathbb{E}[f(x)g(x)]  
\end{equation}
$$

which defines the $l_2$-norm:
$$
\begin{equation}
\Vert f \Vert_2 = \sqrt{\langle f,f \rangle} = \sqrt{\sum_{x \in \{0,1\}^n}f(x)^2}  = \sqrt{\mathbb{E}[f(x)^2] }
\end{equation}
$$


