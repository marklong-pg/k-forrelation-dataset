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
Currently, most research on quantum classifiers (e.g., quantum support vector machine (QSVM), quantum neural networks (QNN)) employ trivial datasets used in the classical machine learning literature (e.g., MNIST, Iris) to benchmark the performance of their variational quantum classifier (VQC), or the construction thereof. //{refs}//

 This project concerns the generation algorithm of a benchmark dataset based on the k-Forrelation problem, which is theoretically proved to be BQP-complete and solvable using QSVM and QNN [(Jager, 2022)](https://arxiv.org/abs/2207.05865). The k-Forrelation dataset is thus worth considering for use as an evaluation criterion for circuit design and selection algorithms.
 
 **Note:** Benchmarking of (classical) machine learning algorithms concern criteria such as model complexity, scalability, sample complexity, interpretability, ability to learn from data stream, performance, among others. It remains an open question what should be the criteria to benchmark quantum algorithms and confirm any hypothesis of performance improvement. This is because the matters of concern for quantum computing are different and include: number of qubits, noise, depth of circuit, trainability, among others. This research suggests some potential use for the k-Forrelation dataset, but leave further discussion on the proper benchmarking of quantum algorithms to future works

 ## **2. Contributions** <a id="contributions"></a>
* An analysis of the properties of the k-Forrelation datasets in classification task
* An algorithm to generate k-Forrelation datasets with high positive class threshold
* Guidelines for the generation of k-Forrelation datasets for benchmarking
* Suggestions and demonstration for potential uses of k-Forrelation datasets

## **3. K-Forrelation Classification Problem** <a id="problem"></a>


