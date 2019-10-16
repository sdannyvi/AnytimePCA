# SSPCA-EXP
[Link the the paper at archive](https://www.google.com)

---
- SSPCA-EXP is an algorithm for the sparse-pca problem (sparsity in &#8467;0-norm), desinged to solve high-dimensional, ill-posed datasets. 
- Standard PCA (eigendecomposion) tends to fit noise in such high-dimensiona settings. To overcome this problem sparsity contsraints are added, which render the problem computationally hard. SSPCA-EXP is an approximation algorithm, whose run time can be calibrated according to the compute resources at hand. The parameter k* (see description below) governs the run-time.

- The full paper can be found here http://arxiv.org/abs/1910.06846
---
## Abstract 
The taxing computational effort that is involved in solving some high-dimensional statistical problems, in particular problems involving non-convex optimization, has popularized the development and analysis of algorithms that run efficiently (polynomial-time) but with no general guarantee on statistical consistency. In light of the ever increasing compute power and decreasing costs, perhaps a more useful characterization of algorithms is by their ability to calibrate the invested computational effort with the statistical features of the input at hand. For example, design an algorithm that always guarantees consistency by increasing the run-time as the SNR weakens. We exemplify this principle in the ‘0-sparse PCA problem. We propose a new greedy algorithm to solve sparse PCA that supports such a calibration. 

## Getting Started

Parameters:

* __fpath__: Enter the file path and name, either absolute path or relative to the run file. the excepted file is a numeric matrix of dimensions n,p. where n is the amount of rows and p is the amount of columns. 
__Default value__: "data.csv"
* __k__: Enter the desired sparsity level, the number of non-zero entries in the resulting sparse PC. 
__Default value__: "20".
* __k_star__: The algorithm exhastively searches all subsets of size k* (each such subset is termed "seed"), completing each one in a greedy way to a candidate set of size k. It then returns the set k which maixmizes variation. The run-time of the algorithm is thus exponential in k*.
__Default value__: "1".
* __batch__: Since the algorithm is built to run parallelly, each cpu will handle |batch| amount of seeds. By changing this parameter, task managment overhead can be decreased thus optimizing the runtime of the algorithm. Due notice - the algorithm does not optimize automaticaly and the optimization is up to the user.
__Default value__: "0".
* __cpus__: The total number of cpus to be used by the algorithm.
__Default value__: "1".
* __newrun__: The state of the algorithm is constatnlt being backed-up for case of bad connection, crashes etc... When starting a new execution with the same parameters as the last saved state, it is loaded automatically and the execution continues from that checkpoint. If you wish to generate a new run, and ignore old checkpoints, set this parameter to 1.
__Default value__: "0".

	
## Installing

In order to run *SSPCA-EXP*, first you need to install the next basic packages (which you probably already have):
- [Numpy](http://www.numpy.org/) (>=1.16.0)
- [Pandas](https://www.pandas.pydata.org/)
- [Scipy](https://www.scipy.org/)

*These packages are default on anaconda enviorment.*

## Examples
Assuming the "sspca_exp.py" is at the same folder as your data, there are two ways to run the algorithm. The first is via command line, the second is via your favorite python IDE (TBD)...

### Command Line Examples:

#### Basic Example:
Find the best 20 features of "mydata", using a seed of size 1, while using multiprocessing on 2 CPUs when the batch size is 210.
```
python3 sspca_exp.py --k 20 --k_star 1 --path "./mydata.csv" --cpus 2 --batch 210
```
The output will be located at "out/[today's date]/sspcaExp_kstar[kstar]_k[k].csv" as a csv file. In it there are 4 columns, which contain algorithm name, explained trace, runtime, k_entries. 
__These k_entries are the actual output and the actual features that were chosen by sspca-exp__.

## Authors

* **Dr. Dan Vilenchik**.
* **Adam Soffer**.
 


## License

This project is licensed under the MIT License.

## Acknowledgments

* Dr. Johnathan Rosenblatt.

## For More Information
Contact Adam Soffer at Soffer@post.bgu.ac.il.

