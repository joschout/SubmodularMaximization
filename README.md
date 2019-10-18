# Unconstrained Submodular Maximization
A collection of optimization algorithms for Unconstrained Submodular Maximization (USM) of non-monotone non-negative set functions. As maximizing such a function is NP-hard, finding a maximum for such a function is often done using a greedy approach resulting in an approximate solution. This repository contains Python implementations of a couple of optimization algorithms tackling USM. 

## Included algorithms

First, this repository includes the three algorithms proposed by Feige, U., Mirrokni, V. S., and Vondrák, J. in their paper:

> Feige, U., Mirrokni, V. S., & Vondrák, J. (2011). Maximizing Non-monotone Submodular Functions. SIAM J. Comput., 40(4), 1133–1153. https://doi.org/10.1137/090779346

These algorithms are called:
* Random Set
* Deterministic Local Search
* Smooth Local Search

Next, this repository contains two of the algorithms propsed by Buchbinder, N., Feldman, M., Naor, J. S., and Schwartz, R. in their paper:

> Buchbinder, N., Feldman, M., Naor, J. S., & Schwartz, R. (2015). A tight linear time (1/2)-approximation for unconstrained submodular maximization. SIAM Journal on Computing, 44(5), 1384–1402. https://doi.org/10.1137/130929205

## Usage

The following describes how to use this repository in your own implementation.

## The set function
To use this in your own code, your function to be maximized should be contained in an object of a class inheriting from `AbstractObjectiveFunction`. This class looks as follows:
``` Python 
class AbstractObjectiveFunction:
    def evaluate(self, input_set: Set[E]) -> float:
        raise NotImplementedError('Abstract Method')
```
That is, `AbstractObjectiveFunction` requires its subclasses to implement an `evaluate()` method, taking as input a `Set[E]` and resulting in a `float`. This method should evaluate the set function on the given set, returning the value of the function. This class corresponds to the *'value oracle'*, which should be able to return the value of the function to be maximixed for every possible subset of the *ground set*.

Typically, your own class inheriting `AbstractObjectiveFunction` can contain instance variables for parameters required by the objective function.

## The Optimizers
Every included optimizer inherits the class `AbstractOptimizer`. Each optimizer should be iniitialized with at least two arguments:
1. the objective function to be optimized
2. the ground set of items. The optimizers will search over the power set of this ground set.

The following shows the definition of the `AbstractOptimizer` class:

``` Python
class AbstractOptimizer:
    def __init__(self, objective_function: AbstractObjectiveFunction, ground_set: Set[E], debug: bool = True):
        self.objective_function: AbstractObjectiveFunction = objective_function
        self.ground_set: Set[E] = ground_set
        self.debug: bool = debug

    def optimize(self) -> Set[E]:
        raise NotImplementedError("abstract method")
```

## Reason behind this repository

This repository came into being while experimenting with the Interpretable Decision Sets algorithm (IDS), as proposed by Lakkaraju, Bach and Leskovec in

> Lakkaraju, H., Bach, S. H., & Leskovec, J. (2016). Interpretable Decision Sets: A Joint Framework for Description and Prediction. In Proceedings of the 22Nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (pp. 1675–1684). New York, NY, USA: ACM. https://doi.org/10.1145/2939672.2939874

[The IDS implementation associated with the original IDS paper can be found here.](https://github.com/lvhimabindu) It is limited in the fact that it includes code to learn an IDS model, but no code to actually apply the model, and no code to replicated the experiments from the paper. A great re-implementation of IDS by Jiri Filip and Tomas Kliegr called [PyIDS can be found here](https://github.com/jirifilip/pyIDS), and is described in

> Jiri Filip, Tomas Kliegr. PyIDS - Python Implementation of Interpretable Decision Sets Algorithm by Lakkaraju et al, 2016. RuleML+RR2019@Rule Challenge 2019. http://ceur-ws.org/Vol-2438/paper8.pdf

PyIDS re-implements IDS from scratch, making it more efficient and adding functionality missing in the original implementation, and some more benefits on top.

IDS learns a classification model by maximizing an unconstrained non-monotone non-negative submodular set function. The original IDS source code and PyIDS contain three optimization algorithms that can be used for USM:

* Random Set
* Deterministic Local Search
* Smooth Local Search

However, these implementations were tightly integrated with their respective IDS implementations. The code in this repository is both based on the original paper proposing the algorithms by Feige et al., and the implementations of Lakkaraju et al. and Filip et al. It decouples the optimization procedures from IDS, and contains versions that are largely rewritten from scratch, trying to make each algorithm more readable and extendable.


## References

The papers proposing the algorithms in this repository:

> Feige, U., Mirrokni, V. S., & Vondrák, J. (2011). Maximizing Non-monotone Submodular Functions. SIAM J. Comput., 40(4), 1133–1153. https://doi.org/10.1137/090779346
> Buchbinder, N., Feldman, M., Naor, J. S., & Schwartz, R. (2015). A tight linear time (1/2)-approximation for unconstrained submodular maximization. SIAM Journal on Computing, 44(5), 1384–1402. https://doi.org/10.1137/130929205

Some good references for submodular maximization
> Krause, A., & Golovin, D. (2011). Submodular function maximization. Tractability, 9781107025, 71–104. https://doi.org/10.1017/CBO9781139177801.004
> Bach, F. (2013). Learning with submodular functions: A convex optimization perspective. Foundations and Trends in Machine Learning, 6(2–3), 145–373. https://doi.org/10.1561/2200000039
> Badanidiyuru, A., & Vondrák, J. (2014). Fast algorithms for maximizing submodular functions. Proceedings of the Annual ACM-SIAM Symposium on Discrete Algorithms, 1497–1514.
> Buchbinder, N., & Feldman, M. (2018). Deterministic Algorithms for Submodular Maximization. ACM Transactions on Algorithms, 14(3).
> Buchbinder, N., & Feldman, M. (2019). Submodular Functions Maximization Problems. Handbook of Approximation Algorithms and Metaheuristics, Second Edition, 753–788. https://doi.org/10.1201/9781351236423-42

Andreas Krause and Carlos Guestrin maintain a [great website about submodular optimization and the submodularity property](https://las.inf.ethz.ch/submodularity/)

Jan Vondrak hosts the [slides for some great presentations he did about submodular functions on his website.](https://theory.stanford.edu/~jvondrak/presentations.html)

References for the Interpretable Decision Set algorithm:

> Lakkaraju, H., Bach, S. H., & Leskovec, J. (2016). Interpretable Decision Sets: A Joint Framework for Description and Prediction. In Proceedings of the 22Nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (pp. 1675–1684). New York, NY, USA: ACM. https://doi.org/10.1145/2939672.2939874
> Jiri Filip, Tomas Kliegr. PyIDS - Python Implementation of Interpretable Decision Sets Algorithm by Lakkaraju et al, 2016. RuleML+RR2019@Rule Challenge 2019. http://ceur-ws.org/Vol-2438/paper8.pdf
