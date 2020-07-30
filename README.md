# Unconstrained Submodular Maximization
__________________________________
[Included algorithms](https://github.com/joschout/SubmodularMaximization#included-algorithms) - 
[Usage](https://github.com/joschout/SubmodularMaximization#usage) - 
[Installing submodmax](https://github.com/joschout/SubmodularMaximization#installing-submodmax) - 
[Reason behind this repo](https://github.com/joschout/SubmodularMaximization#reason-behind-this-repository) - 
[References](https://github.com/joschout/SubmodularMaximization#references)
_________________


A collection of optimization algorithms for Unconstrained Submodular Maximization (USM) of non-monotone non-negative set functions.
 
 Maximizing a non-monotone submodular function is NP-hard. This means there is no guarantee an optimal solution can be found within a polynomial number of function evaluations.  As maximization is NP-hard, finding a 'maximum' is often done using approximation algorithms resulting in an approximate solution. This repository contains Python implementations of a couple of optimization algorithms tackling USM. 

## Included algorithms

First, this repository includes the three algorithms proposed by Feige, U., Mirrokni, V. S., and Vondrák, J. in their paper:

> Feige, U., Mirrokni, V. S., & Vondrák, J. (2011). Maximizing Non-monotone Submodular Functions. SIAM J. Comput., 40(4), 1133–1153. https://doi.org/10.1137/090779346

These algorithms are called:
* Random Set
* Deterministic Local Search
* Smooth Local Search

Next, this repository contains two of the algorithms propsed by Buchbinder, N., Feldman, M., Naor, J. S., and Schwartz, R. in their paper:

> Buchbinder, N., Feldman, M., Naor, J. S., & Schwartz, R. (2015). A tight linear time (1/2)-approximation for unconstrained submodular maximization. SIAM Journal on Computing, 44(5), 1384–1402. https://doi.org/10.1137/130929205

For a lack of a better name, this repository calls these algorithms:
* Deterministic Double Greedy Search (Deterministic USM in the original paper)
* Randomized Double Greedy Search (Randomized USM in the original paper)


## Usage

The following describes how to use this repository in your own implementation.

### The submodular function to be optimized.

Here we describe the interface submodular functions should have to work with this package. We provide two different interfaces: one for general functions, and one for functions for which we can use a computational trick to speed things up. 

Now we describe our case where we can speed things up. Submodular optimization often repeatedly evaluates the function to be optimized, sequentially using different sets as input. Evaluating the function can be costly: the computational effort is often a function of the elements in the set. However, due to the way some optimization algorithms (such as Greedy Search) work, the input sets used for the sequence of evaluations do not differ that much. Often, only one element is added or removed at a time. For such functions, it is sometimes possible to reuse the function value obtained from the previous evaluation, and update it with the difference corresponding to the changed set. This can drastically reduced the number of work. 

An example is the Interpretable Decision Set objective function, of which the evaluation time in function of the input set size improves dramatically when the function value of the previous evaluation is reused, compared to recomputing the whole evaluation:

![Example animations of falling marbles](./images/ids_regular_vs_value_reuse_time_ifo_current_set_size_segment.jpeg)

### Submodular functions - without function value reuse
To use this in your own code, your function to be maximized should be contained in an object of a class inheriting from `AbstractSubmodularFunction`. This class looks as follows:
``` Python 
class AbstractSubmodularFunction:
    def evaluate(self, input_set: Set[E]) -> float:
        raise NotImplementedError('Abstract Method')
```
That is, `AbstractSubmodularFunction` requires its subclasses to implement an `evaluate()` method, taking as input a `Set[E]` and resulting in a `float`. This method should evaluate the set function on the given set, returning the value of the function. This class corresponds to the *'value oracle'*, which should be able to return the value of the function to be maximized for every possible subset of the *ground set*.

Typically, your own class inheriting `AbstractSubmodularFunction` can contain instance variables for parameters required by the objective function.

### Submodular functions - with function value reuse
To use this in your own code, your function to be maximized should be contained in an object of a class inheriting from `AbstractSubmodularFunctionValueReuse`. This class looks as follows:
``` Python 
class AbstractSubmodularFunctionValueReuse:

    def evaluate(self, current_set_info: SetInfo,
                 previous_func_info: Optional[FuncInfo],
                 ) -> FuncInfo:
        raise NotImplementedError('abstract method')
```


### The Optimizers
Every included optimizer inherits the class `AbstractOptimizer`. Each optimizer should be initialized with at least two arguments:
1. the objective function to be optimized
2. the ground set of items. The optimizers will search over the power set of this ground set.

The following shows the definition of the `AbstractOptimizer` class:

``` Python
class AbstractOptimizer:
    def __init__(self, objective_function: AbstractSubmodularFunction, ground_set: Set[E], debug: bool = True):
        self.objective_function: AbstractSubmodularFunction = objective_function
        self.ground_set: Set[E] = ground_set
        self.debug: bool = debug

    def optimize(self) -> Set[E]:
        raise NotImplementedError("abstract method")
```

## Installing submodmax

You can install this as a python package as follows:

```bash
git clone https://github.com/joschout/SubmodularMaximization.git
cd SubmodularMaximization/
python setup.py install develop --user
```

To use it in your project, you can use:
``` Python 
from submodmax import <what-you-need>
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
> 
> Buchbinder, N., Feldman, M., Naor, J. S., & Schwartz, R. (2015). A tight linear time (1/2)-approximation for unconstrained submodular maximization. SIAM Journal on Computing, 44(5), 1384–1402. https://doi.org/10.1137/130929205

Some good references for submodular maximization
> Krause, A., & Golovin, D. (2011). Submodular function maximization. Tractability, 9781107025, 71–104. https://doi.org/10.1017/CBO9781139177801.004
> 
> Bach, F. (2013). Learning with submodular functions: A convex optimization perspective. Foundations and Trends in Machine Learning, 6(2–3), 145–373. https://doi.org/10.1561/2200000039
> 
> Badanidiyuru, A., & Vondrák, J. (2014). Fast algorithms for maximizing submodular functions. Proceedings of the Annual ACM-SIAM Symposium on Discrete Algorithms, 1497–1514.
> Buchbinder, N., & Feldman, M. (2018). Deterministic Algorithms for Submodular Maximization. ACM Transactions on Algorithms, 14(3).
> 
> Buchbinder, N., & Feldman, M. (2019). Submodular Functions Maximization Problems. Handbook of Approximation Algorithms and Metaheuristics, Second Edition, 753–788. https://doi.org/10.1201/9781351236423-42

Andreas Krause and Carlos Guestrin maintain a [great website about submodular optimization and the submodularity property](https://las.inf.ethz.ch/submodularity/), linking to their [Matlab/Octave toolbox for Submodular Function Optimization](https://las.inf.ethz.ch/sfo/index.html).

Jan Vondrak hosts the [slides for some great presentations he did about submodular functions on his website.](https://theory.stanford.edu/~jvondrak/presentations.html)

References for the Interpretable Decision Set algorithm:

> Lakkaraju, H., Bach, S. H., & Leskovec, J. (2016). Interpretable Decision Sets: A Joint Framework for Description and Prediction. In Proceedings of the 22Nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (pp. 1675–1684). New York, NY, USA: ACM. https://doi.org/10.1145/2939672.2939874
> 
> Jiri Filip, Tomas Kliegr. PyIDS - Python Implementation of Interpretable Decision Sets Algorithm by Lakkaraju et al, 2016. RuleML+RR2019@Rule Challenge 2019. http://ceur-ws.org/Vol-2438/paper8.pdf
