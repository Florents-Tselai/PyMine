# Python Implementation of the [MINE](http://www.exploredata.net/) family of statistics

This is an implementation of the algorithms found in the paper [Detecting Novel Associations in Large Data Sets](http://science.sciencemag.org/content/334/6062/1518.full)'

Abstract
--------

Identifying interesting relationships between pairs of variables in large data sets is increasingly important.
Here, we present a measure of dependence for two-variable relationships:
the maximal information coefficient (MIC).
MIC captures a wide range of associations both functional and not,
and for functional relationships provides a score that roughly equals the coefficient of determination (R2)
of the data relative to the regression function.
MIC belongs to a larger class of maximal information-based nonparametric exploration (MINE) statistics for
identifying and classifying relationships.

The code has been written following Test-Driven-Development principles

*Note: Parts of the code are note PEP8-compliant. This is on purpose as I've strictly followed the naming conventions followed in the paper*

#Example

```python
import numpy as np

x = np.array(range(1000))
y = 4 * (x - 1. / 2) ** 2
D = zip(x, y)
n = len(D)
B = pow(n, 0.6)
c = 15
M = ApproxCharacteristicMatrix(D, B, c=1)
print(mine(M, B, c))
```
Result:
{'MCN': 2.0, 'MIC': 1.0, 'MEV': 1.0, 'MAS': 0.0}
