<script type="text/javascript"
        src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0/MathJax.js?config=TeX-AMS_CHTML"></script>


# Bias - Variance experiment

In order to show that overfitting can be present especially under reasonable assumptions when dataset is small, this exercise has for goal to run simulated data generation and compare many modelling results with expectations.

In particular we can show that even using the generating process to fit data can fail if too few samples are available and simpler model are better.


## 1. Generation process

Write a function to generate one data point such that:

\\(x\\) is random in the interval [0, 1]


\\(y = x^2 + \epsilon\\) where \\(\epsilon\\) is taken from a normal distribution of mean 0 and standard deviation 0.25

The universe is fully characterized by the density probability of all pairs \\(x,y\\).


## 2. Experimental data collection process

In the real world, we only see a sample of the universe, say 10 such pairs.

Write a function to generate one such dataset.