# Mathematical Statistics Study Notes

## Probability Foundations

Probability models uncertainty with a sample space, events, and probability
rules. For events A and B:

P(A union B) = P(A) + P(B) - P(A intersection B)

Conditional probability is:

P(A given B) = P(A intersection B) / P(B)

Bayes' rule updates beliefs after observing evidence:

P(A given B) = P(B given A)P(A) / P(B)

## Random Variables

A random variable maps outcomes to numbers. Discrete random variables use a
probability mass function. Continuous random variables use a probability density
function and probabilities are areas under the density curve.

Expected value is the long-run average:

E[X] = sum x p(x) for discrete variables
E[X] = integral x f(x) dx for continuous variables

Variance measures spread:

Var(X) = E[(X - mu)^2] = E[X^2] - (E[X])^2

## Common Distributions

Bernoulli: one trial with success probability p.
Binomial: number of successes in n independent Bernoulli trials.
Poisson: count of events in a fixed interval with average rate lambda.
Normal: bell-shaped continuous model determined by mean and variance.
Exponential: waiting time model for memoryless processes.

## Estimation

An estimator is a statistic used to approximate a population parameter.

Important properties:

- Bias: whether the estimator is correct on average.
- Variance: how much the estimator changes across samples.
- Mean squared error: bias squared plus variance.
- Consistency: whether the estimator converges to the true value with more data.

## Hypothesis Testing

A hypothesis test compares observed data against a null model. The p-value is
the probability, assuming the null hypothesis, of observing a result at least as
extreme as the sample result.

A small p-value is evidence against the null, but it is not the probability that
the null hypothesis is true.

