# Calculus Study Notes

## Limits and Continuity

A limit describes the value a function approaches as the input approaches a
point. A function is continuous at x = a when the limit exists, the function is
defined at a, and both values are equal.

Common strategies:

- Factor and cancel removable discontinuities.
- Rationalize expressions with radicals.
- Use squeeze arguments for bounded oscillating factors.
- Compare dominant terms for limits at infinity.

## Derivatives

The derivative measures instantaneous rate of change and slope of the tangent
line. For position s(t), velocity is s'(t) and acceleration is s''(t).

Core rules:

- Power rule: d/dx x^n = n x^(n-1)
- Product rule: (fg)' = f'g + fg'
- Quotient rule: (f/g)' = (f'g - fg') / g^2
- Chain rule: d/dx f(g(x)) = f'(g(x))g'(x)

## Optimization Workflow

1. Define the quantity to maximize or minimize.
2. Translate constraints into equations.
3. Reduce the objective to one variable.
4. Find critical points and endpoints.
5. Interpret the answer with units and domain restrictions.

Example: To minimize material for a cylindrical can with fixed volume, write
surface area as a function of radius using V = pi r^2 h, substitute h, then
differentiate the area function.

## Integrals

Definite integrals accumulate signed area and total change. If F'(x) = f(x),
then the Fundamental Theorem of Calculus gives:

integral from a to b of f(x) dx = F(b) - F(a)

Useful techniques:

- Substitution reverses the chain rule.
- Integration by parts reverses the product rule.
- Partial fractions decompose rational functions.
- Trigonometric substitution handles roots involving a^2 - x^2, a^2 + x^2,
  and x^2 - a^2.

## Series

A power series represents a function as an infinite polynomial on an interval of
convergence. Always check endpoints separately after finding the radius of
convergence.

Common tests:

- Geometric series test
- p-series test
- Ratio test
- Alternating series test
- Comparison and limit comparison tests

