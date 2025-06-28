# Particle-Swarm-Optimisation
A simple Particle Swarm Optimisation algorithm, finding the minimum of several functions in the N-dimensional space. 

## Benchmark Functions
Several benchmark functions were used to evaluate the effectiveness of the algorithm. 

### Ackley
$$
f_{Ackley}(x) = - a \times e^{- b \times \sqrt{\frac{1}{n} \sum_{i=1}^{n} {x_{i}^{2}}}} \times e^{\frac{1}{n} \sum_{i = 1}^{n} \cos{2\pi x_{i}}} + c + e
$$

, $where\ a = 20, b = 0.2, c = 2\pi$ and n is the dimension.

| ![Ackley PSO gif](./media/ackley.gif)                         |
| --------------------------------------------------------------|
| Ackley Particle Swarm Optimisation in the 2-dimensional space |

### Griwank
$$
f_{Griewank}(x) = a + \frac{1}{b} \sum_{i = 1}^{n} {x_{i}^{2}} - \prod_{i = 1}^{n} {\cos \frac{x_{i}}{\sqrt{i}}}
$$

, $where\ a = 1, b = 4000$ and n is the dimension

| ![Griewank PSO gif](./media/griewank.gif)                         |
| ------------------------------------------------------------------|
| Griewank Particle Swarm Optimisation in the 2-dimensional space   |

### Michalewicz
$$
f_{Michalewicz}(x) = - \sum_{i = 1}{n} \sin {x_{i}} \times {(\sin {\frac {ix_{i}^{2}}{\pi}})}^{2m}
$$

### Rastrigin

### Rosenbrock

### Sphere
