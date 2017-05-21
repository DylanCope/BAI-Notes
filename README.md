

```python
import numpy as np
from copy import copy

import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
%matplotlib inline
pylab.rcParams['figure.figsize'] = (8.0, 8.0)
pylab.rcParams['font.size'] = 20
```

# Artficial CPGs using Nonlinear Oscillators

We can implement a controller that uses nonlinear dynamics to create a system that robustly tends towards a cyclic attractor in the state space. We will use the system of equations,

\begin{align}
    \tau \dot{v} =& -\alpha \frac{x^2 + v^2 - E}{E} v - x \\
    \tau \dot{x} =& v
\end{align}

where $x$ and $v$ are the position and velocity, and $\alpha, \tau$ and $E$ are positive real parameters (note: the notation, $\dot{v}$, indicates the time-derivative of the variable $v$). This oscillator has the interesting property that its limit cycle behavior is a sinusoidal signal with amplitude and period $2\pi \tau$. The state variable $x(t)$ indeed converges to $\hat{x}(t) = \sqrt{E}\,sin(t/\tau + \varphi)$ where $\varphi$ depends on the initial conditions.

These differential equations can be used in conjunction with Euler's approximation method to update state (velocity and position). The general update rule is of the form, 

$$ f_{n+1} = f_n + \delta t\, F( f_n, t_n ) $$

---

These are the equations derived in the paper, [Simulation and Robotics Studies of Salamander Locomotion: Applying Neurobiological Principles to the Control of Locomotion in Robots](
https://www.researchgate.net/publication/259195172_Simulation_and_Robotics_Studies_of_Salamander_Locomotion_Applying_Neurobiological_Principles_to_the_Control_of_Locomotion_in_Robots)



```python
def oscillator(
    # simulation parameters
    x0 = 0,
    v0 = 0,
    t0 = 0,
    tn = 4,
    dt = 0.01,

    # nonlinear parameters
    alpha = lambda t : 1,
    E = lambda t : 1,
    tau = lambda t : 1
    ):
    x, v, t = x0, v0, t0
    
    while t < tn:
        x += dt * v
        dvdt = -alpha(t) * v * (x**2 + v**2 - E(t)) / E(t) - x
        # euler's method for approximation,
        v += dt * dvdt
        yield t, x, v
        t += dt

    raise StopIteration
```


```python
def velocity_time_graphs( **kwargs ):
    D = np.array( [*oscillator( **kwargs )] )
    ts, xs, vs = D[ ..., 0 ], D[ ..., 1 ], D[ ..., 2 ]

    fig, axs = plt.subplots( 1, 2, figsize = (12, 4) )
    axs[0].plot( ts, xs )
    axs[0].set_title( 'Position vs Time')
    axs[0].set_xlabel( 'time' )
    axs[0].set_ylabel( 'position' )
    axs[1].plot( ts, vs )
    axs[1].set_title( 'Velocity vs Time')
    axs[1].set_xlabel( 'time' )
    axs[1].set_ylabel( 'velocity' )
    plt.tight_layout()
    plt.show()
    
velocity_time_graphs( tn = 30, v0 = 0.1 )
```


![png](Nonlinear%20Oscillators_files/Nonlinear%20Oscillators_3_0.png)


As shown, both the position and velocity fall into cyclical patterns. The intial condition is at $(0, 0.1)$ in the position-velocity state space, note that the only position that doesn't converge on towards the perfect oscillator is $(0, 0)$ as from the equations above $\tau \dot{v} = 0$, so 
$$ f_{n+1} = f_n + \delta t\, F( f_n, t_n ) =  f_n + \delta t\times 0 = f_n $$

We can see other points converging on the same cycle in state space by observing the trajectories from random initial conditions,


```python
np.random.seed( 0 )
num_samples = 30
samples = 4*np.random.rand( num_samples, 2 ) - 2
x0s, v0s = samples[ ..., 0 ], samples[ ..., 1 ]
plt.title( 'Random initial conditions' )
plt.xlabel( 'position' )
plt.ylabel( 'velocity' )
plt.scatter( x0s, v0s )
plt.show()
```


![png](Nonlinear%20Oscillators_files/Nonlinear%20Oscillators_5_0.png)



```python
for x, v in zip( x0s, v0s ):
    D = np.array( [*oscillator( tn = 30, x0 = x, v0 = v )] )
    xs, vs = D[ ..., 1 ], D[ ..., 2 ]
    plt.plot( xs, vs )
plt.title( 'Trajectories in the state space' )
plt.xlabel( 'position' )
plt.ylabel( 'velocity' )
plt.xlim([ -2.5, 2.5 ])
plt.ylim([ -2.5, 2.5 ])
plt.show()
```


![png](Nonlinear%20Oscillators_files/Nonlinear%20Oscillators_6_0.png)


## Dynamic Attractors

The cycle that all the trajectories tend towards is called an attractor. Now to see how the system can be controlled (i.e. the attractor changed), let's varying the oscillating parameters as a function of time. As an example, we define, 

$$
    E(t) = \begin{cases}
        1 & \text{t} < 50 \\
        9 & \text{otherwise}
    \end{cases}
$$

Therefore at time $t = 50$, the system will start tending towards an attractor that corresponds to a greater oscillation amplitude.


```python
Et = lambda t : 1 if t < 50 else 3**2
kwargs = dict( tn = 100, v0 = 2, E = Et )
D = np.array( [*oscillator( **kwargs )] )
xs, vs = D[ ..., 1 ], D[ ..., 2 ]
plt.plot( xs, vs )
plt.title( 'Trajectories in the state space' )
plt.xlabel( 'position' )
plt.ylabel( 'velocity' )
plt.xlim([ -4, 4 ])
plt.ylim([ -4, 4 ])
plt.show()
velocity_time_graphs( **kwargs )
```


![png](Nonlinear%20Oscillators_files/Nonlinear%20Oscillators_8_0.png)



![png](Nonlinear%20Oscillators_files/Nonlinear%20Oscillators_8_1.png)


Similarly, we can see how other parameters effect the transistions. We will have two amplitude jumps, but the second will transition be slower due to a reduced $\alpha$ value,

$$
    E(t) = \begin{cases}
        1 & \text{t} < 50 \\
        9 & \text{t} < 100 \\
        27 & \text{otherwise}
    \end{cases}
$$

$$
    \alpha(t) = \begin{cases}
        1 & \text{t} < 100 \\
        0.1 & \text{otherwise}
    \end{cases}
$$



```python
E_t = lambda t : 1 if t < 50 else (9 if t < 100 else 27)
alpha_t = lambda t : 1 if t < 100 else 0.1
kwargs = dict( tn = 200, v0 = 2, E = E_t, alpha = alpha_t )
D = np.array( [*oscillator( **kwargs )] )
xs, vs = D[ ..., 1 ], D[ ..., 2 ]
plt.plot( xs, vs )
plt.title( 'Trajectories in the state space' )
plt.xlabel( 'position' )
plt.ylabel( 'velocity' )
plt.xlim([ -6, 6 ])
plt.ylim([ -6, 6 ])
plt.show()
velocity_time_graphs( **kwargs )
```


![png](Nonlinear%20Oscillators_files/Nonlinear%20Oscillators_10_0.png)



![png](Nonlinear%20Oscillators_files/Nonlinear%20Oscillators_10_1.png)


# Coupling Nonlinear Oscillators

Here we will create a system of equations to update the internal states of multiple oscillators, whereby each affect each other. For each oscillator $i$, we define the dynamical equations,

\begin{align}
    \tau \dot{v_i} =& -\alpha \frac{x_i^2 + v_i^2 - E_i}{E_i} v_i - x_i \\
    &+ \underbrace{\sum_{j\neq i}(a_{ij}x_j + b_{ij}v_j)}_{\text{Influence of other oscillators}} 
     + \underbrace{\sum_j(c_{ij} s_j)}_{\text{Influence from external factors}} \\
    \tau \dot{x_i} =& v_i
\end{align}

where $a_{ij}$ and $b_{ij}$ are constants (positive or negative) determining how oscillator $j$ influences oscillator $i$.


```python
def coupled_oscillators(
    # simulation parameters
    x0,
    v0,
    t0 = 0,
    tn = 50,
    dt = 0.01,
    pos_influence_coefs = None,
    vel_influence_coefs = None,

    # nonlinear parameters
    alpha = lambda t : 1,
    E = lambda t : 1,
    tau = lambda t : 1
    ):
    '''
    '''
    x, v, t = x0, v0, t0
    num_oscillators = len( x0 )
    inconsistent_oscillators = Exception( 
        'Must have a consistent number of oscillators' )
    if len( x0 ) != len( v0 ): raise inconsistent_oscillators
    
    shape = ( num_oscillators,  num_oscillators )
    if pos_influence_coefs is None and num_oscillators > 1:
        pos_influence_coefs = np.zeros( shape )
    if vel_influence_coefs is None and num_oscillators > 1:
        vel_influence_coefs = np.zeros( shape )
    
    while t < tn:
        # compute influences before updating state
        influence = [
            sum([
                pos_influence_coefs[i, j] * x[j] +
                vel_influence_coefs[i, j] * v[j]
                for j in range( num_oscillators )
                if j != i
            ])
            for i in range( num_oscillators )
        ]
        # update internal state for each oscillator
        for i in range( num_oscillators ):
            x[i] += dt * v[i] / tau(t)
            c = (x[i]**2 + v[i]**2 - E(t)) / E(t)
            dvdt = -alpha(t) * v[i] * c - x[i] + influence[i]
            # euler's method for approximation,
            v[i] += dt * dvdt / tau(t)
        yield t, copy(x), copy(v)
        t += dt

    raise StopIteration
```


```python
x0, v0 = [1, .2], [2, -.1]
pos_influence_coefs = np.ones(( 2, 2 ))
vel_influence_coefs = -np.ones(( 2, 2 ))
data = [*coupled_oscillators( x0, v0, tn = 50, dt = 0.05,
                     pos_influence_coefs = pos_influence_coefs,
                     vel_influence_coefs = vel_influence_coefs )]
_, axs = plt.subplots( 1, 2, figsize = (12, 4) )
_, state_space = plt.subplots()

ts = [ t for t, *_ in data ]
for i in range(len(x0)):
    xs = [ x[i] for _, x, _ in data ]
    vs = [ v[i] for _, _, v in data ]
    axs[0].plot( ts, xs )
    axs[1].plot( ts, vs )
    state_space.plot( xs, vs )

axs[0].set_title( 'Position vs Time')
axs[0].set_xlabel( 'time' )
axs[0].set_ylabel( 'position' )
axs[1].set_title( 'Velocity vs Time')
axs[1].set_xlabel( 'time' )
axs[1].set_ylabel( 'velocity' )
state_space.set_title( 'Trajectories in state space' )
state_space.set_xlabel( 'position' )
state_space.set_ylabel( 'velocity' )
state_space.set_xlim([ -1.8, 2.2 ])
state_space.set_ylim([ -2, 2 ])

plt.tight_layout()
plt.show()
```


![png](Nonlinear%20Oscillators_files/Nonlinear%20Oscillators_13_0.png)



![png](Nonlinear%20Oscillators_files/Nonlinear%20Oscillators_13_1.png)


Here the influence matrices have been set up in such a way to push the oscillators into anti-phase behaviours. This is somewhat analgous to two legs running, with one foot reaching a maximum in while the other a minimum. 
