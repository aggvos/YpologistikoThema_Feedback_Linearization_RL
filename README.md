# Reinforcement Learning-Based Feedback Linearization Under Model Uncertainty

MATLAB implementations of **learning-based feedback linearization** for two nonlinear robotic systems:

- A fully actuated **double pendulum**
- The **planar Cepheus space robot**, with a base actuated by a single reaction wheel and a three joint robotic manipulator

This project investigates whether reinforcement learning can compensate for model uncertainties in feedback-linearizing controllers. It builds on the framework proposed by Westenbroek et al. and implements two policy gradient reinforcement learning algorithms: **REINFORCE** and **Proximal Policy Optimization (PPO)**.

## Requirements

- MATLAB
- Symbolic Math Toolbox (for symbolic derivation of the Cepheus dynamics)

## 1. Double Pendulum

The first system is a fully actuated double pendulum that was studied in the work by Westenbroek et al. It serves as an initial simpler system for implementing the method.

The MATLAB implementation includes:

- **Nonlinear dynamics and state-space modeling**
  ```matlab
  dynamics/double_pendulum_matrices.m  
  ```
  ```
  dynamics/double_pendulum_model_state_space.m
  ```

- **Nominal and learned feedback-linearizing controllers**
  ```matlab
  fl_controller/fl_controller.m 
  ```
  ```
  fl_controller/learned_fl_controller.m 
  ```
  ```
  fl_controller/compute_virtual_input.m
  ```

- **Policy features and reference trajectories**
  ```matlab
  policy_features.m
  ```
  ```
  reference_from_task.m
  ```

- **REINFORCE and PPO training implementations**
  ```matlab
  REINFORCE/train_theta_REINFORCE.m 
  ```
  ```
  PPO/train_theta_PPO.m
  ```

- **Simulation**
  ```matlab
  simulation/rollout_simulation.m
  ```
 

The main script used for testing the controllers is:

```matlab
main_test_model.m
```

## 2. Planar Cepheus Space Robot

The second system is a planar model of the Cepheus space robot, consisting of a spacecraft base actuated by a reaction wheel and a three-joint robotic manipulator.

This system provides a more complex validation case for the learning-based feedback linearization method due to the coupled dynamics between the base and manipulator.

The MATLAB implementation includes:

- **Symbolic derivation of the dynamics and state-space modeling**
  ```matlab
  model/planar2D_model_matrices_derivation.m
  ```
  ```
  model/cepheus_model_state_space.m
  ```
  ```
  model/cepheus_H.m
  ```
  ```
  model/cepheus_c.m
  ```

- **Nominal and learned feedback-linearizing controllers**
  ```matlab
  fl_controller/cepheus_fl_controller.m 
  ```
  ```
  fl_controller/cepheus_learned_fl_controller.m
  ```
  ```
  fl_controller/cepheus_virtual_input.m
  ```

- **Policy features and reference trajectories**
  ```matlab
  cepheus_policy_features.m  
  ```
  ```
  cepheus_reference_from_task.m
  ```

- **REINFORCE and PPO training implementations**
  ```matlab
  REINFORCE/cepheus_train_theta_REINFORCE.m
  ```
  ```
  PPO/cepheus_train_theta_PPO.m
  ```

- **Simulation**
  ```matlab
  cepheus_rollout_simulation.m
  ```


The main script used for testing the controllers is:

```matlab
main_test_cepheus.m
```

### Comparison with Skandalakis et al. (MED 2026)

Additional experiments use the trajectory tracking task considered by Skandalakis et al. (MED 2026), allowing the learned feedback-linearizing controllers to be evaluated under the same scenario.

The corresponding scripts are:

```matlab
cepheus_train_theta_PPO_skandalakis.m
```
```
cepheus_train_theta_REINFORCE_skandalakis.m
```
```
compare_skandalakis_cepheus.m
```

## 3. Technical Report

A technical report containing the mathematical formulation, implementation, training procedures and simulation results is included (written in Greek):

[Technical Report — Ypologistiko_Thema.pdf](Ypologistiko_Thema.pdf)



## 4. References

1. T. Westenbroek, D. Fridovich-Keil, E. Mazumdar, S. Arora, V. Prabhu, S. S. Sastry, and C. J. Tomlin, "Feedback Linearization for Uncertain Systems via Reinforcement Learning," *IEEE International Conference on Robotics and Automation (ICRA)*, 2020, pp. 1364–1371. [DOI](https://doi.org/10.1109/ICRA40945.2020.9197158)

2. N. Skandalakis, K. Nanos, and I. Poulakakis, "An L1-Adaptive Control Lyapunov Function Quadratic Program for Torque Constrained Space Manipulators under Uncertainty," *34th Mediterranean Conference on Control and Automation (MED)*, 2026, pp. 698–703. [DOI](https://doi.org/10.1109/MED70602.2026.11598542)

