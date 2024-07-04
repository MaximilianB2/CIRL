# CIRL: Integrating PID Controllers into the deep reinforcement learning framework.

This repository contains the code used in the paper title "Control-Informed Reinforcement Learning: Integrating PID Controllers and Deep Reinforcement Learning" by Maximilian Bloor, Akhil Ahmed, Niki Kotecha, Mehmet Mercangöz, Calvin Tsay, Ehecatl Antonio Del Rio Chanona.


<p align="center">
  <img src="./plots/RL-PID Diagram.png" width="50%">
</P>

## Abstract

This work proposes a control-informed reinforcement learning (CIRL) framework that integrates proportional-integral-derivative (PID) control components into the architecture of deep reinforcement learning policies. The proposed approach augments deep RL agents with a PID controller layer, incorporating prior knowledge from control theory into the learning process. By leveraging the properties of PID control, such as disturbance rejection and tracking capabilities, while preserving the nonlinear modeling capacity of deep RL, CIRL aims to improve the performance and robustness of reinforcement learning algorithms. Simulation studies conducted on a continuously stirred tank reactor system demonstrate the improved performance of CIRL compared to both conventional model-free deep RL and static PID controllers. CIRL exhibits better setpoint tracking ability, particularly when generalising to trajectories outside the training distribution, indicating its enhanced generalization capabilities. Furthermore, the embedded prior control knowledge within the CIRL policy improves its robustness to nonobservable system disturbances. The control-informed RL framework combines the strengths of classical control theory and reinforcement learning to develop sample-efficient and robust deep reinforcement learning algorithms, with potential applications in complex industrial systems.

## Citation

```
@article{cirl2024,
  author = {Max Bloor and Akhil Ahmed and Niki Kotecha and Mehmet Mercangoz and Calvin Tsay and Ehecatl Antonio Del Rio Chanona},
  title = {Control-Informed Reinforcement Learning: Integrating PID Controllers and Deep Reinforcement Learning},
  year = {2024},
}
```