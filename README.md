# Task:

manipulate robot to conduct a valid thrust toward opponent

### Initialize Dojo_env

<img src="assets/ini.png" style="zoom:50%;" />

### State Space

$dim(S) = 16$

<img src="assets/state_space.png" style="zoom:50%;" />

### Action Space

$dim(A) = 4$

<img src="assets/action_space.png" style="zoom:50%;" />

### vaild thrust judgment

In order to restore the rules and manner of tankendo as closely as possible, a vaild thrust is defined as follows:

<img src="assets/judgment.png" style="zoom:50%;" />

### Reward function(Sparse)

One of the challenges of the topic is to use only sparse reward functions:
$$
r = \cases
{
0,& \text{if done}\\
-1,& \text{else}
}
$$
In particular, we give more negative reward on several harmful states. 

<img src="assets/harmful state.png" style="zoom:50%;" />
$$
r \text{+=} \cases
{
-3,& \text{invalid thrust}\\
-2,& \text{collision}\\
-2,& \text{outrange}
}
$$

### Demonstration from rule-based model

​	Another major challenge is the difficulty in collecting human expert data. The main idea of the project is to use non-optimal rule-base demonstrations to provide early assistance in training.

<img src="assets/non-optimal demo.png" style="zoom:50%;" />

​	the model is to define several sets of points that are not included in the observable part of the state space. The agent will be guided to the thrust point, sprint straight, and finally stop at the stop point to complete the thrust.

​	Without adding noise, the success rate is **32.5%**, and with adding noise, the success rate is **17.2%**

# method：

### backbone:

SAC with PER， a modified version from：https://github.com/BY571/Soft-Actor-Critic-and-Extensions

### pseudo code (#ToBeNamed)

<img src="assets/pseudo.png"  />

### accumulate reward (#ModifyFont)

The proposed method is shown in orange.

<img src="assets/injection_experiment.png" style="zoom: 33%;" />

<img src="assets/lambda_experiment.png" alt="lambda_experiment" style="zoom:33%;" />

### render

<img src="assets/output.gif" style="zoom:33%;" />

The learning success rate is **94.06%** when using the **17.2%** non-optimal demonstrations.

The remaining parameters, overfitting analysis as well as extended experiments will be further mentioned in the paper(#TODO).

