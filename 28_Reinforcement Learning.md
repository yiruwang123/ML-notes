# Reinforcement Learning

> 强化学习=Actor+Critic，其中训练Actor的目标是最大化total reward，参数更新过程类似加权的cross entropy；Critic的求法有两种，MC/TD，训练过程中固定Actor，希望V值与真实的reward之差(当前->结束 or 相邻两个)越接近越好

#### Introduction

##### Scenario

强化学习

AI=RL+DL=Deep Reinforcement Learning

RL由机器(Agent)和环境(Environment)组成，机器会去观察这个世界的种种变化，被称为observation (state)，机器会做一些行为(Action)，它跟环境的互动会对环境造成一些影响，而环境也会给机器反馈(Reward)

比如机器看到一杯水，并把水打翻了，环境就会给它一个负面的reward，告诉它不要这么做；接下来机器看到的是打翻的水，它把地板上的水擦干净，就可以得到正面的reward

注：在RL中state表示机器所看到的东西，是指环境的状态，而非机器本身的状态，机器有可能无法看到整个环境的状态

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/rl.png" width="60%"></center>

在整个过程中，机器一直要去学习那些可以让reward最大化的action

##### Alpha Go

以Alpha Go为例，机器的observation是19\*19的棋盘，要做的action是落子的位置，environment就是机器的对手，每次落子之后对手就会做出反应来reward

下棋的过程中，大多数的情况下reward其实是0，只有当赢的时候Reward=1，输的时候Reward=0

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/rl-alpha.png" width="60%"></center>

做RL困难之处在于，很多时候Reward是sparse的，只有少数情况你才能得到有用的Reward，机器如何在只能得到少数Reward的情况下发现正确的action，这是一个困难的问题

在supervised learning中，你会明确地告诉机器，看到某个特定的盘式时下一步要怎么走

- 机器在老师的教导下学习

而在reinforcement learning中，往往会遇到人也不知道该怎么处理的盘式，有时候棋谱上的方案也不一定是最优的，机器需要自己去尝试，通常是两个机器互相博弈训练

- 机器从过去的经验中学习，没有老师告诉它什么是好的什么是不好的，需要自己去体悟

Alpha Go=supervised learning+reinforcement learning

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/rl-alpha2.png" width="60%"></center>

##### Chat-bot

在supervised learning中，聊天机器人会使用Seq2Seq的方式训练

而在reinforcement learning中，聊天机器人会和人直接进行对话，根据人反应的情绪来作为reward

但实际上不可能有人会跟机器做成千上万次没有意义的对话，还是需要像Alpha Go一样让两个聊天机器人相互对话，由于聊天结果并不像棋盘一样有明确的输赢之分，训练起来还是有些困难

当然，我们可以定一些简单的对话规则去对机器人的聊天做评价，参考: *[Deep Reinforcement Learning for Dialogue Generation](https://arxiv.org/pdf/1606.01541v3.pdf)*

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/rl-chat.png" width="60%"></center>

##### Playing Video Game

机器还可以通过RL学习打游戏，此时它观察到的是图片的pixel，所做的action就是在游戏中的动作，reward就是当前的得分

每一个游戏回合叫做episode，机器要做的就是如何在一个episode中得到最高的reward

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/rl-game.png" width="60%"></center>

##### Properties of RL

RL有两个难点：

- Reward的出现会有delay

    比如在坦克的游戏中，只有开火之后才会得到reward，因此机器就只会疯狂开火

    但实际上左移、右移本身虽然不能让机器直接得到reward，但可以帮助它在未来得到reward

    所以机器需要有远见，才能把游戏玩得比较好

- Agent所采取的行为会影响到它之后看到的东西

    Agent要学会去探索这个世界，去尝试所有没有做过的行为，以此来做一个全局的衡量

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/rl-pro.png" width="60%"></center>

#### Outline

很多课程会讲Markov Decision Process或是过去在RL中很常用的Deep Q Network，但这里会先介绍目前最潮流的A3C技术

Reinforcement Learning的方法分为两大块：

- Policy-based Method：训练一个专门负责做事的Actor

- Value-based Method：训练一个只会批评但不做事的Critic


两者相加就得到了Actor-Critic方法

关于model-based方法，就是预测接下来会发生什么事情，在棋类游戏中比较有用，但在传统的电竞游戏中并没有很大的用处

目前最强的方法就是Asynchronous Advantage Actor-Critic，简称A3C

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/rl-outline.png" width="60%"></center>

#### Policy-based Approach

Learning a Actor

##### Look for a Function

RL也是找一个function，它的input是机器看到的Observation，output是机器要采取的Action，要通过reward帮我们找出Actor/Policy

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/rl-actor.png" width="60%"></center>

##### Neural Network as Actor

NN作为Function/Actor的基本结构

在游戏中，如果把neural network当做actor，则观察到的图像pixels vector当做NN的input，采取的action当做NN的output，output的维数由可以采取的行动数量决定

把一张图像丢进NN里，就会输出每个action对应的得分，你可以选择得分最高的action，而在做Policy gradient的时候，我们通常会假设actor是随机的(stochastic)，它的output表示的是采取某个动作的概率

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/rl-actor2.png" width="60%"></center>

在传统的做法里，我们会用一张table来存储面对不同observation所应该采用的action，但通常是无法穷举所有可能的情况，所以用NN的表现会更好，而且就算是没有看过的东西，NN也能给出一个合理的结果

##### Goodness of Actor

在RL中，定义Actor的好坏与在监督学习中定义function好坏的过程是非常类似的

Actor就是NN，用$\pi_{\theta}(s)$表示，其中$s$是机器看到的observation，$\theta$是Actor的参数

接下来让Actor实际地去玩一局游戏，计算total reward：$R_{\theta}=\sum_{t=1}^Tr_t$

我们的目标不是最大化每一步的reward，而是最大化整局游戏的total reward

就算每次都是同个Actor玩游戏，得到的$R_{\theta}$也会是不同的：

- Actor是随机的，即使看到同样的场景也会采取不同的action
- 游戏本身有随机性，每次看到的场景也会不一样

因此我们要去maximize的不是每次游戏结束后的$R_{\theta}$，而是它的期望值$\bar R_{\theta}$

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/rl-actor3.png" width="60%"></center>

这里用$\tau$来表示一场游戏，它是一个序列，包含了state $s_i$、action $a_i$和reward $r_i$，$R(\tau)$则表示total reward

$\tau$代表从游戏开始到结束的某种可能的过程，当你使用Actor进行游戏时，由于某些游戏场景特别不容易出现，可能只会经历几种不同的过程

每种可能的过程都用概率$P(\tau|\theta)$来描述，其中不同的$\theta$决定了不同的Actor

此时$R(\theta)$的期望就变成了$\bar R(\theta)=\sum\limits_{\tau} R(\tau) P(\tau | \theta)$ 

- $P(\tau|\theta)$是$\tau$发生的概率
- $R(\tau)$是$\tau$的reward

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/rl-actor5.png" width="60%"></center>

实际上我们无法穷举$\tau$，但可以让Actor去玩N局游戏，得到$\{\tau^1,\tau^2,...,\tau^N \}$，算出N个reward，做平均值去近似期望

这个过程可以被认为是从无穷多种可能的$\tau$中取样N个样本点，发生概率越高的$\tau$越容易被取样

##### Pick the Best Actor

在得到对Actor的衡量目标之后，我们要做的就是寻找参数$\theta^*$去最大化$\bar R_{\theta}$，使用的方法是梯度上升法(gradient ascent)

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/rl-ga.png" width="60%"></center>

先去计算$\nabla \bar R_{\theta}$，由于$R(\tau)$与$\theta$没有关系，即使不可微或者不知道具体表达式也没有影响，因此运算过程中我们只需对$P(\tau|\theta)$求微分即可：
$$
\nabla \bar R_{\theta}=\sum\limits_{\tau} R(\tau) \nabla P(\tau|\theta)\\
=\sum\limits_{\tau} R(\tau) P(\tau|\theta)\frac{\nabla P(\tau|\theta)}{P(\tau|\theta)}\\
=\sum\limits_{\tau} R(\tau) P(\tau|\theta)\nabla ln P(\tau|\theta)\\
≈\frac{1}{N} \sum\limits_{n=1}^N R(\tau^n) \nabla ln P(\tau^n|\theta)
$$
实际上$R(\tau)$确实是个黑盒子，因为环境是不确定的，reward当然也是不确定的

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/rl-pg.png" width="60%"></center>

接下来还需要计算$\nabla ln P(\tau^n|\theta)$，对$P(\tau|\theta)$来说，某个游戏过程可以被看成是$s_1$发生的概率 × $s_1$发生的情况下采取$a_1$的概率 × $s_1,a_1$发生的情况下发生$s_2$的概率...用$\prod$表示如下：
$$
\begin{split}
P(\tau|\theta)&=p(s_1)p(a_1|s_1,\theta)p(r_1,s_2|s_1,a_1)p(a_2|s_2,\theta)p(r_2,s_3|s_2,a_2)...\\
&=p(s_1)\prod\limits_{t=1}^T p(a_t|s_t,\theta)p(r_t,s_{t+1}|s_t,a_t)
\end{split}
$$
下图中划橙线的项是跟Actor没有关系的，只有划红线的含$\theta$的那一项是跟Actor有关系的

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/rl-actor4.png" width="60%"></center>

然后对上式取$\ln()$即可：$\ln p(s_1)\prod\limits_{t=1}^T p(a_t|s_t,\theta)p(r_t,s_{t+1}|s_t,a_t)$，化简结果如下图所示

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/rl-pg2.png" width="60%"></center>

注意到，跟$\theta$无关的项都是由游戏决定的，与gradient无关，因此可以被删去，化简为
$$
\nabla ln P(\tau|\theta)=\sum\limits_{t=1}^T \nabla ln P(a_t|s_t,\theta)
$$
最终$\nabla \bar R_{\theta}$可以被化简成
$$
\nabla \bar R_{\theta}≈\frac{1}{N}\sum\limits_{n=1}^N R(\tau^n) \nabla ln P(\tau^n|\theta)\\
=\frac{1}{N} \sum\limits_{n=1}^N \sum\limits_{t=1}^{T_n} R(\tau^n) \nabla ln P(a_t^n|s_t^n,\theta)
$$
这个式子其实是很直观的：

- 假如在玩$\tau^n$这次游戏的过程中，机器在看到$s_t^n$场景时采取了$a_t^n$的动作，使得整场游戏得到一个好的结果reward $R(\tau^n)$，那我们就会想调整参数$\theta$让这种情况发生的概率$p(a_t^n|s_t^n)$越大越好

- 反之如果得到坏的结果，就希望让这个概率越小越好

- 需要注意，每个时间点发生的动作概率需要乘上整场游戏的reward，而不是该时间点的reward

    因为我们需要考虑全局观，而不是单纯考虑局部收益

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/rl-pg3.png" width="60%"></center>

Q：那为什么要对概率取$\ln()$呢？

A：实际上在机器进行N次游戏的时候，会遇到很多相同的游戏场景，假设$\tau^{13}$、$\tau^{15}$、$\tau^{17}$、$\tau^{33}$遇到的游戏场景是相同的，但由于机器会随机选取不同的action，因此最终得到的reward $R(\tau^n)$可能会各不相同，假设不同游戏在同一个场景下由于采取了两种不同的action会得到不同的total reward：

- action a：$R(\tau^{13})=2$
- action b：$R(\tau^{15})=1$、$R(\tau^{17})=1$、$R(\tau^{33})=1$

如果不采用$\ln()$，而是单纯累加，即使第13次游戏得到的reward比较好，但由于action a比较罕见，调高它发生的概率所得效益并没有那么明显，因此机器就转而想要提高action b出现的概率

而采用$\ln p=\frac{\nabla p}{p}$，在分母除以p就相当于做归一化(normalization)，发生概率(次数)越高的action，除掉的值就越大，这样机器在做优化的时候就不会偏好那些出现概率(次数)比较高的action

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/rl-pg4.png" width="60%"></center>

##### Add a baseline

如果$R(\tau)$都是正的，那更新的时候每个action的概率都会被放大

在理想情况下，3个action都会被照顾到，如果原先$b$的$R(\tau)$比较大，$a$、$c$的$R(\tau)$比较小，经过update以后，还是会使得$b$、$a$、$c$出现的概率变大，只不过在做归一化的时候，原先概率大的增幅会比较小

但在实作的时候，我们是对$a$、$b$、$c$做随机采样，很可能游戏过程中$a$几乎就没有出现，而$b$、$c$则频繁出现，这会使得只有$b$、$c$的概率在增加，这样就会出问题

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/rl-baseline.png" width="60%"></center>

为了解决这个问题，我们需要让$R(\tau)$有正有负，在$\nabla \bar R_{\theta}$的式子中加一个bias作为baseline，只有当$\tau^n$得到的reward好过baseline，才把对应的action概率增加，这样就可以避免某个action没有被sample到导致发生的概率相对其他action变小

##### Process for Policy Gradient

做Policy Gradient的时候，参数更新过程如下：

- 玩N次游戏，分别得到$\tau^1$、$\tau^2$、...、$\tau^N$，并记录每局游戏的场景$s$、动作$a$、reward $R(\tau)$
- 代入$\nabla \bar R_{\theta}$的式子，对参数$\theta$做更新
- 再重复上述操作

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/rl-pg5.png" width="60%"></center>

回顾分类问题，损失函数通常是交叉熵$-\sum\limits_{i=1}^3 \hat y_i log y_i$

去掉负号，就相当于在maximize $\log y_i$，在下图中就是$\log P(left|s)$，随后计算gradient并做update
$$
\theta \leftarrow \theta+\eta \nabla \log P(left|s)
$$

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/rl-pg6.png" width="60%"></center>

接下来我们解释$\frac{1}{N} \sum\limits_{n=1}^N \sum\limits_{t=1}^{T_n} R(\tau^n) \nabla ln P(a_t^n|s_t^n,\theta)$的含义

假设reward $R(\tau^n)$始终为1，从式子中舍去这一项，而$(s_1^1,a_1^1)$的过程就相当于输入$s_1^1$，让NN的输出$a_1^1$中“left”的概率尽量接近于1，其余概率接近于0，此时这个问题就转变成了分类问题，$(state,action)$的组合就变成了$(input,target)$的组合

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/rl-pg7.png" width="60%"></center>

与分类问题唯一不同的是，**Policy Gradient会在每个分类问题前再乘上$R(\tau^n)$，它代表某个样本点要被复制几次**

假设$R(\tau^1)=2$，$R(\tau^2)=1$，整个过程就相当于：

- 把$\tau^1$里的样本点复制2次、把$\tau^2$里的样本点复制1次作为training data
- 去训练神经网络，更新参数
- 重新采样N次游戏的数据，并计算$R(\tau)$
- 重复上述操作

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/rl-pg8.png" width="60%"></center>

因此在实作的时候，你只需要在训练分类器的时候，给training data加上$R(\tau^n)$的权重即可

一般的分类问题训练一次NN就够了，而在强化学习中则要训练NN很多次

强化学习其实并不复杂，整个过程就是在做**重复分类**，每次都要收集新的data、训练神经网络，再收集、再训练...

#### Value-based Approach

Learning a Critic

##### Define Critic

Critic只负责批评但不负责做事，如果从Critic中学到Actor，就叫做Q-learning

给定Actor $\pi$，需要训练一个函数 $V^{\pi}(s)$，它会对看到的state好坏作出评价，也就是给出从state $s$开始到游戏结束之间的reward大小的期望值

注意：不同的Actor $\pi$会对结果产生不同的影响

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/rl-critic.png" width="60%"></center>

##### How to estimate $V^{\pi}(s)$

- Monte-Carlo based approach(**MC**)

    蒙特卡洛法，让Actor $\pi$去跟环境互动，Critic观察并统计不同情况下的reward，假设$G_a$是从state a到游戏结束期间的reward，则希望$V^{\pi}(s_a)$与$G_a$越接近越好

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/rl-critic2.png" width="60%"></center>

- Temporal-difference approach(**TD**)

    时序差分法，只观察互动中的一小段，假设$r_t$是两个相邻状态$s_t$和$s_{t+1}$的reward之差，则利用差分思想，希望$V^{\pi}(s_t)-V^{\pi}(s_{t+1})$与$r_t$越接近越好

    如果某种活动永远没有结束的那一刻，这种方法就比较适合

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/rl-critic3.png" width="60%"></center>

##### MC v.s. TD

MC的方法会积累state $s_a$后的所有噪声直至游戏结束，它的variance是比较大的，但它会把所有的偏差平均起来，因此是bias会比较小

TD的方法只会受到state $s_{t}$和$s_{t+1}$之间的噪声影响，它的variance是比较小的，但它会受到周围两个状态的偏差影响，因此bias会比较大

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/rl-critic4.png" width="60%"></center>

#### Actor-Critic

A3C=Asynchronous Advantage Actor-Critic

##### Introduction

在Actor-Critic中，$\pi$作为Actor去和环境互动，收集很多data，然后利用TD/MC的方法去训练出$V^{\pi}(s)$，更新$\pi$为$\pi'$，重复上述流程

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/rl-actor-critic.png" width="60%"></center>

##### Advantage Function

在Actor部分，我们已经给出了$\nabla \bar R_{\theta^{\pi}}$的表达式，原来的$R(\tau^n)$考虑了整个互动过程中的reward，其实在A3C中，$R(\tau^n)$可以用Critic部分的内容来表示：
$$
R(\tau^n)=r_t^n-(V^{\pi}(s_t^n)-V^{\pi}(s_{t+1}^n))
$$
其中$r_t^n$表示在state $s_t^n$下采取action $a_t^n$而获得的reward，而$V^{\pi}(s_t^n)-V^{\pi}(s_{t+1}^n)$是由Critic估测出的reward，两者相减的结果，就叫做**Advantage Function**

- 如果Advantage Function结果为正，则要增加action $a_t^n$的概率
- 如果Advantage Function结果为负，则要减少action $a_t^n$的概率

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/rl-actor-critic2.png" width="60%"></center>

##### Tips for “A2C”

Actor $\pi(s)$和Critic $V^{\pi}(s)$的部分参数是可以共享的

- 输入的$s$实际上是一张游戏的画面
- 前期处理图像的CNN是可以被Actor和Critic共享的

希望Actor的output entropy是比较大的

- 使用entropy做regularization
- entropy大意味着Actor的输出分布不是集中的，而是比较平滑的
- entropy大意味着可以让Actor尽可能探索所有可能的情况

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/rl-actor-critic3.png" width="60%"></center>

##### Asynchronous for A3C

$\theta^1$是Global Network的参数，同时也是共有的参数，同时连接着许多Worker(包含了Actor+Critic)，它们复制全局参数$\theta^1$并与环境互动，并把参数更新相关的数据$\Delta \theta$传回Global Network，令全局参数更新

需要注意的是，由于是多个Workers并行运行，因此会同时将多个不同的$\Delta \theta$传入Global Network，此时参数更新为$\theta^2+\eta \Delta \theta$



<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/rl-actor-critic4.png" width="60%"></center>

#### Q Learning

##### Introduction

之前的Critic $V^{\pi}(s)$表示state $s$之后的reward期望值，此外，还有另一种Critic $Q^{\pi}(s,a)$表示在state $s$采取action $a$之后的reward期望值

当action可以穷举时，也可以用下图右侧的形式表示

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/rl-critic-another.png" width="60%"></center>

在原始的Original Actor-Critic中，由于$a_1$的$A^{\pi}(a_1)$值小于$a_2$的$A^{\pi}(a_2)$值，因此机器会减小$a_1$发生的概率，进而增大$a_2$发生的概率，但问题是，没有被使用过的好的$a_i$(如红线处)就会失去增加概率的机会

实际上$Q^{\pi}(s,a)$这个函数的参数我们是知道的，它本质就是个NN，因此在函数准确的前提下，我们完全可以直接得到该函数的最高点，这样机器就不需要穷举每个action以求最后的reward，这个方法就叫做**Q-Learning**

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/rl-critic-another2.png" width="60%"></center>

##### New Critic

这个新Actor-Critic的结构图如下，通过$Q^{\pi}(s,a)$可以找到一个比$\pi$更“好”的Actor $\pi'$

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/rl-critic-another3.png" width="60%"></center>

这里的“Better”指的是，对所有的state $s$来说，$V^{\pi'}(s)\geq V^{\pi}(s)$

注意，$\pi'$并没有额外的参数，Q function就直接决定了$\pi'$
$$
\pi'(s)=\arg \max_a Q^{\pi}(s,a)
$$
如果action是连续的，那这种方法就不适用，而是每次都需要用梯度上升法求让Q值最大的$a$

上述理论的证明如下：

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/rl-critic-Q.png" width="60%"></center>

其中，在$V^{\pi}(s)$的式子中，Actor $\pi$遇到state $s$，接下来要采取的action为$\pi(s)$，因此$V^{\pi}(s)$就等价于$Q^{\pi}(s,\pi(s))$，它一定会小于等于Q值最大的情况，而这种情况对应的action $a$为$\pi'(s)$

故得到关系$V^{\pi}(s)\leq Q^{\pi}(s,\pi'(s))$，经过一些列推导可得$V^{\pi}(s)\leq V^{\pi'}(s)$

也就是说，假设给我们一个Actor $\pi$，我们就可以计算出它的Q-function $Q^{\pi}(s,a)$，那我们就肯定可以找到另外一个比原先的$\pi$更好的$\pi'$，它们之间满足$V^{\pi}(s)\leq V^{\pi'}(s)$

##### Estimate $Q^{\pi}(s,a)$ by TD

关于如何估测$Q^{\pi}(s,a)$函数，和之前类似，可以采用MC的方法也可以采用TD的方法

下图展示了使用TD的方法估测的过程

如果只给了$a_t$，其实并不知道机器下一步会采取的action $a_{t+1}$是什么，但我们可以用$\pi(s_{t+1})$来估测

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/rl-critic-Q2.png" width="60%"></center>

##### Double DQN

在做Q Learning的时候，很容易会高估Q值

由于我们在训练的时候，以$r_t+\max\limits_a Q(s_{t+1},a)$作为$Q(s_t,a_t)$的target，而往往$\max\limits_a Q(s_{t+1},a)$是一个被高估的Q值，因此最终训练得到的$Q(s_t,a_t)$也会被高估

- 下图左侧，1、3被高估，2、4不变，最终1被选取
- 下图右侧，2、4被高估，1、3不变，最终4被选取

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/rl-DQN.png" width="60%"></center>

为了解决这个问题，提出了Double DQN的做法

原先的做法是用同一个Q来决定action和计算Q值，而在Double DQN中需要用到两个Q function：Q和Q‘，其中一个决定要采取哪个action，另外一个去计算Q值

Q和Q’互相制衡：

- Q可以提出要用哪个action，但它不能决定action的value
- Q’可以决定action的value，但它不能决定要用哪个action
- 除非Q和Q‘同时高估某个action的值，否则得到的结果将会是均衡的

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/rl-DQN2.png" width="60%"></center>

##### Dueling DQN

下图分别展示了原始的Q Learning Network和Dueling DQN的架构

Dueling DQN其实只是改变了Q Network的后几层架构，它在倒数第二层分布输出scalar $V(s)$和vector $A(s,a)$，其中$A(s,a)=Q(s,a)-V(s)$

Dueling DQN得到的结果往往会比DQN要好

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/rl-DQN3.png" width="60%"></center>

##### Pathwise Derivative Policy Gradient

其实我们可以把Actor $\pi$的network和Critic $Q^{\pi}$的network连接在一起

在训练Actor $\pi$的时候，就固定住Critic $Q^{\pi}$的参数，再去做梯度上升

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/rl-policy-gd.png" width="60%"></center>

这个网络的结构和GAN很相似，Actor和Critic的关系类似Generator和Discriminator的关系

##### DDPG Algorithm

下面比较具体地介绍下DDPG算法，Deep Deterministic Policy Gradient，是Pathwise Derivative Policy Gradient的一种方式

- 初始化Critic network的参数为$\theta^Q$，Actor network的参数为$\theta^{\pi}$

- 初始化目标Actor $\pi'$和Critic $Q'$的参数$\theta^{Q'}$和$\theta^{\pi'}$为$\theta^Q$和$\theta^{\pi}$，以及replay buffer R

- actor与环境做互动，此外加上噪声得到$\pi(s)+noise$来帮助actor来探索环境，得到的每笔training data为$\{s_t,a_t,r_t,s_{t+1}\}$

- 从R中取样出N笔training data，去训练Critic Q

- 更新Critic Q的参数，使它的输出$Q(s_n,a_n)$跟目标$\hat y_n$接近，最小化$L=\sum_n(\hat y_n-Q(s_n,a_n))^2$

    其中，$\hat y_n=r_n+Q'(s_{n+1},\pi'(s_{n+1}))$，$\pi'$由初始的参数值$$\theta^{\pi'}$$决定

- 更新Actor $\pi$的参数，使它的输出可以让Q function的值增加，最大化$J=\sum_nQ(s_n,\pi(s_n))$

- 更新目标Actor $\pi'$和Critic $Q'$的参数，理论上可以直接把它们设为$\pi$和$Q$，但实际上我们会把$\pi'$、$\pi$加权赋给新的$\pi'$，Q'也同理

    这样会使得虽然$\pi$、$Q$的变化比较快，但$\pi'$和$Q$的变化比较缓慢，从而保持训练的稳定

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/rl-DDPG.png" width="60%"></center>

#### GAN and Actor-Critic

下图是GAN和Actor-Critic用到技术的对比

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/rl-gan-ac.png" width="60%"></center>

可参考的博客：https://www.cnblogs.com/wangxiaocvpr/p/8110120.html