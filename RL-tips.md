# RL-tips

#### Tip1: Add a Baseline

##### classification

RL本质上就是加了reward权重的分类问题：$\sum R(\tau^n) \log p(a^n_t|s^n_t)$

表示如果你在$s_t$这个state采取了$a_t$这个action会给整场游戏正面的reward，就要增加它发生的几率

##### reward is always positive 

但在一些游戏中很有可能reward总是正的，这就导致了不管是什么action，机器都会去提高它发生的概率，当然这并不一定会有问题，因为正的reward也有大有小

比如在同一个state下，可以采用的action有三个选项：$a$、$b$、$c$，当某场游戏的reward值比较大时，对应的action的log probability就会上升得比较大，由于$a$、$b$、$c$发生的概率总和为1，因此上升的少，在做完normalize之后意味着概率会下降

<center><img src="../img/rl-tip1.png" width="60%"/></center>

但这些都是理想状况

##### add a baseline

实际上我们做的是sampling，在同一个state下的$a$、$b$、$c$三个action中，可能有一些是你从来都没有采样到的，那些没有被sample到的action不是本身不好，只是运气不好，所以这个算法存在一些问题

为了让reward不总是正的，我们可以给它减去一个baseline $b$，此时$(R(\tau^n)-b)$就会有正有负，这样只有足够好的action才有机会得到概率上升，就不会出现所有但凡是遇到的action就全都上升的情况

##### how to calculate baseline

如何确定baseline $b$的值呢？

最简单的方式就是令$b$等于一局游戏中每次action得到的reward $R(\tau)$的期望$b=E(R(\tau))$

在实作时，由于$b$是一局游戏中所有reward的期望值，因此每次打完一局游戏，就需要对期望值更新

这个过程可以理解为，一局游戏中求出所有reward的平均值作为bias，再令所有rewards减去这个平均值，这样reward就肯定有正有负

#### Tip2: Assign Suitable Credit

在前面的式子中，对一场游戏的每一个state所采取的action，都乘上了相同的weight $(R(\tau^n)-b)$，这样做其实并不合理，即使整场游戏的结果是好的，这也并不代表游戏中的每个行为都是对的；整场游戏结果不好，也不代表游戏中的每个行为都是错的

下图中，第一场游戏和第二场游戏都是$s_b:a_2$、$s_c:a_3$，但前者得到了增幅后者却被削弱，因为$s_a:a_1$相对$s_a:a_2$得到了正的reward

<center><img src="../img/rl-tip2.png" width="60%"/></center>

我们希望给每个不同的action乘上不同的weight，来真正反映出这个action的好坏

理论上如果sample的次数足够多，也可以解决这个问题，但此时我们需要给不同的action合理的credit

由于在整场游戏中，某个action发生之前的事情是跟它没有关系的，因此**它所乘的reward可以设为该action发生之后的reward之和**，权重就修改成$\sum\limits_{t=t'}^{T_n}r_{t'}^n$

 这里要把握的一个基本精神是，时间拖得越长，action对后续reward的影响就越小

因此我们还会在权重前乘上一个小于1的折扣系数(discounted) $\gamma$，一般设为0.9或0.99，并且附上时间$t'-t$的指数，$\gamma^{t'-t}\cdot\tau^n_{t'}$代表越是后面时间的reward，受到当前action的影响就越小，这也被称为discounted reward

此时log_prob的系数就变成了(此时的bias应该修改为discounted reward的期望值)：

$$
R(\tau^n)-b=\sum\limits_{t=t'}^{T_n} \gamma^{t'-t}\cdot\tau^n_{t'} -b\\
b=E(\sum\limits_{t=t'}^{T_n} \gamma^{t'-t}\cdot\tau^n_{t'})
$$
上式就被成为Advantage Function：$A^{\theta}(s_t,a_t)$，它代表在某个state $s$采取action $a$的时候，advantage有多大，形象地说就是，在state $s$下采取的action $a$相较于其他action有多好(相对的好)

实际上$A^{\theta}(s_t,a_t)$也就是Critic的概念

<center><img src="../img/rl-tip3.png" width="60%"/></center>