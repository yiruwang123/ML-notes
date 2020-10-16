# GAN

> GAN被Yann LeCun称为the coolest thing since sliced bread，这是有史以来最酷的东西，GAN和它的变形是近十年来ML领域最有趣的idea

#### Basic Idea of GAN

##### generation

GAN做的事情，就是generation

比如图像生成中，input一个随机的vector，生成器(generator)就会output一张image；比如语句生成中，input一个随机的vector，生成器就会output一个句子

<center><img src='https://gitee.com/Sakura-gh/ML-notes/raw/master/img/generation.png' width=60%></center>

当然，random的input是没有用的，我们希望的是有条件控制的生成(Conditional Generation)，比如输入一段文字，机器可以生成一张图片；输入一张图片，机器就会产生另外一张新的图片，关于如何控制生成的内容，将在后续的章节介绍，今天要focus的，还是如何用random的input来产生你想要的output这件事上

##### Generator

Generator是一个neural network，它的本质就是一个function，所谓的生成图片，本质上就是生成一个high dimension的vector，这个输出向量的每一维都对应pixel上的颜色，把这个high-dimension的向量排成一张图片的样子，就可以generate一张图片

作为输入的向量，它的每一个dimension实际上就对应着图片的某种特征，如果你改变input vector中某一个dimension的数值，反映到图像上的某个特征就会有所变化，比如头发的长度、头发的颜色

<center><img src='https://gitee.com/Sakura-gh/ML-notes/raw/master/img/generation1.png' width=60%></center>

##### Discriminator

Discriminator也是一个neural network，这个function它要吃Generator生成的东西(比如图片)当做input，它要输出的是一个scalar，这个数值表示Generator所生成图片的quality，也就是评价这张image看起来像不像真实的图片

如下图所示，Discriminator的output=1.0表示生成的图像很逼真，而output=0.1表示这离真实的图像差距还很大

<center><img src='https://gitee.com/Sakura-gh/ML-notes/raw/master/img/discriminator.png' width=60%></center>

##### Generator v.s. Discriminator

Generator和Discriminator之间的关系，就像是枯叶蝶和它的天敌鸟之间的关系，是物竞天择的基本思想

比如我们要生产二次元人物的头像，假设我们现在已经有了真实的二次元人物数据：

- 一开始Generator并不知道如何生成二次元图像，因此第一代的它只能输出杂讯一样的图片；而第一代的Discriminator的任务就是去判断这张图片像是Generator生成的还是真实的图片
- 第二代Generator要做的事情是想办法骗过第一代的Discriminator，假设第一代的Discriminator是根据图片有没有色彩来判断第一代的Generator生成的图片是否真实，那第二代的Generator就进化成能够生成有色彩的图片；但是Discriminator也会进化，第二代的它会学着去分辨第二代Generator生成的图片和已有的真实图片之间的差异，比如它用图片中有没有嘴巴这个feature来判断生成的图片是否真实
- 此时Generator为了骗过第二代的Discriminator，又会进化成第三代的Generator，Discriminator也会跟着进化...它们像是天敌**对抗**之间的关系，也就是adversarial这个名词的由来

<center><img src='https://gitee.com/Sakura-gh/ML-notes/raw/master/img/adversarial.png' width=60%></center>

当然，如果你不喜欢对抗这个词，那其实GAN也可以被理解为和平的关系，就像老师和学生之间的关系一样，学生从一年级->二年级->三年级...一直在不断地学习如何画二次元人物，同年级的老师是见过真正的二次元图像的，所以他会给出相应的意见；随着学生画的越来越好，老师也就越来越严格，最终学生就会学到很高的绘画技术

<center><img src='https://gitee.com/Sakura-gh/ML-notes/raw/master/img/adversarial1.png' width=60%></center>

这里会有两个问题：

- 为什么Generator不能直接从二次元人物的data里去学习如何生成图像呢？为什么一定要通过Discriminator(见过data)才能学习呢？
- Discriminator既然这么厉害，那他为什么不自己去做生成图像这件事呢？

一个很形象的比喻：Generator和Discriminator之间的关系，就像是写作敌人，念做朋友

就像是塔矢亮和进藤光之间的关系，就像是佐助与鸣人之间的关系一样

<center><img src='https://gitee.com/Sakura-gh/ML-notes/raw/master/img/adversarial2.png' width=60%></center>

#### Algorithm

##### basic idea of algorithm

在实际的算法过程中，首先要initialize Generator和Discriminator这两个neural network的参数

然后去iterative地训练这两个NN，那在每一个iteration里面都要做两个步骤：

- step1：把Generator的参数固定住，只去train Discriminator的参数

    首先从真实图像的database里sample一些图像出来，它们贴上1的标签，而Generator生成的图像就贴上0的标签，它们一起作为Discriminator的input；我们的训练目标是，如果image是真实的，那Discriminator的output就要接近于1，如果image是从Generator产生出来的，那output就要接近于0

    基本的训练方法就跟平常训练一个regression或是classification的neural network是一样的

    <center><img src='https://gitee.com/Sakura-gh/ML-notes/raw/master/img/algorithm1.png' width=60%></center>

- step2：把Discriminator的参数固定住，只去train Generator的参数

    基本训练目标是，把一个vector作为Generator的input，它output出来的图片，通过Discriminator会得到一个接近于1的高分，由于在step1里Discriminator只会给真实的图片高分，所以此时拿到高分的Generator，它生成的图片会是比较逼真的

    实际操作的时候，就是把Generator和Discriminator接起来形成一个large network，这个network的input是一个vector，它的output是一个数值，需要注意的是这个large network中间的某一个hidden layer会很宽，它的output就是一张image(本质上就是一个高维的向量)，也就是充当合并之前Generator的output层的角色，在训练的时候，就固定后面几个hidden layer，只调前面几个hidden layer，让它output的值越大越好

    <center><img src='https://gitee.com/Sakura-gh/ML-notes/raw/master/img/algorithm2.png' width=60%></center>

##### algorithm procedure

假设Discriminator的参数是$\theta_d$，Generator的参数是$\theta_g$，并且已经有一个二次元人物图像的database

###### 训练Discriminator

- 先从这个database里面sample出m笔example $\left \{ x^1,x^2,...,x^m \right \}$，这个m就像是train network时候的batch size

- 然后从一个distribution(可以是uniform distribution、gaussian distribution...)同样sample出m个vector，也就是$\left \{ z^1,z^2,...,z^m \right \}$这个vector的dimension是一个需要你去调的参数

- 接下来根据这m个vector，Generator产生m张image，即$\left \{ \tilde x^1,\tilde x^2,...,\tilde x^m \right \}$

- 然后去调整Discriminator的参数$\theta_d$，在GAN的原始paper中说明的update方式如下图所示(当然还有很多更好的其他方法)，它要去Maximize这个objective function，其中$D(x^i)$表示第i张真实的图片通过Discriminator得到的分数，而$D(\tilde x^i)$则表示第i张由Generator生成的图片通过Discriminator得到的分数

    $\tilde V$中的第一项$\frac{1}{m} \sum\limits_{i=1}^m log D(x^i)$表示m张真实图片得到的分数取log后的平均值，我们要让它越大越好，也就是让真实图像的得分$D( x^i)$普遍比较高

    $\tilde V$中的第二项$\frac{1}{m} \sum\limits_{i=1}^m log(1-D(\tilde x^i))$表示m张生成的假图片得到的分数被1减去之后再取log的平均值，我们要让它越大越好，也就是让生成的假图像的得分$D(\tilde x^i)$普遍比较低

    注：这里的$D(\tilde x^i)$会经过一个sigmoid function从而介于0-1之间，因此$log(1-D(\tilde x^i))$中的$1-D(\tilde x^i)$不会小于0；当然上述的方法是最原始paper的解法，并不是最优解法

- 通过gradient ascent的方式去update $\theta_d$，注意这里的目标是要让$\tilde V$越大越好，因此是gradient ascent，update的时候用的是“+”

###### 训练Generator

- 一样去sample出m个random的vector $\left \{ z^1,z^2,...,z^m \right \}$，它们可以与训练Discriminator过程中的$z^i$不一样
- $G(z^i)$表示vector $z^i$通过Generator生成的图片，再把这张图片丢到$D(G(z^i))$里面得到它的评价分数并取log，然后把这个batch里面的log(得分)取均值，我们希望它越大越好，也就是让Generator生成出来的image，通过Discriminator得到的评价分数普遍比较高
- 通过gradient ascent的方式去update $\theta_g$，让$\tilde V$越大越好

<center><img src='https://gitee.com/Sakura-gh/ML-notes/raw/master/img/algorithm3.png' width=60%></center>

#### GAN as Structured Learning

##### introduction of Structured Learning

我们都知道，regression的output是一个scalar，classification的output是一个class，而structured Learning的output则是一个结构化的对象，它可以是一个sequence、一个matrix、一个graph、一个tree...

GAN其实就是一种Structured Learning，可以是输入一张抽象或是黑白的image，输出一张真实上色的image，也可以是输入一段文字，输出一张真实的image；比如我们可以输出二次元人物的特征，比如头发的颜色、眼睛的颜色，然后让GAN输出二次元人物的头像

<center><img src='https://gitee.com/Sakura-gh/ML-notes/raw/master/img/structured.png' width=60%></center>

##### one-shot/zero-shot learning

事实上我们做分类器的时候，需要提供给机器不同类别的不同范例，而one-shot/zero-shot learning的意思是可能有些类别根本就没有任何范例或者是只有非常少的范例

而structured learning完全可以被看作是一个极端的one-shot/zero-shot learning的problem，比如structured learning output的东西是一个句子(比如翻译的task)，在整个training data里面可能就没有句子是重复的，而testing data又都是training data里没有出现过的东西，如果把各种不同的output都视为一个class的话，如何让structured learning学着去输出在training data里从来都没有看过的东西，也就是说机器要学会**创造**

##### the concept of planning

机器必须要有**规划**的概念，它要生成的复杂物件，是由很多简单的component组成的，比如生成图像的时候，机器的输出本质上也还是许多简单的pixel，但在输出之前，机器必须要在心里分辨清楚，这些pixel最终要能够拼凑成一张人脸，某些地方的pixel要组成眼睛，某些地方的pixel要组成眉毛...

所以在structured learning里面，真正重要的不是产生了什么component，而是component和component之间的关系，单看生成的一部分像素点或是一部分文字，是没有办法评断生成图像的好坏，也没有办法判断整个文本的语义的

<center><img src='https://gitee.com/Sakura-gh/ML-notes/raw/master/img/structured1.png' width=60%></center>

##### structured learning approach

GAN其实就是structured learning的一种解决方案，在传统的structured learning的文献里，有两套方法：

- Bottom Up：自底向上，机器是一个一个component分开去产生这个物件，缺点是很容易失去大局观
- Top Down：生成一个完整的物件以后，再去从整体看这个物件的好坏，缺点是很难做Generation

Generator可以视为是一个Bottom Up的方法，而Discriminator可以被视为是Top Down的方法，把这两个方法结合起来就是Generative Adversarial Network，就是GAN

<center><img src='https://gitee.com/Sakura-gh/ML-notes/raw/master/img/structured2.png' width=60%></center>

#### Can Generator learn by itself？

Bottom-UP：Generator从pixel的层次开始生成图像，但很难考虑component之间的关系

只要建立起code和image的对应关系，再用传统的supervised learning训练即可

这个关系可以由Auto-encoder来建立，编码器可以抽取出一张图像的特征，同时解码器又能够利用该特征还原回图像，在这里解码器Decoder扮演的角色就是Generator

因此Generator完全是可以自己独立学习和训练的

<center><img src='https://gitee.com/Sakura-gh/ML-notes/raw/master/img/GAN-auto.png' width=60%></center>

一般的Auto-encoder无法生成过渡状态的图像，借此提出了有考虑噪声的VAE(Variational Auto-encoder)，具体可参考Unsupervised Learning Generation一节的笔记

但是，Auto-encoder最大的问题在于，它只会比较pixel与pixel之间的相似度，却无法考虑component与component之间的关系，不同的组件之间无法相互影响和配合

#### Can Discriminator Generate？

Up-Bottom：Discriminator从整体的层次评价component之间的关系，但很难

Discriminator做生成的方式是，穷举所有可能的$x$，并取得分最高的作为生成结果

训练时应该给它好的图像并让它给高分，给它坏的图像并让它给低分，但我们只有真实的图像作为好的图像，如何去生成坏的图像数据呢？

<center><img src='https://gitee.com/Sakura-gh/ML-notes/raw/master/img/GAN-dis.png' width=60%></center>

首先拿真实图像作为好的图像，随机生成的噪声作为坏的图像，拿去给Discriminator识别训练，训练完成后让Discriminator生成它觉得好的图像，并拿这些图像替代原先噪声图像作为坏的图像，继续训练，重复上述步骤后，Discriminator生成的图像会越来越好

下图中，绿线表示用于训练的好的图像，蓝线表示用于训练的坏的图像(由判别器生成)，上方的红线是判别器的认知，在没有被绿线和蓝线覆盖的地方，红线是不确定的，因此上述过程可以被认为是不断地找判别器原本以为是好的而实际上是坏的点，使不断的训练过程中，红线逐渐符合真实的图像分布，当绿线和蓝线重合时，停止更新，此时判别器生成的图像与真实图像一致

<center><img src='https://gitee.com/Sakura-gh/ML-notes/raw/master/img/GAN-dis2.png' width=60%></center>

#### Generator Vs Discriminator

Generator：

- 优点：很容易做生成
- 缺点：不容易考虑不同component之间的关系，只能学到表象，但学不到大局

Discriminator：

- 优点：会考虑大局和不同component之间的关系
- 缺点：做生成是很困难的

两者是优势互补的，自底向上+自上而下