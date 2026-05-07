第三章 系统模型与问题描述
=============

3.1 系统场景与任务模型
-------------

考虑一个由单一边缘服务器（Edge Server, ES）和  $M$  个能量采集终端设备（User Equipment, UE）组成的移动边缘计算（MEC）系统，设备集合记为

$$
\mathcal{M}=\{1,2,\dots,M\}.
$$

系统运行在离散时间轴上，时隙索引为

$$
t=0,1,\dots,T-1,
$$

每个时隙的长度为  $\Delta$  秒。

在每个时隙  $t$  开始时，设备  $m$  产生一个计算任务，其输入数据量记为

$$
\lambda_m(t)\quad (\text{bits}).
$$

与传统 MEC 模型中常见的硬截止时间（Deadline）设定不同，本文不再假设任务必须在预设时限内完成。任务一旦产生，将持续在系统中排队并等待处理，直到被成功执行完成或因缓冲区容量限制而被丢弃。

考虑到终端设备和边缘服务器均具有有限的物理存储空间，系统中各级缓冲区均存在容量上限。当新到达任务在加入相应缓冲区后导致积压数据量超过其最大容量时，该任务将被判定为溢出并直接丢弃。

此外，终端设备依靠环境中的可再生能源（如太阳能、风能等）供电。设时隙  $t$  内设备  $m$  采集到的能量为

$$
H_m(t),
$$

并存储于容量有限的电池中，设备  $m$  的最大电池容量记为

$$
E_m^{\max}.
$$

* * *

3.2 决策变量
--------

在每个时隙  $t$ ，设备  $m$  需要为新到达任务制定卸载决策

$$
a_m(t)\in\{\mathrm{L},\mathrm{O}\},
$$

其中  $\mathrm{L}$  表示本地计算（Local Computing）， $\mathrm{O}$  表示任务卸载（Offloading）。

为进一步刻画资源消耗，系统还引入固定资源控制变量：

1.  **本地 CPU 频率**  
    当  $a_m(t)=\mathrm{L}$  时，任务进入本地计算队列，并由终端设备以固定 CPU 频率
    $$
    f_m
    $$
    进行处理。
2.  **发射功率**  
    当  $a_m(t)=\mathrm{O}$  时，任务进入上行传输队列，并以固定发射功率
    $$
    p_m
    $$
    发送至边缘服务器。

为简化表示，定义：

*   当  $a_m(t)\neq \mathrm{L}$  时，令  $f_m=0$ ；
*   当  $a_m(t)\neq \mathrm{O}$  时，令  $p_m=0$ 。

需要指出的是，当设备当前可用电池能量不足以支持相应计算或传输操作时，即使决策选择了本地执行或任务卸载，也无法真正消耗资源完成服务。此时可等效视为本地计算能力或传输能力暂时不可用，即对应的服务速率为零，任务继续滞留在队列中等待后续时隙处理。

* * *

3.3 通信与计算模型
-----------

### 3.3.1 上行传输模型

设设备  $m$  与边缘服务器之间在时隙  $t$  的上行信道增益为  $h_m(t)$ 。根据香农公式，其上行传输速率可表示为

$$
r_m(t)=B\log_2\left(1+\frac{p_m|h_m(t)|^2}{\sigma^2}\right),
$$

其中， $B$  为系统带宽， $\sigma^2$  为噪声功率。

若任务在时隙  $t$  被选择为卸载执行，则其纯传输所需时隙数定义为

$$
T_m^{\mathrm{tx}}(t)=\left\lceil \frac{\lambda_m(t)}{r_m(t)\Delta}\right\rceil.
$$

对应的传输能耗为

$$
e_m^{\mathrm{tx}}(t)=p_m\Delta.
$$

* * *

### 3.3.2 本地计算模型

设设备  $m$  处理 1 bit 数据所需的 CPU 周期数为  $\rho_m$ 。若任务在时隙  $t$  选择本地执行，则其本地处理所需时隙数为

$$
T_m^{\mathrm{loc}}(t)=\left\lceil \frac{\lambda_m(t)\rho_m}{f_m\Delta}\right\rceil.
$$

采用基于动态电压频率调整（DVFS）的经典功耗模型，本地计算能耗表示为

$$
e_m^{\mathrm{loc}}(t)=\kappa_m f_m^3\Delta,
$$

其中  $\kappa_m$  为设备  $m$  的有效电容系数。

* * *

### 3.3.3 边缘计算模型

边缘服务器维护一个统一的先入先出（FIFO）计算队列，用于处理所有终端设备卸载而来的任务。设边缘服务器采用恒定 CPU 频率

$$
f^{\mathrm{edge}}
$$

执行任务，且处理 1 bit 数据所需 CPU 周期数为  $\rho^{\mathrm{edge}}$ 。

则设备  $m$  在时隙  $t$  产生并成功卸载到边缘侧的任务，其边缘执行所需时隙数为

$$
T_m^{\mathrm{edge}}(t)=\left\lceil \frac{\lambda_m(t)\rho^{\mathrm{edge}}}{f^{\mathrm{edge}}\Delta}\right\rceil.
$$

* * *

3.4 能量动态与约束
-----------

设设备  $m$  在时隙  $t$  开始时的电池能量状态为  $E_m(t)$ 。则其在下一时隙的电池能量更新为

$$
E_m(t+1)=\min\left\{E_m^{\max},\,E_m(t)-e_m(t)+H_m(t)\right\},
$$

其中  $e_m(t)$  表示设备  $m$  在时隙  $t$  的总能耗。

由于每个时隙内设备至多选择一种服务方式，因此总能耗定义为

$$
e_m(t)= \begin{cases} e_m^{\mathrm{loc}}(t), & a_m(t)=\mathrm{L},\\[4pt] e_m^{\mathrm{tx}}(t), & a_m(t)=\mathrm{O},\\[4pt] 0, & \text{若因能量不足导致当时隙无法实际服务}. \end{cases}
$$

为保证系统满足能量中立运行（Energy Neutral Operation），需满足硬能量约束：

$$
e_m(t)\le E_m(t),\qquad \forall m,\forall t.
$$

* * *

3.5 混合队列模型：基于缓冲区准入与时间戳时延计算
--------------------------

为了同时刻画有限内存约束与任务完成时延，本文采用“**数据队列判断准入，时间戳计算时延**”的混合建模方式。

定义：

*    $Q_m^{\mathrm{loc}}(t)$ ：时隙  $t$  开始时设备  $m$  本地计算队列中的积压数据量（bits）；
*    $Q_m^{\mathrm{tx}}(t)$ ：时隙  $t$  开始时设备  $m$  上行传输队列中的积压数据量（bits）；
*    $Q^{\mathrm{edge}}(t)$ ：时隙  $t$  开始时边缘服务器总计算队列中的积压数据量（bits）。

相应的缓冲区容量上限分别记为

$$
Q_m^{\mathrm{loc},\max},\qquad Q_m^{\mathrm{tx},\max},\qquad Q^{\mathrm{edge},\max}.
$$

同时，引入资源忙闲截止时间戳：

*    $F_m^{\mathrm{loc}}(t)$ ：设备  $m$  本地 CPU 在时隙  $t$  开始时的忙闲截止时间；
*    $F_m^{\mathrm{tx}}(t)$ ：设备  $m$  上行链路在时隙  $t$  开始时的忙闲截止时间；
*    $F^{\mathrm{edge}}(t)$ ：边缘服务器在时隙  $t$  开始时的忙闲截止时间。

此外，定义任务因任一阶段缓冲区溢出而被丢弃时的统一惩罚时延为

$$
\Psi>0.
$$

* * *

### 3.5.1 本地执行路径

当  $a_m(t)=\mathrm{L}$  时，任务选择本地执行。

设设备  $m$  在单个时隙内的本地处理能力为

$$
D_m^{\mathrm{loc}}=\frac{f_m\Delta}{\rho_m}.
$$

首先，根据当前时隙开始时上一时隙已完成的数据量，对本地队列进行预更新：

$$
Q_m^{\mathrm{loc}\prime}(t)=\left[Q_m^{\mathrm{loc}}(t)-D_m^{\mathrm{loc}}\right]^+,
$$

其中  $[x]^+=\max\{x,0\}$ 。

#### （1）缓冲区准入判定

若

$$
Q_m^{\mathrm{loc}\prime}(t)+\lambda_m(t)>Q_m^{\mathrm{loc},\max},
$$

则新任务因本地缓冲区溢出被丢弃。此时有

$$
Q_m^{\mathrm{loc}}(t+1)=Q_m^{\mathrm{loc}\prime}(t),
$$
 
$$
F_m^{\mathrm{loc}}(t+1)=F_m^{\mathrm{loc}}(t),
$$

并定义该任务时延为

$$
D_m(t)=\Psi.
$$

若

$$
Q_m^{\mathrm{loc}\prime}(t)+\lambda_m(t)\le Q_m^{\mathrm{loc},\max},
$$

则任务被接纳进入本地队列，此时

$$
Q_m^{\mathrm{loc}}(t+1)=Q_m^{\mathrm{loc}\prime}(t)+\lambda_m(t).
$$

#### （2）基于时间戳的时延计算

对于被成功接纳的本地任务，其开始执行时刻定义为

$$
S_m^{\mathrm{loc}}(t)=\max\{t,F_m^{\mathrm{loc}}(t)\}.
$$

相应的完成时刻为

$$
C_m^{\mathrm{loc}}(t)=S_m^{\mathrm{loc}}(t)+T_m^{\mathrm{loc}}(t)-1.
$$

因此，本地 CPU 的忙闲截止时间更新为

$$
F_m^{\mathrm{loc}}(t+1)=C_m^{\mathrm{loc}}(t)+1.
$$

该任务的时延定义为

$$
D_m(t)=C_m^{\mathrm{loc}}(t)-t.
$$

* * *

### 3.5.2 卸载执行路径

当  $a_m(t)=\mathrm{O}$  时，任务沿“上行传输—边缘执行”路径处理。

#### （1）传输阶段：准入与传输时延

设设备  $m$  在时隙  $t$  的传输能力为

$$
D_m^{\mathrm{tx}}(t)=r_m(t)\Delta.
$$

则对传输队列进行预更新：

$$
Q_m^{\mathrm{tx}\prime}(t)=\left[Q_m^{\mathrm{tx}}(t)-D_m^{\mathrm{tx}}(t)\right]^+.
$$

若

$$
Q_m^{\mathrm{tx}\prime}(t)+\lambda_m(t)>Q_m^{\mathrm{tx},\max},
$$

则任务因传输缓冲区溢出被丢弃。此时有

$$
Q_m^{\mathrm{tx}}(t+1)=Q_m^{\mathrm{tx}\prime}(t),
$$
 
$$
F_m^{\mathrm{tx}}(t+1)=F_m^{\mathrm{tx}}(t),
$$

并定义

$$
D_m(t)=\Psi.
$$

若

$$
Q_m^{\mathrm{tx}\prime}(t)+\lambda_m(t)\le Q_m^{\mathrm{tx},\max},
$$

则任务被接纳进入传输队列，此时

$$
Q_m^{\mathrm{tx}}(t+1)=Q_m^{\mathrm{tx}\prime}(t)+\lambda_m(t).
$$

对于被成功接纳的卸载任务，其传输开始时刻为

$$
S_m^{\mathrm{tx}}(t)=\max\{t,F_m^{\mathrm{tx}}(t)\},
$$

传输完成时刻为

$$
C_m^{\mathrm{tx}}(t)=S_m^{\mathrm{tx}}(t)+T_m^{\mathrm{tx}}(t)-1.
$$

因此，上行链路忙闲截止时间更新为

$$
F_m^{\mathrm{tx}}(t+1)=C_m^{\mathrm{tx}}(t)+1.
$$

* * *

#### （2）边缘阶段：准入与边缘执行时延

仅当任务在传输阶段未被丢弃时，其才有资格进入边缘服务器缓冲区。

设边缘服务器单个时隙内可处理的数据量为

$$
D^{\mathrm{edge}}=\frac{f^{\mathrm{edge}}\Delta}{\rho^{\mathrm{edge}}}.
$$

则边缘队列预更新为

$$
Q^{\mathrm{edge}\prime}(t)=\left[Q^{\mathrm{edge}}(t)-D^{\mathrm{edge}}\right]^+.
$$

若

$$
Q^{\mathrm{edge}\prime}(t)+\lambda_m(t)>Q^{\mathrm{edge},\max},
$$

则任务到达边缘后因边缘缓冲区溢出被丢弃。此时有

$$
Q^{\mathrm{edge}}(t+1)=Q^{\mathrm{edge}\prime}(t),
$$
 
$$
F^{\mathrm{edge}}(t+1)=F^{\mathrm{edge}}(t),
$$

并定义

$$
D_m(t)=\Psi.
$$

若

$$
Q^{\mathrm{edge}\prime}(t)+\lambda_m(t)\le Q^{\mathrm{edge},\max},
$$

则任务被接纳进入边缘执行队列，此时

$$
Q^{\mathrm{edge}}(t+1)=Q^{\mathrm{edge}\prime}(t)+\lambda_m(t).
$$

对于被边缘侧成功接纳的任务，其边缘执行开始时刻应不早于传输完成之后，因此定义为

$$
S_m^{\mathrm{edge}}(t)=\max\{C_m^{\mathrm{tx}}(t)+1,\;F^{\mathrm{edge}}(t)\}.
$$

相应的边缘执行完成时刻为

$$
C_m^{\mathrm{edge}}(t)=S_m^{\mathrm{edge}}(t)+T_m^{\mathrm{edge}}(t)-1.
$$

于是，边缘服务器忙闲截止时间更新为

$$
F^{\mathrm{edge}}(t+1)=C_m^{\mathrm{edge}}(t)+1.
$$

对于成功完成边缘处理的卸载任务，其总时延定义为

$$
D_m(t)=C_m^{\mathrm{edge}}(t)-t.
$$

* * *

3.6 问题描述
--------

本文的目标是在长期运行过程中最小化所有设备任务的平均时延。由于本文不设置硬截止时间，任务一旦被成功接纳，将持续排队等待直至完成，因此拥塞情况下任务会自然产生较大的完成时延；而对于在任一阶段因缓冲区溢出而被丢弃的任务，则赋予固定惩罚时延  $\Psi$ 。由此，系统优化目标可以统一表示为长期平均时延最小化问题：

$$
\min_{\boldsymbol{\pi}} \quad \lim_{T\to\infty} \frac{1}{MT} \mathbb{E} \left[ \sum_{t=0}^{T-1}\sum_{m=1}^{M} D_m(t) \right],
$$

其中  $\boldsymbol{\pi}$  表示系统的联合决策策略。

该优化问题需满足以下约束：

$$
\text{C1:}\quad a_m(t)\in\{\mathrm{L},\mathrm{O}\},\qquad \forall m,\forall t,
$$
 
$$
\text{C2:}\quad \begin{aligned} &Q_m^{\mathrm{loc}\prime}(t)+\lambda_m(t)\le Q_m^{\mathrm{loc},\max} \quad \text{或任务被丢弃},\\ &Q_m^{\mathrm{tx}\prime}(t)+\lambda_m(t)\le Q_m^{\mathrm{tx},\max} \quad \text{或任务被丢弃},\\ &Q^{\mathrm{edge}\prime}(t)+\lambda_m(t)\le Q^{\mathrm{edge},\max} \quad \text{或任务被丢弃}, \end{aligned}
$$
 
$$
\text{C3:}\quad e_m(t)\le E_m(t),\qquad \forall m,\forall t.
$$

由于该问题中同时包含离散卸载决策、随机任务到达、随机信道状态、能量采集过程以及跨时隙耦合的队列/时间戳动态，难以通过传统凸优化方法直接求解，因此后续将其建模为马尔可夫决策过程，并采用深度强化学习方法进行求解。