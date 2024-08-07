# ViTime A Visual Intelligence-Based Foundation Model for Time Series Forecasting

代码地址：https://github.com/IkeYang/ViTime

前置知识：MobileNetV2、ViT、MAE、ASPP

## 摘要

大型预训练模型在自然语言处理( NLP )和计算机视觉( CV )领域的成功为构建时间序列预测( TSF )基础模型开辟了新的途径。传统的TSF地基模型严重依赖于数值数据拟合。相比之下，人脑天生擅长处理视觉信息，更喜欢通过观察可视化的序列来预测未来的趋势。从仿生的角度来看，利用模型直接处理数值序列可能并不是实现通用人工智能( Artificial General Intelligence，AGI )最有效的途径。本文提出了一种新的基于视觉智能的TSF**基础模型**ViTime。ViTime利用可视化数据处理范式，克服了数值时间序列数据拟合的局限性。在一组以前看不到的预测数据集上的实验表明，ViTime取得了最先进的**零样本**性能，甚至在某些情况下超过了最好的单独训练的监督模型。这些发现表明，视觉智能可以显著增强时间序列分析和预测，为该领域更先进和通用的模型铺平道路。

## 引言

尽管取得了这些进展，但现有的基础时间序列模型仍然面临两个重大挑战。

1. 数值建模的局限性：首先，与大多数TSF模型类似，现有的基础模型都是通过直接拟合数值时间序列数据来训练的，这表明这些模型的主要信息载体是时间维度内的数值关系。相比之下，人类倾向于通过视觉表征来观察趋势，而不是直接处理数值。研究表明，人脑对视觉信息的处理效率高于数值型数据。Pettersson 发现人脑比数值型数据更擅长处理视觉信息。类似地，Dondis 证明了视觉皮层快速识别模式、形状和颜色，使得图像和视频的处理速度快于文本和数字。这些发现自然地导致了一个假设问题：在通向AGI的道路上，使用视觉智能进行时间序列建模可能比传统的数值方法更有效?

2. 训练数据的局限性：其次，基础模型的训练数据通常由大规模的真实世界数据集组成，这就提出了一个关键的问题：大规模的真实世界数据集能否全面地捕获通用时间序列模式的多样化范围?具体来说，一个模型应该具备哪些基础能力来解决一个通用的时间序列问题?

为了应对这些挑战，本文提出了一种新的基于视觉智能的时间序列基础模型——视觉时间基础模型( Visual Time Foundation Model，ViTime )，旨在从视觉智能的角度开拓时间序列基础模型研究的新范式。

- 此外，我们还提出了一种新的时间序列数据生成方法- - Real Time Series ( RealTS )，它将时间序列分析的基础知识分为"趋势性"和"周期性"，并在ViTime的训练过程中合成训练数据。
- ViTime通过将数值时间序列转换为**二值图像**，将数值时间相关性转换为二进制像素空间相关性。这种方法与大脑处理时间序列数据的熟练程度相一致。
- 实验结果表明，当应用于不同领域和分辨率的各种未知数据集时，所提出的ViTime获得了最先进的零样本结果，并且在某些情况下，超过了最佳的单独训练的监督模型。此外，仅用10%的领域数据微调，ViTime就可以获得比使用100%域数据的最新最先进的监督模型更优越的性能。

## 相关工作

### 2.1 时间序列预测

- 论文回顾了传统的时间序列预测方法，如 ARIMA 和其扩展模型，以及近年来深度学习在 TSF 中的进展，包括 RNN、LSTM、GRU 和基于 Transformer 的模型。

近年来，深度学习在时间序列预测（TSF）领域取得了显著进展。循环神经网络（RNN）及其变体，如长短期记忆网络（LSTM）和门控循环单元（GRU），在建模复杂的时间依赖关系方面展示了卓越的能力。最近，基于Transformer的模型因其能够有效地建模长距离依赖关系而受到关注。像Informer、Autoformer和PatchTST等模型利用自注意力机制，在TSF中设定了新的性能基准。

## 方法

### 3.2 总体架构

periodic pattern 周期模式

![1721977694623](assets/1721977694623.png)

​       图1描述了所提ViTime框架的总体架构，由4个关键模块、实时序列合成模块、映射函数、所提ViTime模型和逆映射函数组成。本文首先介绍了一种新颖的时间序列数据生成算法Real Time Series ( Real TS )，该算法将时间序列基础模型所需的知识分为**周期模式和趋势模式**两种，并据此合成数值时间序列数据。然后通过**映射函数**将生成的数值型时间序列数据映射为二值图像，将时间序列数据的数值关系转换为空间分布关系。

### 3.3 真实时间序列（RealTS）合成

​        一个稳健的TSF基础模型应该集成两种基本的时间序列变化知识：周期模式和趋势模式，它们包含了时间序列数据中的固有模式和方向变化。然而，真实世界的数据集通常缺乏对这些周期性和基于趋势的波动的完整频谱的表示，限制了模型在不同场景下的泛化能力和有效地学习潜在的动态。

在**周期假设$φ_p​$ **下，我们采用两种截然不同的数据行为模式：

- **快速傅里叶逆变换行为( IFFTB )**：为了确保合成的数据充分反映真实世界时间序列的变化模式，我们利用( 4 )中的快速傅里叶逆变换( IFFT )来模拟真实世界周期时间序列的基本行为：

  ![1721982466992](assets/1721982466992.png)

- **周期波行为( Periodic Wave Behavior，PWB )**：该行为通过叠加多个周期波来产生数据。将数据建模为正弦、余弦和其他周期函数，$f_{periodic }$，具有不同的频率和振幅：

  ![1721982833159](assets/1721982833159.png)

![1722934834540](assets/1722934834540.png)

在**趋势数据假设$φ_t$**下采用了三种不同的数据行为模式：

- **随机游走行为( Random Walk Behavior，RWB )**：RWB将数据建模为一个随机过程，其中每个值都是前一个值加上一个随机步长：

- **Logistic增长行为( LGB )**：LGB对数据进行Logistic增长函数建模，捕捉S型增长模式：

- **趋势波动数据行为( Trend Wave Data Behavior，TWDB )**：TWDB结合了线性趋势和周期性波动：

![1722934871513](assets/1722934871513.png)

在合成过程中，我们使用了各种数据增强技术来增强合成数据的多样性和鲁棒性，

- 多周期复制，它将生成的周期数据在多个周期上重复，以捕获长期的周期模式；
- 数据翻转；
- 卷积平滑和去趋势，从数据中去除潜在的趋势以分离出周期成分，使模型更容易学习这些模式；
- 数据扰动，即在数据中引入突变或异常，模拟现实世界的扰动，提高模型处理突发变化的能力等。

图2展示了RealTS产生的合成数据，展示了其产生具有各种周期模式的广泛时间序列的能力，使模型能够获得关于周期和趋势的广泛知识。预定义的先验/经验分布的详细设置见附录I。

### 3.4 基于二值图像的时间序列度量空间

空间定义函数、映射函数以及相关定理

Earth Mover’s Distance (EMD)，也称为Wasserstein距离

### 3.5 ViTime模型

![1721985064969](assets/1721985064969.png)

图3描述了所提出的ViTime模型的框架，它包括三个网络：视觉时间分词器、解码器和精炼模块。首先，对映射后的二值图像进行时间掩蔽，保证时间信息不被泄露。然后将被掩蔽的二值图像输入到视觉时间表征器中，并输出嵌入的表征。这些令牌随后被解码器解码，从而产生初始预测。最后，为了提高图像块交界处的生成质量，使用一个优化模块来输出最终的二值图像预测。

**Visual Time Tokenizer** 视觉时间标识符的主要作用是将被掩模的二值图像分割成多个块，整合位置编码，并将这些块映射到特征空间。该模块利用Vision Transformer( ViT ) 架构，捕捉patch之间的空间关系，从而将时间序列的时间依赖关系转化为像素值空间内的空间依赖关系。

**Decoder**  解码器将标记化的图像块转换回原始的二值像素度量空间，提供了一个初步预测，同时也采用了ViT架构。

**Refining Module** 解码器中的Transformer结构会导致 patch 连接处的不连续性，这可能会影响逆映射过程的准确性。为了解决这个问题，使用了带有卷积神经网络的精化模块构建。

- 首先，Decoder解码的tokens被还原为图像块，并被馈送到基于CNN的骨干网络中（MobileNetV2）。
- 然后，使用**融合空洞卷积金字塔( ASPP ) **模块来扩大模型的感受野。
- 最后，将输出结果**上采样**到原始的二值像素度量空间，生成最终的二值图像预测结果。

## 实验

### 4.1 实验配置

数据集：Electricity, Traffic, Weather, ETTh1, ETTh2 , ETTm1 and ETTm2

#### 4.1.3 评估指标

针对使用真实世界数据训练的时间序列预测（TSF）基础模型可能面临的**测试集泄漏问题**，为了解决这一问题并确保公平的实验比较，提出了两种零样本评估指标：Rescale-MAE（ReMAE）和Rescale-MSE（ReMSE）。ReMAE/ReMSE的主要概念包括以不同的时间分辨率重新缩放测试数据集。例如，长度为T的原始测试时间序列使用时间序列插值（TSI）方法重新缩放为βT，如公式(16)所示。

![1721991644712](assets/1721991644712.png)

ReMAE和ReMSE的计算公式如下所示：

![1721991694006](assets/1721991694006.png)

### 4.2 零样本评估

基线：TimesFM、PatchTST

表1报告了零样本实验结果。本文提出的ViTime - 1072模型在几乎所有的实验中都取得了最高的准确率。当使用与其他基线模型相同的输入序列长度时，ViTime在几乎所有的数据集和预测长度中始终排名第一，显著优于TimesFM。值得注意的是，ViTime在某些情况下的精度甚至超过了全监督PatchTST模型，显示了本文提出的ViTime的优越性。基于RealTS生成数据训练的ViTime和PatchTST - ZS的对比结果表明，从视觉智能的角度建模TSF任务可以显著提高模型的鲁棒性和准确性，这进一步验证了ViTime框架从视觉智能角度设计的合理性

![1721992072336](assets/1721992072336.png)

### 4.3 微调评估

为了进一步评估ViTime的性能，我们在本节中进行了一系列微调实验。我们对一些**基础模型**如TimesFM、GPT4TS和TIME-LLM使用10%的训练数据进行微调。此外，我们还考虑了近期的**SOTA（最先进的）监督时间序列预测模型**，包括SiMBA、TIMESNET和PatchTST，这些模型使用其论文中报告的100%训练数据。我们提出的ViTime分别使用10%和100%的训练数据进行微调，结果见表II。

![1721992333220](assets/1721992333220.png)

### 4.4 消融实验

（a）不同的分辨率

（b）不同的回溯窗口长度

（c）不同模型规模

![1722997918684](assets/1722997918684.png)

## 结论

​        在本文中，我们介绍了一种基于视觉智能的TSF基础模型ViTime，以及一种新颖的数据生成方法**RealTS**。我们的方法旨在通过利用视觉处理能力来解决传统数值数据拟合模型的固有局限性，这与人脑在处理视觉信息方面的优势更加吻合。提出的**ViTime框架将数值化的时间序列数据转化为二值图像**，使应用视觉智能技术分析和预测时间序列趋势成为可能。此外，提出的RealTS算法可以系统地生成多样化的合成时间序列数据，封装了重要的周期和趋势特征，并为ViTime模型的训练提供了足够的知识。

​       大量的实验评估表明，ViTime可以达到最先进的零样本性能。同时，微调实验表明，即使使用10 %的数据进行训练，所提出的ViTime也能优于最新的完全监督模型，证实了视觉智能在时间序列分析中的优势。我们相信，ViTime的提出可以从方法论的角度为AGI处理时间序列提供重要的见解。



## 代码

ViTime

```python
class ViTime(nn.Module):
    """
    A combined model using Masked Autoencoder (MAE) and DeepLab for image processing.
    """

    def __init__(self, args=None):
        super().__init__()
        MAE_Modelsize = copy.deepcopy(args.modelSize)
        args.modelAda = True
        self.args = args
        self.model = ViTimeAutoencoder(args=args
        )
        args.modelSize = 40
        self.RefiningModel = RefiningModel(
  
            downsample_factor=args.downsample_factor,
            dropout=args.dropout, args=args
        )
        self.EMD = nn.Softmax(dim=-1)
        args.modelSize = MAE_Modelsize
        self.dataTool=Dataset_ViTime(args)
        self.device=args.device
```

对于Refining Module，我们设置mobilenetv2作为骨干网络

```python
class RefiningModel(nn.Module):
    def __init__(self,  downsample_factor=16, dropout=0.1, args=None):
        super(RefiningModel, self).__init__()
        num_classes=1
        image_C=num_classes
        pretrained=False
        modelSize = args.modelSize if args else 1
        self.DO_ASPP = getattr(args, 'aspp', True)
        self.DO_LOWLEVEL = getattr(args, 'lowlevel', True)

        AdaFactor = args.modelSize / 20 if getattr(args, 'modelAda', False) else modelSize
        if AdaFactor <= (1 / 8):
            AdaFactor = 1 / 8
        dim_out = int(8 * 30 * AdaFactor)
        low_level_channels_out = int(8 * 30 * 1 / 5 * AdaFactor)

        self.backbone = MobileNetV2(AdaFactor=AdaFactor, downsample_factor=downsample_factor, pretrained=pretrained, image_C=image_C, dropout=dropout)
        in_channels = self.backbone.in_channels
        low_level_channels = self.backbone.low_level_channels


        self.aspp = ASPP(dim_in=in_channels, dim_out=dim_out, rate=16 // downsample_factor)
        self.upsampleASPP = DeconvASPP(dim_in=num_classes, dim_out=num_classes)

        self.shortcut_conv = nn.Sequential(
            nn.Conv2d(low_level_channels, low_level_channels_out, 1),
            nn.BatchNorm2d(low_level_channels_out),
            nn.ReLU(inplace=True)
        )
        if not self.DO_ASPP:
            dim_out = in_channels
        catin = low_level_channels_out + dim_out if self.DO_LOWLEVEL else dim_out
        self.cat_conv = nn.Sequential(
            nn.Conv2d(catin, dim_out, 3, padding=1),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(dim_out, dim_out, 3, padding=1),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.cls_conv = nn.Conv2d(dim_out, num_classes, 1)
```



推理结果：

![1722410101401](assets/1722410101401.png)