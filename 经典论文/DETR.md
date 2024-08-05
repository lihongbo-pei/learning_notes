# End-to-End Object Detection with Transformers

前向流程：

- 使用CNN网络（**resnet50**）提取图片特征
- 全局建模：图片特征拉成一维，输入Transformer Encoder 中进行全局建模，进一步通过自注意力学习全局特征。
  之所以使用Transformer Encoder，是因为Transformer 中的自注意力机制，使得图片中的每个点（特征）都能和图片中所有其他特征做交互了，这样模型就能大致知道哪块区域是一个物体，哪块区域又是另一个物体，从而能够尽量保证每个物体只出一个预测框。所以说这种全局特征非常有利于移除冗余的框。
- 通过Transformer Decoder 生成N个预测框set of box prediction（默认取N=100，也就是一张图固定生成100个预测框）。
- 计算二分图匹配损失（bipartite matching loss），选出最优预测框，然后计算最优框的损失。
  计算N个预测框与所有GT box（真实框）的matching loss，然后通过**二分图匹配算法**来选出与每个物体最匹配的预测框。比如上图中有两个物体，那么最后只有两个框和它们是最匹配的，归为前景；剩下98个都被标记为背景（no object）。最后和之前的目标检测算法一样，计算这两个框的分类损失和回归损失。

![1722843987880](assets/1722843987880.png)

### 3.2 DETR architecture

下面参考官网的一个demo，以输入尺寸3×800×1066为例进行前向过程：

- CNN提取特征（[800,1066,3]→[25,34,256]）
  backbone为ResNet-50，最后一个stage输出特征图为25×34×2048（32倍下采样），然后用1×1的卷积将通道数降为256；
- Transformer encoder 计算自注意力（[25,34,256]→[850,256]）
  将上一步的特征拉直为850×256，并加上同样维度的位置编码（Transformer本身没有位置信息），然后输入的Transformer encoder进行自注意力计算，最终输出维度还是850×256；
- Transformer decoder解码，生成预测框
    decoder输入除了encoder部分最终输出的图像特征，还有前面提到的`learned object query`，其维度为100×256。在解码时，learned object query和全局图像特征不停地做across attention，最终输出100×256的自注意力结果。
    这里的object query即相当于之前的anchor/proposal，是一个硬性条件，告诉模型最后只得到100个输出。然后用这100个输出接FFN得到分类损失和回归损失。
- 使用检测头输出预测框
  检测头就是目标检测中常用的全连接层（FFN），输出100个预测框（$ x_{center},y_{center}$,w,h）和对应的类别。
  使用二分图匹配方式输出最终的预测框，然后计算预测框和真实框的损失，梯度回传，更新网络。

![1722854828620](assets/1722854828620.png)



## 附录



![1722843392293](assets/1722843392293.png)