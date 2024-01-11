# Derain Methods Based on Uformer
## Project for Statistical Learning Course
[Ruo-Chen Cui](https://github.com/421zuoduan)

> **Abstract:** *Our project is based on Restormer, (write later)* 
<hr />


## Related Work

Restormer : [Paper](https://www.ijcai.org/proceedings/2022/0205.pdf) | [Code](https://github.com/swz30/Restormer)

SENet :  [Paper](https://arxiv.org/pdf/1709.01507.pdf) | [Code](https://github.com/hujie-frank/SENet)

Uformer : [Paper](https://www.ijcai.org/proceedings/2022/0205.pdf) | [Code](https://github.com/ZhendongWang6/Uformer)


Our training framework : [Code](https://github.com/XiaoXiao-Woo/derain)


## Our Work

我们主要参考了以下网络：
* Restormer
* Uformer
* SENet

在以上网络的基础上，我们对网络做出了一定的改进，并提出了一些自己的创新方法：

* Restormer_SE：加入了channel attention和SE module，其中channel attention和MDTA并行形成Hybrid Attention

* Restormer_KA：加入了利用Dynamic Conv的Kernel Attention，有NAN问题

* Restormer_KAv2：加入了利用普通卷积的Kernel Attention，没有nan问题；Kernel Attention分窗大小变化（16->8->4）

* Restormer_KAv3：加入了利用普通卷积的Kernel Attention，分窗大小不变（16）

* Uformer_KAv1：加入了利用普通卷积的Kernel Attention，在feature map大小为64和32时（输入大小为128）与window_attention形成两个分支，分窗大小不变均为16；而且仅在未进行shift window attention时进行kernel attention

* Uformer_KAv2：基于v1，更新了Kernel Attention的结构
* Uformer_KAv3：
* Uformer_KAv4：串行结构代码，窗口卷积+global kernel
* Uformer_KAv5：开始使用并行结构代码
* Uformer_KAv6：BasicUformerLayer层数增加，只有LeWinTransformerBlock去除了shift window，LeWinTransformerBlock没有去除了shift window
* Uformer_KAv7：只使用global kernel
* Uformer_KAv8：BasicUformerLayer层数没有增加，LeWinTransformerBlock_KA和LeWinTransformerBlock都去除了shift window
* Uformer_KAv9：基于KAv7，去除了shift window
* Uformer_KAv10：基于KAv4，去除了SA和SE
* Uformer_KAv11：基于KAv4，去除了global kernel的se
* Uformer_KAv12：基于KAv4，去除了移位
* Uformer_KAv13：基于KAv4，普通卷积改为深度可分离卷积

**Note**：Restormerv2和Restormerv3的代码应该有问题，在ConvLayer部分权重的维度变换有问题





## Results and Analysis

### 统计学习

我们设置学习率为1e-4，使用AdamW优化器，以L1Loss和SSIMloss作为损失函数。


以下为1000epoch训练下，各个模型的效果对比：

<table>
    <tr>
        <th align="center">模型</th>
        <th align="center">PSNR</th>
        <th align="center">SSIM</th>
        <th align="center">Params</th>
        <th align="center">FLOPs</th>
    </tr>
    <tr>
        <td align="left">Restormer</td>
        <td align="center">38.74</td>
        <td align="center">0.97994</td>
    </tr>
    <tr>
        <td align="left">Restormer_SE</td>
        <td align="center">38.41</td>
        <td align="center">0.97752</td>
    </tr>
    <tr>
        <td align="left">Restormer_KAv2</td>
        <td align="center">-</td>
        <td align="center">-</td>
    </tr>
    <tr>
        <td align="left">Uformer(bs=8)</td>
        <td align="center">38.49</td>
        <td align="center">0.97810</td>
        <td align="center">20.628M</td>
        <td align="center">10.308G</td>
    </tr>
    <tr>
        <td align="left">Uformer</td>
        <td align="center">38.76</td>
        <td align="center">0.97927</td>
        <td align="center">20.628M</td>
        <td align="center">10.308G</td>
    </tr>
    <tr>
        <td align="left">Uformer_KAv1</td>
        <td align="center">38.63</td>
        <td align="center">0.97798</td>
        <td align="center">21.953M</td>
        <td align="center">10.886G</td>
    </tr>
    <tr>
        <td align="left">Uformer_KAv2</td>
        <td align="center">38.78</td>
        <td align="center">0.97920</td>
        <td align="center">24.196M</td>
        <td align="center">12.704G</td>
    </tr>
    <tr>
        <td align="left">Uformer_KAv3</td>
        <td align="center">38.78</td>
        <td align="center">0.97940</td>
        <td align="center">24.196M</td>
        <td align="center">12.704G</td>
    </tr>
    <tr>
        <td align="left">Uformer_KAv4</td>
        <td align="center">38.82</td>
        <td align="center">0.97947</td>
        <td align="center">24.667M</td>
        <td align="center">13.548G</td>
    </tr>
</table>

除此之外，我们还测试了一些模型在不同epoch训练下的效果：


<table>
    <tr>
        <th>模型</th>
        <th>epoch</th>
        <th>PSNR</th>
        <th>SSIM</th>
        <th>模型</th>
        <th>epoch</th>
        <th>PSNR</th>
        <th>SSIM</th>
    </tr>
    <tr>
        <td rowspan="4">Restormer</td>
        <td>500</td>
        <td>37.30</td>
        <td>0.97369</td>
        <td rowspan="4">Restormer_SE</td>
        <td>500</td>
        <td>37.46</td>
        <td>0.97369</td>
    </tr>
    <tr>
        <td>1000</td>
        <td>38.76</td>
        <td>0.97927</td>
        <td>1000</td>
        <td>38.41</td>
        <td>0.97752</td>
    </tr>
    <tr>
        <td>2000</td>
        <td>39.47</td>
        <td>0.98284</td>
        <td>2000</td>
        <td>39.35</td>
        <td>0.98114</td>
    </tr>
    <tr>
        <td>3000</td>
        <td>39.79</td>
        <td>0.98382</td>
        <td>3000</td>
        <td>39.64</td>
        <td>0.98200</td>
    </tr>
    <!-- 按照需要添加更多行 -->
</table>




<table>
    <tr>
        <th>模型</th>
        <th>epoch</th>
        <th>PSNR</th>
        <th>SSIM</th>
        <th>模型</th>
        <th>epoch</th>
        <th>PSNR</th>
        <th>SSIM</th>
    </tr>
    <tr>
        <td rowspan="4">Uformer</td>
        <td>1000</td>
        <td>37.30</td>
        <td>0.97369</td>
        <td rowspan="4">Uformer_KAv4</td>
        <td>1000</td>
        <td>38.82</td>	
        <td>0.97947</td>
    </tr>
    <tr>
        <td>2000</td>
        <td>39.60</td>
        <td>0.98264</td>
        <td>2000</td>
        <td>39.36</td>
        <td>0.98219</td>
    </tr>
    <!-- 按照需要添加更多行 -->
</table>



更多的，我们在Rain100H上取200对图片作为训练集，经过训练，测试结果如下：

<table>
    <tr>
        <th>模型</th>
        <th>epoch</th>
        <th>PSNR</th>
        <th>SSIM</th>
        <th>模型</th>
        <th>epoch</th>
        <th>PSNR</th>
        <th>SSIM</th>
    </tr>
    <tr>
        <td rowspan="4">Uformer</td>
        <td>1000</td>
        <td>35.06</td>
        <td>0.95970</td>
        <td rowspan="4">Uformer_KAv4</td>
        <td>1000</td>
        <td>29.62</td>
        <td>0.86394</td>
    </tr>
</table>


### tiny paper

1000epoch情况下：

|模型|PSNR|SSIM|Params|FLOPs|
|-|-|-|-|-|
|Uformer bs=6|38.67999|0.97879|20.628M|10.308G|
|Uformer bs=12|37.87080|0.97472|20.628M|10.308G|
|Uformer_tinypaperv1|38.00910|0.97563|||
|Uformer_tinypaperv2|38.29627|0.97658|||
|Uformer_KAv4|38.81872|0.97947|24.667M|13.548G|
|Uformer_KAv5|38.72450|0.97846|24.854M|17.238G|
|Uformer_KAv7|38.71809|0.97864|||
|Uformer_KAv8|38.78743|0.97921|||
|Uformer_KAv9|38.81705|0.97939|||
|Uformer_KAv10|38.78690|0.97930|||
|Uformer_KAv11|38.77819|0.97959|||
|Uformer_KAv12|38.67999|0.97879|||

3
<!-- |Uformer_tinypaperv1|38.67999|0.97879|||
|Uformer_tinypaperv2|38.67999|0.97879||| -->

 ssim: , psnr: 
2000epoch情况下：

|模型|PSNR|SSIM|Params|FLOPs|
|-|-|-|-|-|
|Uformer|39.60271|0.98264|20.628M|10.308G|
<!-- |Uformer_tinypaperv1|39.60748|0.98275|||
|Uformer_tinypaperv2|39.60748|0.98275||| -->


3000epoch情况下：

|模型|PSNR|SSIM|Params|FLOPs|
|-|-|-|-|-|
|Uformer|39.92405|0.98404|20.628M|10.308G|
<!-- |Uformer_tinypaperv1|39.84044|0.98386||| -->

## Training and Evaluation

Training and Testing for Deraining:

<table>
  <tr>
    <th align="left">Derain</th>
    <th align="center">Dataset</th>
    <th align="center">Visual Results</th>
  </tr>
  <tr>
    <td align="left">Rain100L</td>
    <td align="center"><a href="https://www.icst.pku.edu.cn/struct/Projects/joint_rain_removal.html">Link</a></td>
    <td align="center">Empty</td>
  </tr>
  <tr>
    <td align="left">Rain100H</td>
    <td align="center"><a href="https://www.icst.pku.edu.cn/struct/Projects/joint_rain_removal.html">Link</a></td>
    <td align="center">Empty</td>
  </tr>
</table>

**Note**
* Download datasets and put it with the following format. 

* We introduce more methods based on Restormer to the single image deraing task.

* The project is based on MMCV, but you needn't to install it and master MMCV. More importantly, it can be more easy to introduce more methods.

* We have modified part of the code for reading data set, specifally for Rain100L, which can be read as the same way with Rain200L. But you have to operate on the dataset. Target and rainy figure should be operated into one figure. You can finish the operation above through dataset_rename.py, *but be careful to use them*.


```
|-$ROOT/datasets
├── Rain100L
│   ├── train_c
│   │   ├── norain-001.png
│   │   ├── ...
│   ├── test_c
│   │   │   ├── norain-001.png
│   │   │   ├── ...
```
