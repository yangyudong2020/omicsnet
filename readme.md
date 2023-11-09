这个是最后一版的代码，两周前的。

然后nnunetv2/configuration.py 是各种配置参数
![image](https://github.com/yangshurong/nnUNet_cluster/assets/73787862/f4acf386-4dd8-461e-9b96-b8883afbef67)

在nnunetv2/run/run_training.py里面，选择使用clip还是不用clip（这个很脑残的操作新代码里修了，但我没保存）
代码的十二行和十三行负责选择
![image](https://github.com/yangshurong/nnUNet_cluster/assets/73787862/3afc93d9-8ba4-4d1e-b2a4-6d1fba8a71df)

在nnunetv2/training/nnUNetTrainer/synapse目录下，包含了clip所需要的代码

在nnunetv2/training/nnUNetTrainer/nnUNetTrainerUNETR.py和nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py里面，包含了nnUNet的核心trainer。他的大体思路是自己写了一个多进程加载训练数据的，这个数据读取器是无线读取的，比如从id为1的样本读取到100，然后下一次读取又会读取id为1的样本。nnunet数据设置里面可以设置batchsize。然后每个epoch是对着训练集，先跑固定次数的iterations，然后如果当前epoch达到可以start_num_valid的要求时，并且满足相应间隔时进行测试。测试是读取完成图片，然后裁剪成4个或者几个patchsize（模型输入的图片要求尺寸），然后跑模型预测，最后把得到的4个或者几个结果叠加求平均。

组学的思路是在完成预测后，先把所有病灶都看成一个类，然后用opencv划分成每个病灶，对每个病灶先形态学再纹理学，最后聚类分类得到相应的类别。组学使用多进程进行计算，内存优化还行，能跑。

组学计算思路：
代码在cluster_work下，需要先用me_cluster_small_save.py计算，并保存每个样本的组学特征。然后用me_cluster_small.py实现聚类算法。这样可以防止多次计算组学，还能保存组学便于clip时使用。

公开数据集上，需要重新转换数据格式（之前转好了代码丢了）。然后重新计算组学特征并保存，以及单独计算聚类。
目前的思路是全部三维，因此：
先用SimpleITK，结合mask，对3Dmask的区域进行划分（对应到二维上就是用opencv进行划分），然后根据三维mask提取该病灶，并拿只包含该病灶的3D数据和3Dmask去计算组学形态学和纹理学特征，最后为每个样本保存一个组学特征。推理的时候，最后组学聚类投票的时候，也是在推理代码里面修改，在最后得到三维结果后，对每个病灶计算组学并分类，然后进行投票。

运行所有代码的方式，与nnUNet相同，安装方法也相同。这是nnUNetv2，可以参考nnUNet最新的文档
