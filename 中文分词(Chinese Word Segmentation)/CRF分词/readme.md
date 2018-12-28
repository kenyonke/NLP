# 利用CRF++实现中文分词

#### 训练样本
人民报`people-daily.txt`

#### 样本预处理
`convert.py`可以将训练样本转化为4tags（S/B/M/E）符合crf++需求的逐字形式

将70%作为训练样本`CHN-train` and 30%作为测试样本`CHN-test`

#### CRF++特征模板定义
template为crf++所需的设定features

关于CRF++的使用
```
crf_learn template_file train_file model_file
crf_test -m model_file test_files > result_file
```
详细可以查看crf++的官方文档

#### 查看结果
`result`为crf++输出的测试样本结果，最后一列为预测tags，倒数第二列为真实tags

`eval_cws.py`可以查看结果
