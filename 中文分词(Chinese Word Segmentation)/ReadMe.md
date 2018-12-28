中文分词方法
====

英文分词与中文分词中，最大的一个区别就是分词。英文中单词之间自带空格分词但中文的单词是直接连在一起的，所以中文NLP比英文NLP多了一个分词任务。

但是英文中，同一个词的字母大小写，形态和时态有所不同，之前做project的时候用Spacy可以高效处理这个问题。


从网上以及书上的资料，我理解的分词方法大概为3类：

    1.传统的字典匹配方法（最大长度匹配）
    2.统计方法（HMM, MEMM, CRF）
    3.深度学习方法
    
## 字典匹配
Trie三叉树+最大分词匹配

    1.正向最大匹配
    2.逆向最大匹配（通常效果比正向好）
    3.双向最大匹配（若正反结果不一样，选分词数目比较小的结果）
    
## 统计方法
#### 基于HMM
Generative有向概率图模型，利用联合概率以及Bayes推断词的tags

#### 基于MEMM
Discriminative局域归一化的有向概率图模型

#### 基于CRF
Discriminative全局归一化的无向概率图模型

## 深度学习方法
BiLSTM+CRF : https://github.com/Determined22/zh-NER-TF

BiLSTM+CRT论文Neural Architectures for Named Entity Recognition本是用于NER，所以用来做中文NER，延申出来就可以做基于逐字分词的分词方法
