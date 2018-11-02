To implement the Huffman Coding text compression algorithm in Python.

Data: The Project Gutenberg version of Melville's novel mobydick.txt is supplied as a data file for development and testing.

## Compression 
```
python huff-compress.py -I mobydick.txt -s char (or word)
```

## Decompression
```
python huff-decompress.py -I workfile -M infile_symbol_model.pkl
```


## Experiment Results
<img src="https://github.com/kenyonke/NLP/blob/master/Huffman%20Tree/result.png" width="600" height="400">

The performances of two different symbols are shown in table1. It is easy to find out that the char is little faster than word in building model and its model is much smaller than word-base model because it has much less symbols. However, word compression can compress file with less bytes and quicker in decompression. Also, it plays quicker when given the symbol model in compression. So, we can know that Huffman is effective when used with a word-based model rather than char-based model.
