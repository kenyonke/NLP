# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 17:16:15 2019

@author: kenyon
"""

class data:
    def __init__(self):
        #load data 
        # English source data
        with open("data/small_vocab_en", "r", encoding="utf-8") as f:
            self.source_text = f.read().split('\n')
        # French target data
        with open("data/small_vocab_fr", "r", encoding="utf-8") as f:
            self.target_text = f.read().split('\n')
        
        self.eng_maxlen,self.fr_maxlen,self.eng_word2id,self.eng_id2word,self.fr_word2id,self.fr_id2word = self.get_data()

    
    def get_data(self):        
        '''
        max length of text  && dictionary constraction 
        '''
        eng_word2id = dict()
        eng_id2word = dict()
        eng_maxlen = 0
        eng_id = 0
        for sen in self.source_text:
            sen_split = sen.lower().split()
            eng_maxlen = max(eng_maxlen,len(sen_split))
            for word in sen_split:
                if word not in eng_word2id:
                    eng_word2id[word] = eng_id
                    eng_id2word[eng_id] = word
                    eng_id += 1
        #eng special notes added
        SOURCE_CODES = ['<PAD>', '<UNK>']
        for code in SOURCE_CODES:
            eng_word2id[code] = eng_id
            eng_id2word[eng_id] = code
            eng_id += 1    
        
        fr_word2id = dict()
        fr_id2word = dict()
        fr_id = 0    
        fr_maxlen = 0
        for sen in self.target_text:
            sen_split = sen.lower().split()
            fr_maxlen = max(fr_maxlen,len(sen_split))
            for word in sen_split:
                if word not in fr_word2id:
                    fr_word2id[word] = fr_id
                    fr_id2word[fr_id] = word
                    fr_id += 1
        #fr special notes added
        TARGET_CODES = ['<PAD>', '<EOS>', '<UNK>', '<GO>']   
        for code in TARGET_CODES:
            fr_word2id[code] = fr_id
            fr_id2word[fr_id] = code
            fr_id += 1    
            
        return eng_maxlen, fr_maxlen, eng_word2id, eng_id2word, fr_word2id, fr_id2word

    def seq2id(self, sentence, max_length, is_Eng=True):
        ids = []
        #English
        if is_Eng:
            pad_id = self.eng_word2id['<PAD>']
            for word in sentence.lower().split():
                ids.append(self.eng_word2id.get(word, self.eng_word2id['<UNK>']))
        #French:
        else:
            pad_id = self.fr_word2id['<PAD>']
            for word in sentence.lower().split():
                ids.append(self.fr_word2id.get(word, self.fr_word2id['<UNK>']))
            ids.append(self.fr_word2id['<EOS>']) #add <EOS> at the end of target outputs    
        #pading
        if len(ids)>max_length:
            ids = ids[:max_length]
        else:
            ids = ids + [pad_id for _ in range(max_length-len(ids))]
        return ids
    
    def get_batches(self, batch_size, source_max_len=20, target_max_len=25):
        for i in range(len(self.source_text)//batch_size):
            #lengths of every sequence in sources and targets(inputs in decoder of tensorflow)
            source_lengths = []
            target_lengths = []
            # convert sources and targets into int 
            source_batch = []
            target_batch = []
            
            for source in self.source_text[i*batch_size:(i+1)*batch_size]:
                en_seq2id = self.seq2id(source,max_length=source_max_len,is_Eng=True)
                source_lengths.append(len(en_seq2id))
                source_batch.append(en_seq2id)
                
            for target in self.target_text[i*batch_size:(i+1)*batch_size]:
                fr_seq2id = self.seq2id(target,max_length=target_max_len,is_Eng=False)
                target_lengths.append(len(fr_seq2id))
                target_batch.append(fr_seq2id)

            yield source_batch, source_lengths, target_batch, target_lengths
    
    def id2seq(self, ids):
        return [self.fr_id2word[id_num] for id_num in ids]
        
    
if __name__ == '__main__':
    data = data()
    #print(data.eng_maxlen,'-',data.fr_maxlen)
    #print(len(data.eng_word2id.keys()))
    #print(len(data.fr_word2id.keys()))
    for source_batch, source_lengths, target_batch, target_lengths in data.get_batches(100):
        print(source_batch)
        print(target_batch)
        print(source_lengths)
        print(target_lengths)
        break