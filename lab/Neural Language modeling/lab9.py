# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class NGramLanguageModeler(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramLanguageModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)
        
    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs
    
    def predict_word(self,inputs,word_to_ix):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        probs = F.softmax(out, dim=1)
        index = probs.argmax()
        return NGramLanguageModeler.get_key(word_to_ix, index)
    
    def get_softmax(self,inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        probs = F.softmax(out, dim=1)
        return probs
                
    def get_key(dic, value):
        return [key for key,v in dic.items() if v == value]
    
    def predict_function(self,test_sentence,word_to_ix):
        changed_test_sentence = 'START_OF_SENTENCE ' + test_sentence + ' End_OF_SENTENCE '
        changed_test_sentence = changed_test_sentence.split()
        test_data = []
        for i in range(len(changed_test_sentence) - 2):
            context = [changed_test_sentence[i], changed_test_sentence[i+1]]
            test_data.append(context)
        for context in test_data:
            print('context:','[',context[0],",",context[1],"]")
            print('predicted_word:',self.predict_word(torch.tensor([word_to_ix[context[0]],word_to_ix[context[1]]], dtype=torch.long),word_to_ix))
            print('----')


#process data
train_sentences = ['The mathematician ran .',
                  'The philosopher thought about it .',
                  'The mathematician ran to the store .',
                  'The physicist ran to the store .',
                  'The mathematician solved the open problem .']
torch.manual_seed(1)
trigrams = []
splited_text = []
#add 'START_OF_SENTENCE ' and ' End_OF_SENTENCE ' in every sentence
for sentence in train_sentences:
    changed_sentence = 'START_OF_SENTENCE ' + sentence + ' End_OF_SENTENCE '
    changed_sentence = changed_sentence.split()
    splited_text.extend(changed_sentence)
    for i in range(len(changed_sentence) - 2):
        context = [changed_sentence[i], changed_sentence[i+1]]
        target = changed_sentence[i+2]
        trigrams.append((context, target))
       
vocab = set(splited_text)
vocab_size = len(vocab)
word_to_ix = {word: i for i, word in enumerate(vocab)}  

#trainning
losses = []
loss_function = nn.NLLLoss()
CONTEXT_SIZE = 2
EMBEDDING_DIM = 10
loss_function = nn.NLLLoss()
model = NGramLanguageModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
optimizer = optim.SGD(model.parameters(), lr=0.01)
for epoch in range(15):
    total_loss = torch.Tensor([0])
    for context, target in trigrams:
        context_idxs = torch.tensor([word_to_ix[w] for w in context], dtype=torch.long)
        model.zero_grad()
        log_probs = model(context_idxs)
        loss = loss_function(log_probs, torch.tensor([word_to_ix[target]], dtype=torch.long))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    losses.append(total_loss)
print("losses:" , losses)
print()

#Sanity check
print("predict_text: 'The mathematician ran to the store .'")
check_sentence = 'The mathematician ran to the store .'
model.predict_function(check_sentence,word_to_ix)
print()

#compare "physicist" with "philosopher"
prob = model.get_softmax(torch.tensor([word_to_ix['START_OF_SENTENCE'],word_to_ix['The']],dtype=torch.long))
print('probability of physicist and philosopher in the gap')
print('physicist:',prob[0][word_to_ix['physicist']])
print('philosopher:',prob[0][word_to_ix['philosopher']])
print()

#calculate cosines
physicist = model.embeddings(torch.tensor([word_to_ix['physicist']], dtype=torch.long))
philosopher = model.embeddings(torch.tensor([word_to_ix['philosopher']], dtype=torch.long))
mathematician = model.embeddings(torch.tensor([word_to_ix['mathematician']], dtype=torch.long))
print('similarity between physicist with mathematician:',F.cosine_similarity(mathematician,physicist,dim=1))
print('similarity between philosopher with mathematician:',F.cosine_similarity(mathematician,philosopher,dim=1))
