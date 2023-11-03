## Generate Synthetic Data
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import pandas as pd
import seaborn as sns
from tqdm import tqdm


def generate_synthetic_data(datasize=10000,vocab_size=2000,polarity_size=200):
    np.random.seed(123)
    temp = np.random.permutation(vocab_size)
    Pos = temp[:polarity_size//2]
    Neg = temp[polarity_size//2:polarity_size]
    Neu = temp[polarity_size:]
    
    data = []
    labels = []
    for i in range(datasize):
        temp = []
        if i < datasize/2:
            labels.append(+1.)
            temp = temp + list(np.random.choice(Pos,2)) + list(np.random.choice(Neu,10))
        else:
            labels.append(0.)
            temp = temp + list(np.random.choice(Neg,2)) + list(np.random.choice(Neu,10))

        data.append(temp)
        
    return data,labels,Pos,Neg,Neu

def make_data(doc_list,word2idx,max_len):
    import re
    vocab = list(word2idx.keys())[1:]
    token_list = []
    for doc in doc_list:
        temp = []
        splitted_doc = re.sub('<br />','',re.sub('[()""'',.:;#&?!]','',doc.lower())).split(' ')[:max_len]
        for word in splitted_doc:
            if word in vocab:
                temp.append(word2idx[word])
            else:
                temp.append(0)

        if len(temp) < max_len:
            n_pad = max_len - len(temp)
            temp.extend([0] * n_pad)
        token_list.append(temp)
    return token_list

class MyDataSet(Data.Dataset):
    def __init__(self, input_ids,label):
        self.input_ids = input_ids
        self.label = label
  
    def __len__(self):
        return len(self.input_ids)
  
    def __getitem__(self, idx):
        return self.input_ids[idx], self.label[idx]
    
def TanhMax(a):
    return (torch.exp(a) - torch.exp(-a)) / (torch.exp(a) + torch.exp(-a)).sum(dim=-1).unsqueeze(dim=-1)
    
class SelfAttnClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim=1, scale=1,score_function='dot', activation='TanhMax'):
        super(SelfAttnClassifier, self).__init__()
        
        self.embeddings = nn.Embedding(vocab_size, embed_dim)
        self.embeddings.weight.data.uniform_(-0.1, 0.1)
        self.scale = scale
        self.Wk = nn.Linear(embed_dim,hidden_dim)
        self.Wv = nn.Linear(embed_dim,hidden_dim)
        self.Wq = nn.Linear(embed_dim,hidden_dim)
        self.score_function = score_function
        if self.score_function == 'additive':
            self.v = nn.Parameter(torch.randn(hidden_dim, 1))
        
        if activation.lower() == 'softmax':
            self.activate = nn.Softmax(dim=-1)
        elif activation.lower() == 'tanhmax':
            self.activate = TanhMax
        elif activation.lower() == 'tanh':
            self.activate = nn.Tanh()
        
        self.dropout = nn.Dropout(0.5)
        self.dense = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
        

    # batch_size * sent_l * dim
    def forward(self, seq_ids, labels=None):
        seq_embs = self.embeddings(seq_ids)
        
        Q = self.Wq(seq_embs)
        K = self.Wk(seq_embs)
        V = self.Wv(seq_embs)
        
        if self.score_function == 'dot':
            scores = (Q*K).sum(dim=-1) / self.scale
        elif self.score_function == 'additive':
            scores = torch.matmul(nn.Tanh()(K+Q),self.v).squeeze(-1)
        #print(scores.shape,V.shape)
            
        if labels is None:
            scores = scores 
        else:
            scores = scores * labels.unsqueeze(dim=-1)
            
        attn = self.activate(scores)
        #print(attn.shape)
        final_vec = torch.bmm(attn.unsqueeze(1), V).squeeze(1)
        senti_scores = self.dense(self.dropout(final_vec))
        probs = self.sigmoid(senti_scores)
        return probs

def plot_cm_matrix(model,batch,labels,save_url):
    # 计算模型预测的准确率
    pred_labels = []
    true_labels = []
    model.eval()
    with torch.no_grad():
        for i in range(len(batch)):
            input_ids = torch.LongTensor(batch[i]).unsqueeze(dim=0)
            output = model(input_ids.cuda(),labels=labels[i].cuda().float())
            if output > 0.5:
                pred_labels.append(1)
            else:
                pred_labels.append(0)
            true_labels.append(labels[i].cpu().numpy().tolist())
            input_ids.cpu()
            labels[i].cpu()

    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(true_labels, pred_labels)

    from mlxtend.plotting import plot_confusion_matrix
    fig, ax = plot_confusion_matrix(conf_mat=cm, figsize=(6, 6), cmap=plt.cm.Greens)
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.savefig(save_url)
    plt.show()

    accuracy = (cm[0][0] + cm[1][1]) / sum(sum(cm))
    print('accuracy:{:.3f}%'.format(accuracy*100))
    
def compute_coefficients(model,batch,Pos,Neg,Neu):
    token_score = dict()
    token_weight = dict()
    num_dict = dict()
    for sample in batch:
        with torch.no_grad():
            seq_embs = model.embeddings(sample.cuda())

            Q = model.Wq(seq_embs)
            K = model.Wk(seq_embs)
            V = model.Wv(seq_embs)

            if model.score_function == 'dot':
                scores = (Q*K).sum(dim=-1) / model.scale
            elif model.score_function == 'additive':
                scores = torch.matmul(nn.Tanh()(K+Q),model.v).squeeze(-1)
            attn = model.activate(scores)
            sample.cpu()

            sent = sample.numpy().tolist()
            for i in range(len(sent)):
                word = sent[i]
                if word in token_score.keys():
                    token_score[word] += scores[i].cpu().numpy().tolist()
                    token_weight[word] += attn[i].cpu().numpy().tolist()
                    num_dict[word] += 1
                else:
                    token_score[word] = scores[i].cpu().numpy().tolist()
                    token_weight[word] = attn[i].cpu().numpy().tolist()
                    num_dict[word] = 1
                    
    pos_index = []
    pos_score = []
    pos_weight = []
    for tok in Pos:
        if tok in token_score.keys():
            num = num_dict[tok]
            pos_index.append(tok)
            pos_score.append(token_score[tok] / num)
            pos_weight.append(token_weight[tok] / num)

    neg_index = []
    neg_score = []
    neg_weight = []
    for tok in Neg:
        if tok in token_score.keys():
            num = num_dict[tok]
            neg_index.append(tok)
            neg_score.append(token_score[tok] / num)
            neg_weight.append(token_weight[tok] / num)

    neutral_index = []
    neutral_score = []
    neutral_weight = []
    for tok in Neu:
        if tok in token_score.keys():
            num = num_dict[tok]
            neutral_index.append(tok)
            neutral_score.append(token_score[tok] / num)
            neutral_weight.append(token_weight[tok] / num)   
            
    return pos_index,pos_score,pos_weight,neg_index,neg_score,neg_weight,neutral_index,neutral_score,neutral_weight

def GradientImportance(model,test_batch,test_labels,mark):
    kendall = []
    pos_kendall = []
    neg_kendall = []
    num1 = 0 # 0.05
    num2 = 0 # 0.01
    pos_tot_num = 0
    pos_num1 = 0
    pos_num2 = 0
    neg_tot_num = 0
    neg_num1 = 0
    neg_num2 = 0
    for i,sample in tqdm(enumerate(test_batch)):
        seq_embs = model.embeddings(sample.cuda())
        Q = model.Wq(seq_embs)
        K = model.Wk(seq_embs)
        V = model.Wv(seq_embs)
        scores = (Q*K).sum(dim=-1)
        scores = scores/model.scale
        attn = model.activate(scores)

        df = pd.DataFrame({'attn':attn.cpu().detach().numpy(),
                      'feaImp':torch.matmul(V,model.Wv.weight.grad).sum(-1).abs().cpu().detach().numpy()})
        tau,p = stats.kendalltau(df['attn'].abs(),df['feaImp'])
        if np.isnan(tau) == True:
            pass
        else:
            kendall.append(tau)
            if test_labels[i].cpu().numpy() == 1:    
                pos_kendall.append(tau)
                pos_tot_num += 1
                if p < 0.01:
                    pos_num2 += 1
                    num2 += 1
                if p < 0.05:
                    pos_num1 += 1
                    num1 += 1
            else:
                neg_kendall.append(tau)
                neg_tot_num += 1
                if p < 0.01:
                    neg_num2 += 1
                    num2 += 1
                if p < 0.05:
                    neg_num1 += 1
                    num1 += 1

        sample.cpu()
        
    print('Both label')
    print('\tp-value < 0.05:',num1,' ratio:', num1 / len(kendall))
    print('\tp-value < 0.01:',num2,' ratio:', num2 / len(kendall))
    print('\tBoth-Mean:',np.mean(kendall),'Std:',np.std(kendall))

    print('\nPos')
    print('\tp-value < 0.05:',pos_num1,' ratio:', pos_num1 / pos_tot_num )
    print('\tp-value < 0.01:',pos_num2,' ratio:', pos_num2 / pos_tot_num )
    print('\tPos-Mean:',np.mean(pos_kendall),'Std:',np.std(pos_kendall))

    print('\nNeg')
    print('\tp-value < 0.05:',neg_num1,' ratio:', neg_num1 / neg_tot_num )
    print('\tp-value < 0.01:',neg_num2,' ratio:', neg_num2 / neg_tot_num )
    print('\tNeg-Mean:',np.mean(neg_kendall),'Std:',np.std(neg_kendall))
    
    plt.rcParams['axes.unicode_minus'] = False 
    
    plt.hist(kendall,edgecolor='black',density=True,stacked=False,bins=25,alpha=0.5,range=(-1,1))
    if mark.lower() == 'softmax':
        plt.savefig('graph/gradient/picture1.png')
    elif mark.lower() == 'tanhmax':
        plt.savefig('graph/gradient/picture4.png')
    plt.show()
    
    plt.hist(pos_kendall,edgecolor='black',label='pos',density=True,bins=20,alpha=0.5,range=(-1,1),color='red')
    plt.hist(neg_kendall,edgecolor='black',label='neg',density=True,bins=20,alpha=0.5,range=(-1,1),color='blue')
    plt.legend()
    if mark.lower() == 'softmax':
        plt.savefig('graph/gradient/picture2.png')
    elif mark.lower() == 'tanhmax':
        plt.savefig('graph/gradient/picture5.png')
    plt.show()
    
    plt.scatter(range(len(pos_kendall)),pos_kendall,s=10,color='red',label='Pos',alpha=0.5)
    plt.scatter(range(len(pos_kendall),len(pos_kendall)+len(neg_kendall)),neg_kendall,s=10,color='blue',label='Neg',alpha=0.5)
    plt.ylim(-1.1,1.1)
    plt.legend()
    if mark.lower() == 'softmax':
        plt.savefig('graph/gradient/picture3.png')
    elif mark.lower() == 'tanhmax':
        plt.savefig('graph/gradient/picture6.png')
    plt.show()
    
def FeatureErasure(model,test_batch,test_labels,mark):
    kendall_loo = []
    pos_kendall_loo = []
    neg_kendall_loo = []
    num_loo = 0
    num1_loo = 0 # 0.05
    num2_loo = 0 # 0.01
    pos_num1_loo = 0
    pos_num2_loo = 0
    neg_num1_loo = 0
    neg_num2_loo = 0
    for index,sample in tqdm(enumerate(test_batch)):
        seq_embs = model.embeddings(sample.cuda())
        Q = model.Wq(seq_embs)
        K = model.Wk(seq_embs)
        V = model.Wv(seq_embs)
        scores = (Q*K).sum(dim=-1)
        scores = scores/model.scale
        attn = model.activate(scores)
        final_vec = torch.matmul(attn.unsqueeze(1).transpose(0,1), V).squeeze(1)
        senti_scores = model.dense(model.dropout(final_vec))
        probs = model.sigmoid(senti_scores)

        container = []
        for i in range(attn.size(0)):
            temp_attn = attn.clone()
            temp_attn[i] = 0.
            temp_final_vec = torch.matmul(temp_attn.unsqueeze(1).transpose(0,1), V).squeeze(1)
            temp_senti_scores = model.dense(model.dropout(temp_final_vec))
            temp_probs = model.sigmoid(temp_senti_scores)

            temp_delta = probs - temp_probs

            container.append(temp_delta.cpu().detach().numpy()[0][0])

        df = pd.DataFrame({'attn':attn.cpu().detach().numpy(),
                      'feaImp':container})
        tau,p = stats.kendalltau(df['attn'],df['feaImp'])
        if np.isnan(tau):
            continue
        else:         
            kendall_loo.append(tau)
            num_loo += 1
            if test_labels[index].cpu().numpy() == 1:    
                pos_kendall_loo.append(tau)
                if p < 0.01:
                    pos_num2_loo += 1
                    num2_loo += 1
                if p < 0.05:
                    pos_num1_loo += 1
                    num1_loo += 1
            else:
                neg_kendall_loo.append(tau)
                if p < 0.01:
                    neg_num2_loo += 1
                    num2_loo += 1
                if p < 0.05:
                    neg_num1_loo += 1
                    num1_loo += 1
        sample.cpu()
        
    print('Both label')
    print('\tp-value < 0.05:',num1_loo,' ratio:',num1_loo / num_loo)
    print('\tp-value < 0.01:',num2_loo,' ratio:',num2_loo / num_loo)
    print('\ttotal num:',num_loo,' mean:',np.mean(kendall_loo),' std:',np.std(kendall_loo))

    print('Pos')
    print('\tp-value < 0.05:',pos_num1_loo,' ratio:',pos_num1_loo / len(pos_kendall_loo))
    print('\tp-value < 0.01:',pos_num2_loo,' ratio:',pos_num2_loo / len(pos_kendall_loo))
    print('\ttotal num:',len(pos_kendall_loo),' mean:',np.mean(pos_kendall_loo),' std:',np.std(pos_kendall_loo))

    print('Neg')
    print('\tp-value < 0.05:',neg_num1_loo,' ratio:',neg_num1_loo / len(neg_kendall_loo))
    print('\tp-value < 0.01:',neg_num2_loo,' ratio:',neg_num2_loo / len(neg_kendall_loo))
    print('\ttotal num:',len(neg_kendall_loo),' mean:',np.mean(neg_kendall_loo),' std:',np.std(neg_kendall_loo))
    
    plt.rcParams['axes.unicode_minus'] = False 

    plt.hist(kendall_loo,edgecolor='black',density=True,bins=25,alpha=0.5,range=(-1,1))
    if mark.lower() == 'softmax':
        plt.savefig('graph/erasure/picture1.png')
    elif mark.lower() == 'tanhmax':
        plt.savefig('graph/erasure/picture4.png')
    plt.show()
    
    plt.hist(pos_kendall_loo,edgecolor='black',density=True,color='red',bins=25,alpha=0.5,range=(-1,1),label='Pos')
    plt.hist(neg_kendall_loo,edgecolor='black',density=True,color='blue',bins=25,alpha=0.5,range=(-1,1),label='Neg')
    plt.legend()
    if mark.lower() == 'softmax':
        plt.savefig('graph/erasure/picture2.png')
    elif mark.lower() == 'tanhmax':
        plt.savefig('graph/erasure/picture5.png')
    plt.show()
    
    plt.scatter(range(len(pos_kendall_loo)),pos_kendall_loo,s=10,label='Pos',color='red',alpha=0.5)
    plt.scatter(range(len(neg_kendall_loo)),neg_kendall_loo,s=10,label='Neg',color='blue',alpha=0.5)
    plt.ylim(-1.1,1.1)
    plt.legend()
    if mark.lower() == 'softmax':
        plt.savefig('graph/erasure/picture3.png')
    elif mark.lower() == 'tanhmax':
        plt.savefig('graph/erasure/picture6.png')
    plt.show()
    
def FeatureErasureABS(model,test_batch,test_labels,mark):
    kendall_loo_abs = []
    pos_kendall_loo_abs = []
    neg_kendall_loo_abs = []
    num_loo_abs = 0
    num1_loo_abs = 0 # 0.05
    num2_loo_abs = 0 # 0.01
    pos_num1_loo_abs = 0
    pos_num2_loo_abs = 0
    neg_num1_loo_abs = 0
    neg_num2_loo_abs = 0
    for index,sample in tqdm(enumerate(test_batch)):
        seq_embs = model.embeddings(sample.cuda())
        Q = model.Wq(seq_embs)
        K = model.Wk(seq_embs)
        V = model.Wv(seq_embs)
        scores = (Q*K).sum(dim=-1)
        scores = scores/model.scale
        attn = model.activate(scores)
        final_vec = torch.matmul(attn.unsqueeze(1).transpose(0,1), V).squeeze(1)
        senti_scores = model.dense(model.dropout(final_vec))
        probs = model.sigmoid(senti_scores)

        container = []
        for i in range(attn.size(0)):
            temp_attn = attn.clone()
            temp_attn[i] = 0.
            temp_final_vec = torch.matmul(temp_attn.unsqueeze(1).transpose(0,1), V).squeeze(1)
            temp_senti_scores = model.dense(model.dropout(temp_final_vec))
            temp_probs = model.sigmoid(temp_senti_scores)

            temp_delta = (temp_probs - probs).abs()

            container.append(temp_delta.cpu().detach().numpy()[0][0])

        df = pd.DataFrame({'attn':attn.cpu().detach().numpy(),
                      'feaImp':container})
        tau,p = stats.kendalltau(df['attn'].abs(),df['feaImp'])
        if np.isnan(tau):
            continue
        else:         
            kendall_loo_abs.append(tau)
            num_loo_abs += 1
            if test_labels[index].cpu().numpy() == 1:    
                pos_kendall_loo_abs.append(tau)
                if p < 0.01:
                    pos_num2_loo_abs += 1
                    num2_loo_abs += 1
                if p < 0.05:
                    pos_num1_loo_abs += 1
                    num1_loo_abs += 1
            else:
                neg_kendall_loo_abs.append(tau)
                if p < 0.01:
                    neg_num2_loo_abs += 1
                    num2_loo_abs += 1
                if p < 0.05:
                    neg_num1_loo_abs += 1
                    num1_loo_abs += 1
        sample.cpu()
        
    print('Both label')
    print('\tp-value < 0.05:',num1_loo_abs,' ratio:',num1_loo_abs / num_loo_abs)
    print('\tp-value < 0.01:',num2_loo_abs,' ratio:',num2_loo_abs / num_loo_abs)
    print('\ttotal num:',num_loo_abs,' mean:',np.mean(kendall_loo_abs),' std:',np.std(kendall_loo_abs))

    print('Pos')
    print('\tp-value < 0.05:',pos_num1_loo_abs,' ratio:',pos_num1_loo_abs / len(pos_kendall_loo_abs))
    print('\tp-value < 0.01:',pos_num2_loo_abs,' ratio:',pos_num2_loo_abs / len(pos_kendall_loo_abs))
    print('\ttotal num:',len(pos_kendall_loo_abs),' mean:',np.mean(pos_kendall_loo_abs),' std:',np.std(pos_kendall_loo_abs))

    print('Neg')
    print('\tp-value < 0.05:',neg_num1_loo_abs,' ratio:',neg_num1_loo_abs / len(neg_kendall_loo_abs))
    print('\tp-value < 0.01:',neg_num2_loo_abs,' ratio:',neg_num2_loo_abs / len(neg_kendall_loo_abs))
    print('\ttotal num:',len(neg_kendall_loo_abs),' mean:',np.mean(neg_kendall_loo_abs),' std:',np.std(neg_kendall_loo_abs))
    
    plt.rcParams['axes.unicode_minus'] = False 
    plt.hist(kendall_loo_abs,edgecolor='black',density=True,bins=25,alpha=0.5,range=(-1,1))
    if mark.lower() == 'softmax':
        plt.savefig('graph/erasure_abs/picture1.png')
    elif mark.lower() == 'tanhmax':
        plt.savefig('graph/erasure_abs/picture4.png')
    plt.show()
    
    plt.hist(pos_kendall_loo_abs,edgecolor='black',density=True,color='red',bins=25,alpha=0.5,range=(-1,1),label='Pos')
    plt.hist(neg_kendall_loo_abs,edgecolor='black',density=True,color='blue',bins=25,alpha=0.5,range=(-1,1),label='Neg')
    plt.legend()
    if mark.lower() == 'softmax':
        plt.savefig('graph/erasure_abs/picture2.png')
    elif mark.lower() == 'tanhmax':
        plt.savefig('graph/erasure_abs/picture5.png')
    plt.show()
    
    plt.scatter(range(len(pos_kendall_loo_abs)),pos_kendall_loo_abs,s=10,label='Pos',color='red',alpha=0.5)
    plt.scatter(range(len(neg_kendall_loo_abs)),neg_kendall_loo_abs,s=10,label='Neg',color='blue',alpha=0.5)
    plt.ylim(-1.1,1.1)
    plt.legend()
    if mark.lower() == 'softmax':
        plt.savefig('graph/erasure_abs/picture3.png')
    elif mark.lower() == 'tanhmax':
        plt.savefig('graph/erasure_abs/picture6.png')
    plt.show()
    
def AttnPermutation(model,test_batch,test_labels,mark):
    median_list = []
    pos_median = []
    neg_median = []
    instance_label = []
    value_group = []

    for index,sample in tqdm(enumerate(test_batch)):
        seq_embs = model.embeddings(sample.cuda())
        Q = model.Wq(seq_embs)
        K = model.Wk(seq_embs)
        V = model.Wv(seq_embs)
        scores = (Q*K).sum(dim=-1)
        scores = scores/model.scale
        attn = model.activate(scores)
        final_vec = torch.matmul(attn.unsqueeze(1).transpose(0,1), V).squeeze(1)
        senti_scores = model.dense(model.dropout(final_vec))
        probs = model.sigmoid(senti_scores)

        container = []
        for i in range(100):
            idx = torch.randperm(attn.shape[0])
            temp_attn = attn[idx].view(attn.size())
            temp_final_vec = torch.matmul(temp_attn.unsqueeze(1).transpose(0,1), V).squeeze(1)
            temp_senti_scores = model.dense(model.dropout(temp_final_vec))
            temp_probs = model.sigmoid(temp_senti_scores)

            temp_delta = (temp_probs - probs).abs()

            container.append(temp_delta.cpu().detach().numpy()[0][0])

        temp_value = np.median(container)

        median_list.append(temp_value)
        if test_labels[index].cpu().numpy() == 1:    
            instance_label.append('Pos')
            pos_median.append(temp_value)
        else:
            instance_label.append('Neg')
            neg_median.append(temp_value)

        max_attn = attn.max().cpu().detach().numpy()
        if max_attn < 0.25:
            value_group.append('[0.,0.25)')
        elif max_attn < 0.5:
            value_group.append('[0.25,0.5)')
        elif max_attn < 0.75:
            value_group.append('[0.5,0.75)')
        else:
            value_group.append('[0.75,1.0)')
        sample.cpu()
        
    print('Both label')
    print('\ttotal num:',len(median_list),' mean:',np.mean(median_list),' std:',np.std(median_list))
    print('Pos')
    print('\ttotal num:',len(pos_median),' mean:',np.mean(pos_median),' std:',np.std(pos_median))
    print('Neg')
    print('\ttotal num:',len(neg_median),' mean:',np.mean(neg_median),' std:',np.std(neg_median))
    
    plt.rcParams['axes.unicode_minus'] = False 

    plt.hist(median_list,edgecolor='black',density=True,bins=25,alpha=0.5,range=(0,1))
    if mark.lower() == 'softmax':
        plt.savefig('graph/permutation/picture1.png')
    elif mark.lower() == 'tanhmax':
        plt.savefig('graph/permutation/picture4.png')
    plt.show()
    
    plt.hist(pos_median,edgecolor='black',density=True,bins=25,alpha=0.5,color='red',label='Pos',range=(0,1))
    plt.hist(neg_median,edgecolor='black',density=True,bins=25,alpha=0.5,color='blue',label='Neg',range=(0,1))
    plt.legend()
    if mark.lower() == 'softmax':
        plt.savefig('graph/permutation/picture2.png')
    elif mark.lower() == 'tanhmax':
        plt.savefig('graph/permutation/picture5.png')
    plt.show()
    
    df = pd.DataFrame({'median':median_list,'label':instance_label,'group':value_group})
    sns.violinplot(x = "median", 
                   y = "group", 
                   hue = "label", 
                   data = df, 
                   scale = 'count', 
                   split = True, 
                   cut = 0,
                   order = ['[0.,0.25)','[0.25,0.5)','[0.5,0.75)','[0.75,1.0)'],
                   palette = 'RdBu' 
                  )
    if mark.lower() == 'softmax':
        plt.savefig('graph/permutation/picture3.png')
    elif mark.lower() == 'tanhmax':
        plt.savefig('graph/permutation/picture6.png')
    plt.show()
    print(df.groupby(['label','group']).describe())
    
def AttnRandom(model,test_batch,test_labels,mark):
    median_list = []
    pos_median = []
    neg_median = []
    instance_label = []
    value_group = []

    for index,sample in tqdm(enumerate(test_batch)):
        seq_embs = model.embeddings(sample.cuda())
        Q = model.Wq(seq_embs)
        K = model.Wk(seq_embs)
        V = model.Wv(seq_embs)
        scores = (Q*K).sum(dim=-1)
        scores = scores/model.scale
        attn = model.activate(scores)
        final_vec = torch.matmul(attn.unsqueeze(1).transpose(0,1), V).squeeze(1)
        senti_scores = model.dense(model.dropout(final_vec))
        probs = model.sigmoid(senti_scores)

        container = []
        for i in range(100):
            if mark.lower() == 'softmax':
                temp_attn = torch.Tensor(attn.size()).uniform_(0,1).cuda()
            elif mark.lower() == 'tanhmax':
                temp_attn = torch.Tensor(attn.size()).uniform_(-1,1).cuda()
            temp_final_vec = torch.matmul(temp_attn.unsqueeze(1).transpose(0,1), V).squeeze(1)
            temp_senti_scores = model.dense(model.dropout(temp_final_vec))
            temp_probs = model.sigmoid(temp_senti_scores)

            temp_delta = (temp_probs - probs).abs()

            container.append(temp_delta.cpu().detach().numpy()[0][0])

        temp_value = np.median(container)

        median_list.append(temp_value)
        if test_labels[index].cpu().numpy() == 1:    
            instance_label.append('Pos')
            pos_median.append(temp_value)
        else:
            instance_label.append('Neg')
            neg_median.append(temp_value)

        max_attn = attn.abs().max().cpu().detach().numpy()
        if max_attn < 0.25:
            value_group.append('[0.,0.25)')
        elif max_attn < 0.5:
            value_group.append('[0.25,0.5)')
        elif max_attn < 0.75:
            value_group.append('[0.5,0.75)')
        else:
            value_group.append('[0.75,1.0)')
        sample.cpu()
        
    print('Both label')
    print('\ttotal num:',len(median_list),' mean:',np.mean(median_list),' std:',np.std(median_list))
    print('Pos')
    print('\ttotal num:',len(pos_median),' mean:',np.mean(pos_median),' std:',np.std(pos_median))
    print('Neg')
    print('\ttotal num:',len(neg_median),' mean:',np.mean(neg_median),' std:',np.std(neg_median))
    
    plt.hist(median_list,edgecolor='black',density=True,bins=25,alpha=0.5,range=(0,1))
    if mark.lower() == 'softmax':
        plt.savefig('graph/random/picture1.png')
    elif mark.lower() == 'tanhmax':
        plt.savefig('graph/random/picture4.png')
    plt.show()
    
    plt.hist(pos_median,edgecolor='black',density=True,bins=25,alpha=0.5,color='red',label='Pos',range=(0,1))
    plt.hist(neg_median,edgecolor='black',density=True,bins=25,alpha=0.5,color='blue',label='Neg',range=(0,1))
    plt.legend()
    if mark.lower() == 'softmax':
        plt.savefig('graph/random/picture2.png')
    elif mark.lower() == 'tanhmax':
        plt.savefig('graph/random/picture5.png')
    plt.show()
    
    df = pd.DataFrame({'median':median_list,'label':instance_label,'group':value_group})
    import seaborn as sns
    sns.violinplot(x = "median", 
                   y = "group", 
                   hue = "label", 
                   data = df, 
                   scale = 'count', 
                   split = True, 
                   cut = 0,               
                   order = ['[0.,0.25)','[0.25,0.5)','[0.5,0.75)','[0.75,1.0)'],
                   palette = 'RdBu' 
                  )
    if mark.lower() == 'softmax':
        plt.savefig('graph/random/picture3.png')
    elif mark.lower() == 'tanhmax':
        plt.savefig('graph/random/picture6.png')
    plt.show()
    print(df.groupby(['label','group']).describe())