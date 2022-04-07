# importing all the required libraries
from torchcrf import CRF
from transformers import BertPreTrainedModel, BertModel, BertConfig
from transformers import BertConfig
from transformers import BertTokenizer
import torch.nn as nn
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import regex as re
from io import StringIO
import letsum
import letsum_test
from nltk.tokenize import word_tokenize, sent_tokenize
import transformers
import sentencepiece
import os
import torch
from tqdm import tqdm
from transformers import pipeline
from transformers import BertTokenizerFast, EncoderDecoderModel
import torch
import graphicalModel
#from predict import predict_file
import sys
import os
import logging
import argparse
from tqdm import tqdm, trange
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
sys.setrecursionlimit(1500)
from summarizer import Summarizer
from streamlit import caching

#!pip install rouge
from rouge import Rouge
rouge = Rouge()

#!pip install nltk==3.6.2
import numpy as np
import nltk
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

#!pip install bert-score
from bert_score import score as score_



############################################################ JointBERT combination start ############################################
from seqeval.metrics import precision_score, recall_score, f1_score


class IntentClassifier(nn.Module):
    def __init__(self, input_dim, num_intent_labels, dropout_rate=0.):
        super(IntentClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, num_intent_labels)

    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)


class SlotClassifier(nn.Module):
    def __init__(self, input_dim, num_slot_labels, dropout_rate=0.):
        super(SlotClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, num_slot_labels)

    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)


class JointBERT(BertPreTrainedModel):
    def __init__(self, config, intent_label_lst, slot_label_lst):
        super(JointBERT, self).__init__(config)
        #self.args = args
        self.num_intent_labels = len(intent_label_lst)
        self.num_slot_labels = len(slot_label_lst)
        #self.config = BertConfig
        self.bert = BertModel(config=config)  # Load pretrained bert

        self.intent_classifier = IntentClassifier(
            config.hidden_size, self.num_intent_labels, 0.1)
        self.slot_classifier = SlotClassifier(
            config.hidden_size, self.num_slot_labels, 0.1)

        if 0:
            self.crf = CRF(num_tags=self.num_slot_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, token_type_ids, intent_label_ids, slot_labels_ids):
        outputs = self.bert(input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids)  # sequence_output, pooled_output, (hidden_states), (attentions)
        sequence_output = outputs[0]
        pooled_output = outputs[1]  # [CLS]

        intent_logits = self.intent_classifier(pooled_output)
        slot_logits = self.slot_classifier(sequence_output)

        total_loss = 0
        # 1. Intent Softmax
        if intent_label_ids is not None:
            if self.num_intent_labels == 1:
                intent_loss_fct = nn.MSELoss()
                intent_loss = intent_loss_fct(
                    intent_logits.view(-1), intent_label_ids.view(-1))
            else:
                intent_loss_fct = nn.CrossEntropyLoss()
                intent_loss = intent_loss_fct(
                    intent_logits.view(-1, self.num_intent_labels), intent_label_ids.view(-1))
            total_loss += intent_loss

        # 2. Slot Softmax
        if slot_labels_ids is not None:
            if 0:
                slot_loss = self.crf(
                    slot_logits, slot_labels_ids, mask=attention_mask.byte(), reduction='mean')
                slot_loss = -1 * slot_loss  # negative log-likelihood
            else:
                slot_loss_fct = nn.CrossEntropyLoss(
                    ignore_index=0)
                # Only keep active parts of the loss
                if attention_mask is not None:
                    active_loss = attention_mask.view(-1) == 1
                    active_logits = slot_logits.view(-1,
                                                     self.num_slot_labels)[active_loss]
                    active_labels = slot_labels_ids.view(-1)[active_loss]
                    slot_loss = slot_loss_fct(active_logits, active_labels)
                else:
                    slot_loss = slot_loss_fct(
                        slot_logits.view(-1, self.num_slot_labels), slot_labels_ids.view(-1))
            total_loss += 1.0 * slot_loss

        # add hidden states and attention if they are here
        outputs = ((intent_logits, slot_logits),) + outputs[2:]

        outputs = (total_loss,) + outputs

        # (loss), logits, (hidden_states), (attentions) # Logits is a tuple of intent and slot logits
        return outputs


def get_intent_labels():
    return [label.strip() for label in open("intent_label.txt", 'r', encoding='utf-8')]


def get_slot_labels():
    return [label.strip() for label in open("slot_label.txt", 'r', encoding='utf-8')]


def load_tokenizer():
    return BertTokenizer.from_pretrained('bert-base-uncased')

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

@st.cache() 
def load_model():
    model = JointBERT.from_pretrained("AnonymousSub/bert_snips",config = "config.json", intent_label_lst=get_intent_labels(), slot_label_lst=get_slot_labels())
    model.to(get_device())
    model.eval()
    return model


def read_input_file(pred_config_text):

    return pred_config_text

@st.cache()
def convert_input_file_to_tensor_dataset(lines,
                                         tokenizer,
                                         pad_token_label_id,
                                         cls_token_segment_id=0,
                                         pad_token_segment_id=0,
                                         sequence_a_segment_id=0,
                                         mask_padding_with_zero=True):
    # Setting based on the current model type
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    unk_token = tokenizer.unk_token
    pad_token_id = tokenizer.pad_token_id

    all_input_ids = []
    all_attention_mask = []
    all_token_type_ids = []
    all_slot_label_mask = []

    for words in lines:
        tokens = []
        slot_label_mask = []
        for word in words:
            word_tokens = tokenizer.tokenize(word)
            if not word_tokens:
                word_tokens = [unk_token]  # For handling the bad-encoded word
            tokens.extend(word_tokens)
            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
            slot_label_mask.extend(
                [pad_token_label_id + 1] + [pad_token_label_id] * (len(word_tokens) - 1))

        # Account for [CLS] and [SEP]
        special_tokens_count = 2
        if len(tokens) > 128 - special_tokens_count:
            tokens = tokens[: (128 - special_tokens_count)]
            slot_label_mask = slot_label_mask[:(128 - special_tokens_count)]

        # Add [SEP] token
        tokens += [sep_token]
        token_type_ids = [sequence_a_segment_id] * len(tokens)
        slot_label_mask += [pad_token_label_id]

        # Add [CLS] token
        tokens = [cls_token] + tokens
        token_type_ids = [cls_token_segment_id] + token_type_ids
        slot_label_mask = [pad_token_label_id] + slot_label_mask

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = 128 - len(input_ids)
        input_ids = input_ids + ([pad_token_id] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
        slot_label_mask = slot_label_mask + ([pad_token_label_id] * padding_length)

        all_input_ids.append(input_ids)
        all_attention_mask.append(attention_mask)
        all_token_type_ids.append(token_type_ids)
        all_slot_label_mask.append(slot_label_mask)

    # Change to Tensor
    all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
    all_attention_mask = torch.tensor(all_attention_mask, dtype=torch.long)
    all_token_type_ids = torch.tensor(all_token_type_ids, dtype=torch.long)
    all_slot_label_mask = torch.tensor(all_slot_label_mask, dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_attention_mask,all_token_type_ids, all_slot_label_mask)

    return dataset

@st.cache()
def predict(file_content):
    # load model and args
    #pred_config
    #args = get_args(pred_config)
    device = get_device()
    model = load_model()
    #logger.info(args)

    intent_label_lst = get_intent_labels()
    slot_label_lst = get_slot_labels()

    # Convert input file to TensorDataset
    pad_token_label_id = 0
    tokenizer = load_tokenizer()
    # lines = read_input_file(pred_config)
    lines = file_content
    dataset = convert_input_file_to_tensor_dataset(lines,tokenizer, pad_token_label_id)

    # Predict
    sampler = SequentialSampler(dataset)
    data_loader = DataLoader(dataset, sampler=sampler,batch_size=16)

    all_slot_label_mask = None
    intent_preds = None
    slot_preds = None

    for batch in tqdm(data_loader, desc="Predicting"):
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            inputs = {"input_ids": batch[0],
                      "attention_mask": batch[1],
                      "intent_label_ids": None,
                      "slot_labels_ids": None}
            if "bert" != "distilbert":
                inputs["token_type_ids"] = batch[2]
            outputs = model(**inputs)
            _, (intent_logits, slot_logits) = outputs[:2]

            # Intent Prediction
            if intent_preds is None:
                intent_preds = intent_logits.detach().cpu().numpy()
            else:
                intent_preds = np.append(
                    intent_preds, intent_logits.detach().cpu().numpy(), axis=0)

            # Slot prediction
            if slot_preds is None:
                if 0:
                    # decode() in `torchcrf` returns list with best index directly
                    slot_preds = np.array(model.crf.decode(slot_logits))
                else:
                    slot_preds = slot_logits.detach().cpu().numpy()
                all_slot_label_mask = batch[3].detach().cpu().numpy()
            else:
                if 0:
                    slot_preds = np.append(slot_preds, np.array(
                        model.crf.decode(slot_logits)), axis=0)
                else:
                    slot_preds = np.append(
                        slot_preds, slot_logits.detach().cpu().numpy(), axis=0)
                all_slot_label_mask = np.append(
                    all_slot_label_mask, batch[3].detach().cpu().numpy(), axis=0)
                del batch
    intent_preds = np.argmax(intent_preds, axis=1)

    if 1:
        slot_preds = np.argmax(slot_preds, axis=2)

    slot_label_map = {i: label for i, label in enumerate(slot_label_lst)}
    slot_preds_list = [[] for _ in range(slot_preds.shape[0])]

    for i in range(slot_preds.shape[0]):
        for j in range(slot_preds.shape[1]):
            if all_slot_label_mask[i, j] != pad_token_label_id:
                slot_preds_list[i].append(slot_label_map[slot_preds[i][j]])

    line_list = []
    for words, slot_preds, intent_pred in zip(lines, slot_preds_list, intent_preds):
    	line = ""
    	for word, pred in zip(words, slot_preds):
    		if pred == 'O':
    			line = line + word + " "
    		else:
    			line = line + "[{}:{}] ".format(word, pred)
    	xyz = "<" + intent_label_lst[intent_pred] + "> -> " + line.strip()
    	line_list.append(xyz)

    del model
    del intent_label_lst,slot_label_lst
    return line_list


def predict_file(file_content):

    #pred_config = Args()
    #pred_config.file_content = file_content
    #pred_config.model_dir = model_dir
    line_list = predict(file_content)
    return line_list

############################################################ JointBERT combination end ############################################

#@st.cache()
def calculate_metrics(test_list, summary):
    
    test = ""
    for sent in test_list:
        test += " ".join(sent)

    metrics_dict = {}
    
    # Rouge
    scores=rouge.get_scores(summary, test)
    f = scores[0]['rouge-l']['f']
    p = scores[0]['rouge-l']['p']
    r = scores[0]['rouge-l']['r']

    metrics_dict['F1-score-rouge-I'] = f
    metrics_dict['Precision-rouge-I'] = p
    metrics_dict['Recall-rouge-I'] = r

    # BLEU
    hypothesis =summary.split()
    reference = test.split()

    list_of_references = [reference] # list of references for all sentences in corpus.
    list_of_hypotheses = [hypothesis] # list of hypotheses that corresponds to list of references.
    score = nltk.translate.bleu_score.corpus_bleu(list_of_references, list_of_hypotheses)
    metrics_dict['Bleu-score'] = score

    # METEOR
    
    reference=test.split('\n')
    text=summary.split('\n')
    
    score=[]
    
    for j in text:
        score.append(nltk.translate.meteor_score.meteor_score(reference,j))
    
    metrics_dict['Meteor-score'] = np.mean(np.array(score))

    # BERT Score

    #hypothesis = test
    #temp_summary = summary

    #hypethesis = hypothesis.split('\n')
    #hypothesis = [' '.join(hypothesis[:])]
    #temp_summary = temp_summary.split('\n')
    #temp_summary = [' '.join(temp_summary[:])]
    #P, R, F1 = score_(temp_summary, hypothesis, lang = "en", verbose = True, rescale_with_baseline = True)
    
    #metrics_dict['Bert-score'] = F1.mean()

    # Intent Metric

    output = summary

    r = []
    pr = []
    test1 = test
    ### recall 
    t = 0
    p = 0
    for line in test1: 
            line = line.lower()
            if line.strip() in output:
                p += 1
                t += 1
    r = p / t

    ### precision
    ctr = 0
    flag = 0
    ctr1 = 0
    for sent in output.split('\n'):
        if sent!='\n':
            flag=0
        for line in test1:
            line=line.strip()
            if line in sent:
                flag=1
                break
        if flag==1:
            ctr1+=1
        ctr+=1
        #print(sent)
        
    pr = ctr1/ctr

    f = 2 * r * pr / (r + pr)

    metrics_dict['Intent-F1-score'] = f
    metrics_dict['Intent-Recall'] = r
    metrics_dict['Intent-Precision'] = pr

    return metrics_dict

@st.cache() 
def letsum(text):
	summary = letsum_test.LetSum(text, 0.3)
	return summary

@st.cache() 
def legal_led(text):
	model = pipeline("summarization", model="nsi319/legal-led-base-16384", tokenizer="nsi319/legal-led-base-16384", framework="pt")
	summary = model(text, min_length=128, max_length=512)
	del model
	return summary[0]['summary_text']

@st.cache() 
def bert(text):
	bert_model = Summarizer('bert-base-uncased')
	bert_summary = ''.join(bert_model(clean_data, ratio=0.3))
	del bert_model
	return bert_summary

@st.cache() 
def graphical(text):
	summary = graphicalModel.get_summary(text,0.3)
	return summary

def give_final_summary(summary):
	text = re.sub(r'[^\w\s.,]', ' ', summary)
	return text


st.title("An Evaluation Framework for Legal Document Summarization")
# markdown in streamlit
st.markdown('''
This demonstration can perform three different tasks:\n
1. Summarize your document using 4 different models, namely:\n
    a. Graphical Model (Saravanan et al., 2006)\n
    b. LetSum (Farzindar et al., 2004)\n
    c. BERT Extractive Summarizer (Devlin et al., 2018)\n
    d. Legal-Longformer Encoder Decoder (Legal-LED) (Beltagy et al., 2020)\n\n
2. Extraction of Intent from the uploaded documents using JointBERT (Chen et al., 2019)\n
3. Evaluation of summary generated by one or more selected from the above models\n 
''')
#st.write("The Error in red will be resolved when you(user) uploads the file.")

file_1 =r"**Example Test File 1 :** " + "https://drive.google.com/file/d/1Lsiswn37SeeZJl6ynBJfV_wpS2kQYlE_/view"
file_2 =r"**Example Test File 2 :** " + "https://drive.google.com/file/d/1S0QsZwXBlG78fA26QMDjkmnU7qDZrYSm/view" 
st.write("")
st.markdown(file_1, unsafe_allow_html=True)
st.markdown(file_2, unsafe_allow_html=True)
st.write("")

#Textbox for text user is entering
st.subheader("Upload the text (.txt) file that you would like to summarize:")
file = st.file_uploader('Please upload a text(.txt) file (containing not more than 2000 words)') #text is stored in this variable
if file is not None:
    raw_data = str(file.read())
    clean_data = re.sub('\n', ' ', raw_data)
    clean_data = re.sub('\n\n', ' ', clean_data)
# this clean data is for summarization model feeding
# raw data preproc to be done later for intent extraction

#st.markdown(clean_data)

st.subheader("Select the Summarization model that you would like to use: (wait for around 1 min after choice selection to get summary)")
choice = st.selectbox("Choose Model:",["Graphical Model","LetSum","BERT Extractive Summarizer","Legal-Longformer Encoder Decoder (Legal-LED)"]) 

start = st.button("Click to start summarization")

if start == True:    
    if choice == "LetSum":
        summary = letsum(clean_data)
        #clean_summary = re.sub(r'[^\w\s]', ' ', summary)
        clean_summary = give_final_summary(summary)
        st.success(clean_summary)
        st.write("The uploaded document has ",str(len(word_tokenize(clean_data)))+ " words.")
        st.write("The summary has ",str(len(word_tokenize(clean_summary)))+ " words.")
        #st.write("The summary has ", str(len(sent_tokenize(summary))) + " sentences")
    elif choice == "Graphical Model":
        summary = graphical(clean_data)
        #clean_summary = re.sub(r'[^\w\s]', ' ', summary)
        clean_summary = give_final_summary(summary)
        st.success(clean_summary)
        st.write("The uploaded document has ",str(len(word_tokenize(clean_data)))+ " words.")
        st.write("The summary has ",str(len(word_tokenize(clean_summary)))+ " words.")
        #st.write("The summary has ", str(len(sent_tokenize(summary))) + " sentences")
    elif choice == "BERT Extractive Summarizer":
        summary = bert(clean_data)
        #text = re.sub(r'\n', ' ', summary)
        #text = re.sub(r'\r', ' ', text)
        #clean_summary = re.sub(r'\t', ' ', text)
        clean_summary = give_final_summary(summary)
        st.success(clean_summary)
        st.write("The uploaded document has ",str(len(word_tokenize(clean_data)))+ " words.")
        st.write("The summary has ",str(len(word_tokenize(clean_summary)))+ " words.")
        #st.write("The summary has ", str(len(sent_tokenize(summary))) + " sentences")
    elif choice == "Legal-Longformer Encoder Decoder (Legal-LED)":
        summary = legal_led(clean_data)
        #text = re.sub(r'\n', ' ', summary)
        #text = re.sub(r'\r', ' ', text)
        #clean_summary = re.sub(r'\t', ' ', text)
        clean_summary = give_final_summary(summary)
        st.success(clean_summary)
        st.write("The uploaded document has ",str(len(word_tokenize(clean_data)))+ " words.")
        st.write("The summary has ",str(len(word_tokenize(clean_summary)))+ " words.")
        #st.write("The summary has ", str(len(sent_tokenize(summary))) + " sentences")



    st.subheader("The following are a set of extracted intent phrases by using JointBERT:")
    st.write("(Please wait for around 2 minutes since the model is being run on CPU)")

    intent_str_file_uploaded = StringIO(file.getvalue().decode("utf-8"))

    lines = []
    with intent_str_file_uploaded as f:
        for line in f:
            line = line.strip()
            words = line.split()
            lines.append(words)

    with st.spinner(text="Please wait while JointBERT is predicting."):
    	lines_list = predict_file(lines)

    final_lines_list = []
    for k in lines_list:
    	if len(word_tokenize(re.sub(r'[^\w\s]', '', k)))>5:
    		final_lines_list.append(k)

    dict_count = {'Land_Dispute':0,'Corruption':0,'Murder':0,'Robbery':0, 'UNK':0}
    f = final_lines_list
    for i in f:
        i_tokens = re.split(' ',i)
        if (len(i_tokens)>=6):
            dict_count[i_tokens[0][1:-1]] +=1



    LD_count = dict_count['Land_Dispute']/(dict_count['Land_Dispute']+dict_count['Robbery']+dict_count['Murder']+dict_count['Corruption']+dict_count['UNK'])
    Corruption_count = dict_count['Corruption']/(dict_count['Land_Dispute']+dict_count['Robbery']+dict_count['Murder']+dict_count['Corruption']+dict_count['UNK'])
    Murder_count = dict_count['Murder']/(dict_count['Land_Dispute']+dict_count['Robbery']+dict_count['Murder']+dict_count['Corruption']+dict_count['UNK'])
    Robbery_count = dict_count['Robbery']/(dict_count['Land_Dispute']+dict_count['Robbery']+dict_count['Murder']+dict_count['Corruption']+dict_count['UNK'])
    unk_count = dict_count['UNK']/(dict_count['Land_Dispute']+dict_count['Robbery']+dict_count['Murder']+dict_count['Corruption']+dict_count['UNK'])

    percent_count_dict = {"Corruption": Corruption_count*100,"Land_Dispute":LD_count*100,"Murder":Murder_count*100,"Robbery":Robbery_count*100,"UNK": unk_count*100}

    k = dict(sorted(dict_count.items(),key=lambda x:x[1],reverse = True))
    file_intent_tag = list(k.keys())[0]

    if file_intent_tag == "UNK":
    	max_var = percent_count_dict['UNK']
    	percent_count_dict['UNK'] = percent_count_dict[list(k.keys())[1]]
    	percent_count_dict[list(k.keys())[1]] = max_var

    percent_df = pd.DataFrame([[Corruption_count*100, LD_count*100, Murder_count*100, Robbery_count*100, unk_count*100]],columns=['Corruption', 'Land_Dispute', 'Murder','Robbery',"UNK"])


    intent_phrase_list = []
    for i in final_lines_list:
        words = re.split(' ',i)

        line_words = []
        if (words[0][1:-1]== file_intent_tag):
            for j in words[2:]:
                splitted = re.split(':',j)
                if (len(splitted)>=2):
                    line_words.append(splitted[0][1:])

        if (len(line_words)>2):
            sentence = ' '.join(x for x in line_words)

            if (sentence[-3:]== '</c'):  
                intent_phrase_list.append(sentence[:-3])
            else:
                intent_phrase_list.append(sentence)

    final_intent_phrases = []

    for i in intent_phrase_list:
        if i not in final_intent_phrases:
            final_intent_phrases.append(i)

    #st.markdown(final_intent_phrases)
    for intent in range(len(final_intent_phrases)):
    	st.write(str(intent+1)+".- "+final_intent_phrases[intent])


    st.subheader("The results of percentages of different intents are shown in the table:")
    st.dataframe(percent_df)

    st.subheader("The following are the reported metrics for the summary and the intent phrases extracted:")


    with st.spinner(text="Please wait while the metrics are being calculated."):
        metrics_dict = calculate_metrics(final_intent_phrases, clean_summary)
        st.json(metrics_dict)
        caching.clear_cache()

