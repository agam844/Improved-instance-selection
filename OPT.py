import pandas as pd
import numpy as np
from openprompt.data_utils import InputExample
from sklearn.metrics import accuracy_score,f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

np.random.seed(42)

file_path = '/train.csv'
file_path1 = 'test.csv'
file_path2 = 'test_labels.csv'
df = pd.read_csv(file_path)
df_test = pd.read_csv(file_path1)
df_label = pd.read_csv(file_path2)

df_label['sum_toxic_categories'] = df_label[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].sum(axis=1)
df_label = df_label[['sum_toxic_categories']]
df_label = df_label[df_label['sum_toxic_categories'] >= 0]
df_label['Toxic'] = (df_label['sum_toxic_categories'] > 0).astype(int)
df_label.drop('sum_toxic_categories', axis=1, inplace=True)

indices_to_keep = df_label.index.tolist()
df_test = df_test.iloc[indices_to_keep]

df_test['toxic'] = df_label['Toxic']

df['toxic'] = (df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].sum(axis=1) > 0 ).astype(int)
df = df[['comment_text', 'toxic']]

df.head()

df_test.head()

df_test.reset_index(drop = True)

tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(df['comment_text'])

num_clusters = 4
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(tfidf_matrix)

df['cluster'] = cluster_labels

toxic_0_df = df[df['toxic'] == 0]
toxic_1_df = df[df['toxic'] == 1]

cluster_0_samples = toxic_0_df[toxic_0_df['cluster'] == 0]
random_sample = cluster_0_samples.sample(n=1, random_state=42)
cluster_0_samples1 = toxic_0_df[toxic_0_df['cluster'] == 1]
random_sample1 = cluster_0_samples1.sample(n=1, random_state=42)
cluster_1_samples = toxic_1_df[toxic_1_df['cluster'] == 2]
random_sample2 = cluster_1_samples.sample(n=1, random_state=42)
cluster_1_samples1 = toxic_1_df[toxic_1_df['cluster'] == 3]
random_sample3 = cluster_1_samples1.sample(n=1, random_state=42)

random_samples_combined = pd.concat([random_sample, random_sample1, random_sample2, random_sample3])
random_samples_combined.reset_index(drop=True, inplace=True)

random_samples_combined

#using Opt-2.7B instead of BERT
from openprompt.plms import load_plm
plm, tokenizer, model_config, WrapperClass = load_plm("opt", "/scratch/2780992k/facebook/opt-2.7b")

df_toxic = df_label[df_label['Toxic'] == 1]
df_non_toxic = df_label[df_label['Toxic'] == 0]

num_samples = 5000
df_toxic_sampled = df_toxic.sample(n=num_samples, random_state=42)
df_non_toxic_sampled = df_non_toxic.sample(n=num_samples, random_state=42)

concatenated_df = pd.concat([df_toxic_sampled, df_non_toxic_sampled])
concatenated_df = concatenated_df.sort_index()

index_list = concatenated_df.index.tolist()
df_test_filtered = df_test.loc[df_test.index.isin(index_list)]

concatenated_df = concatenated_df.reset_index(drop=True)
df_test_filtered = df_test_filtered.reset_index(drop=True)

classes = [
    "not toxic",
    "toxic",
]

count = 0
dataset = []

for idx, row in df_test.iterrows():
  example = InputExample(guid=count, text_a=row['comment_text'])
  dataset.append(example)
  count += 1

da = []
mask = []
count = 0
for idx, row in random_samples_combined.iterrows():
  if row['toxic'] == 1:
    mask.append( 'It is toxic')
  else:
    mask.append("It is not toxic")
  example = InputExample(guid=count, text_a=row['comment_text'])
  da.append(example)
  count += 1

text1 = " "

for i in range(len(mask)):
  da[i].text_a = da[i].text_a.replace('\\', '\\\\')
  da[i].text_a = da[i].text_a.replace('{', '')
  da[i].text_a = da[i].text_a.replace('}', '')
  text1 = da[i].text_a + mask[i] + " " + text1

from openprompt.prompts import ManualTemplate
pTemplate = ManualTemplate(
    # text = text1 + '{"placeholder":"text_a"} It is {"mask"}',
    text = '{"placeholder":"text_a"} It is {"mask"}',
    tokenizer = tokenizer,
)

from openprompt.prompts import ManualVerbalizer
pVerbalizer = ManualVerbalizer(
    classes = classes,
    label_words = {
        "not toxic":['safe','harmless'],
        "toxic": ['obscene', 'threat', 'insult', 'identity hate', 'racism', 'abuse'],
    },
    tokenizer = tokenizer,
)

from openprompt import PromptForClassification
promptModel = PromptForClassification(
    template = pTemplate,
    plm = plm,
    verbalizer = pVerbalizer,
)

from openprompt import PromptDataLoader
data_loader = PromptDataLoader(
    dataset = dataset,
    tokenizer = tokenizer,
    template = pTemplate,
    tokenizer_wrapper_class=WrapperClass,
)

import torch
list1 = []
list2 = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
promptModel.to(device)

promptModel.eval()

with torch.no_grad():
    for batch in data_loader:
        batch = batch.to(device)
        logits = promptModel(batch)
        preds = torch.argmax(logits, dim = -1)
        list1.append(torch.softmax(logits, dim=-1))
        list2.append(preds.item())
        # print(classes[preds])

print("Final performance on the test set (subset 2):")
y_true = concatenated_df['Toxic'].tolist()
y_pred = list2
f1 = f1_score(y_true, y_pred)
acc = accuracy_score(y_true, y_pred)
print(f"F1 Score: {f1}")
print("accuracy: ", acc)

for i in range(len(y_true)):
  if y_true[i] == 1:
    print(i)

for i in range(len(y_pred)):
  if y_pred[i] == 1:
    print(i)