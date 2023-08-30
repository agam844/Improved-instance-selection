import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertModel
from torch import cuda
from openprompt.prompts import ManualTemplate
from openprompt.data_utils import InputExample
from openprompt.prompts import ManualVerbalizer
from openprompt.plms import load_plm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
device = torch.device('cuda' if cuda.is_available() else 'cpu')

#Initialising the csv files
file_path = 'train.csv'
file_path1 = 'test.csv'
file_path2 = 'test_labels.csv'
df = pd.read_csv(file_path)
df_test = pd.read_csv(file_path1)
df_label = pd.read_csv(file_path2)

#converting all the labels into 1 label to make it binary classification problem
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
df_test = df_test.reset_index(drop = True)
df["comment_text"] = df["comment_text"].str.replace("\xa0", " ", regex=False).str.split().str.join(" ")
df["comment_text"] = df["comment_text"].str.lower()


toxic_0_df = df[df['toxic'] == 0]
toxic_1_df = df[df['toxic'] == 1]

random_sample = toxic_0_df.sample(n=44435, random_state=42)
random_samples2 = toxic_1_df.sample(n=5565, random_state=42)
conc = pd.concat([random_sample, random_samples2])
conc = conc.sample(frac=1, random_state=42)
conc = conc.reset_index(drop=True)


toxic_0_df = conc[conc['toxic'] == 0]
toxic_1_df = conc[conc['toxic'] == 1]
random_sample = toxic_0_df.sample(n=2, random_state=42)
random_sample1 = toxic_1_df.sample(n=2, random_state=42)

#Splitting training dataset into T0 and T1
random_samples_combined = pd.concat([random_sample, random_sample1])
random_samples_combined_rest = conc.drop(random_samples_combined.index)
random_samples_combined = random_samples_combined.reset_index(drop=True)
random_samples_combined_rest= random_samples_combined_rest.reset_index(drop=True)

new_df = random_samples_combined

class DistilBERTClass(torch.nn.Module):

    def __init__(self):
        super(DistilBERTClass, self).__init__()
        self.l1 = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(768, 1)

    def forward(self, input_ids, attention_mask, token_type_ids):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.Tanh()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output

def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)

#Custom class for pre-processing before text can be fed into DistilBERT
class BinaryDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_len: int, eval_mode: bool = False):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.text = dataframe.comment_text
        self.eval_mode = eval_mode
        if self.eval_mode is False:
            self.targets = torch.tensor(dataframe.toxic.values, dtype=torch.float32)
        self.max_len = max_len

    def add_from_other_dataset(self, other_dataset, indexes_to_add):
        # Get the data points from the other_dataset based on the indexes
        for i in range(len(indexes_to_add)):
            other_data_id = other_dataset.data.id.iloc[indexes_to_add[i]]
            other_data_com = other_dataset.data.comment_text.iloc[indexes_to_add[i]]
            other_data_tox = other_dataset.data.toxic.iloc[indexes_to_add[i]]
            other_text = other_dataset.text.iloc[indexes_to_add[i]]
            if not self.eval_mode:
                other_targets = other_dataset.targets[indexes_to_add[i]]
                if len(other_targets.shape) == 0:
                    other_targets = other_targets.unsqueeze(0)

            # Append the data points to the current dataset
            new_row = pd.Series({'id': other_data_id, 'comment_text': other_data_com, 'toxic': other_data_tox})
            self.data = self.data.append(new_row, ignore_index=True)
            self.text = pd.concat([self.text, pd.Series(other_text)], ignore_index=True)
            if not self.eval_mode:
                self.targets = torch.cat([self.targets, other_targets], dim=0)

    def remove_by_index(self, indexes_to_remove):
        self.data = self.data.reset_index(drop=True)
        for i in range(len(indexes_to_remove)):
                ind = [indexes_to_remove[i]]
                mask = ~self.data.index.isin(ind)
                self.data = self.data[mask]
                self.text = self.text[mask]
                if not self.eval_mode:
                    self.targets = self.targets[mask]

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text.iloc[index])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )

        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        output = {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
        }

        if self.eval_mode is False:
        #     output['targets'] = torch.tensor(self.targets.iloc[index], dtype=torch.float).unsqueeze(0)
             output['targets'] = torch.tensor(self.targets[index], dtype=torch.float).unsqueeze(0)
        return output

def training(epoch, data_loader):
    model.train()
    for _, data in tqdm(enumerate(data_loader, 0)):
        ids = data['ids'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
        targets = data['targets'].to(device, dtype=torch.float)

        outputs = model(ids, mask, token_type_ids)

        optimizer.zero_grad()
        loss = loss_fn(outputs, targets)
        if _ % 500 == 0:
            print(f'Epoch: {epoch}, Loss:  {loss.item()}')

        loss.backward()
        optimizer.step()


all_test_pred = []

def test(model, data_loader):
    model.eval()
    all_test_pred = []
    preds = []

    with torch.inference_mode():
        for _, data in tqdm(enumerate(data_loader, 0)):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            outputs = model(ids, mask, token_type_ids)
            probas = torch.sigmoid(outputs)
            all_test_pred.append(probas)
        return torch.cat(all_test_pred)

def iter(y_true, y_pred):
    inn = []
    for i in range(len(y_pred)):
        if y_pred[i] != y_true[i]:
            inn.append(i)
    return inn

def iter2(y_true, y_pred):
    inn = []
    for i in range(len(y_pred)):
        if y_pred[i] == y_true[i]:
            inn.append(i)
    return inn

#Training Parameters:
max_len = 512
batch_size = 8
epochs = 1
LR = 1e-05
workers = 2
query_size = 5000
num_iterations = 5
path = '/scratch/2780992k/last1.pth'
tokenizer1 = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', truncation=True, do_lower_case=True)

columns = ['iteration', 'accuracy', 'precision', 'recall', 'F1']
bert_df = pd.DataFrame(columns=columns)

columns = ['iteration', 'accuracy', 'precision', 'recall', 'F1']
opt_df = pd.DataFrame(columns=columns)

print("1. High Confidence and Wrong Labels")
print("2. Low Confidence and Wrong Labels")
print("3. High Confidence and Correct Labels")
print("4. Low Confidence and Correct Labels")
ch = '1'

cou = 0

plm, tokenizer, config, WrapperClass = load_plm("opt", "/scratch/2780992k/facebook")

classes = [
    "not toxic",
    "toxic",
]
count = 0
dataset = []
for idx, row in random_samples_combined_rest.iterrows():
    example = InputExample(guid=count, text_a=row['comment_text'])
    dataset.append(example)
    count += 1

# making the dataset for In context learning
da = []
mask = []
count = 0
for idx, row in new_df.iterrows():
    if row['toxic'] == 1:
        mask.append('It is toxic')
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
    da[i].text_a = da[i].text_a.replace('\n', ' ')
    text1 = da[i].text_a + mask[i] + " " + text1

#initiallising the manual template for the prediction text and tokeniser
pTemplate = ManualTemplate(
    text = text1 +'\n'+  '{"placeholder":"text_a"} It is {"mask"}',
    # text = '{"placeholder":"text_a"} It is {"mask"}',
    tokenizer = tokenizer,
)

#Initialising the Verbaliser
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
    max_seq_length=1024
)

promptModel.to(device)

list1 = []
list2 = []

c = 0

with torch.no_grad():
    for batch in data_loader:
        batch = batch.to(device)
        logits = promptModel(batch)
        preds = torch.argmax(logits, dim = -1)
        list1.append(torch.softmax(logits, dim=-1))
        list2.append(preds.item())
        c = c + 1
        if(c % 5000 == 0):
            print(c)

values_list = []
pr_dict = dict()
for index, tensor in enumerate(list1):
    values_list.append(tensor.flatten().tolist())
    pr_dict[index] = tensor.flatten().tolist()

print("performance on the test set using open-prompt")
y_true = random_samples_combined_rest['toxic'].tolist()
y_pred = list2
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average='weighted')
recall_avg= recall_score(y_true, y_pred, average='weighted')
pres_avg= precision_score(y_true, y_pred, average='weighted')
conf_matrix = confusion_matrix(y_true, y_pred)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Precision avg:", pres_avg)
print("Recall avg:", recall_avg)
print("Confusion Matrix:\n", conf_matrix)

opt_df = opt_df.append(
{"iteration": cou, "accuracy": accuracy, "precision": precision, "recall": recall, 'F1': f1},
ignore_index=True)


#Active learning loop
for iteration in range(num_iterations):
    # Now Using DistillBERT
    training_subset_1_df = random_samples_combined
    training_subset_rest_df = random_samples_combined_rest

    training_subset_1 = BinaryDataset(training_subset_1_df, tokenizer1, max_len)
    training_subset_rest = BinaryDataset(training_subset_rest_df, tokenizer1, max_len)

    training_parameters = {
        'shuffle': True,
        'batch_size': batch_size,
        'num_workers': workers
    }
    testing_parameters = {
        'shuffle': False,
        'batch_size': batch_size,
        'num_workers': workers
    }

    training_loader_subset_1 = DataLoader(training_subset_1, **training_parameters)
    training_loader_subset_rest = DataLoader(training_subset_rest, **testing_parameters)

    model = DistilBERTClass()
    model.to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LR)

    print(f"Active Learning Iteration {iteration + 1}")

    for epoch in range(epochs):
        training(epoch, training_loader_subset_1)

    all_test_pred = test(model, training_loader_subset_rest)

    output_array = all_test_pred.cpu().numpy()
    binary_array = (output_array >= 0.5).astype(int)
    simple_array = binary_array.flatten()

    print("Performance on the test set DistillBERT:")
    y_true = training_subset_rest.data['toxic'].values
    y_pred = simple_array
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_true, y_pred)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("Confusion Matrix:\n", conf_matrix)

    bert_df = bert_df.append(
        {"iteration": cou, "accuracy": accuracy, "precision": precision, "recall": recall, 'F1': f1},
        ignore_index=True)

    inn2 = []
    inn3 = []

    if ch == '1' or ch == '2':
        inn2 = iter(y_true, y_pred)
        inn3 = iter2(y_true, y_pred)
    else:
        inn3 = iter(y_true, y_pred)
        inn2 = iter2(y_true, y_pred)

    # initialling the plm, tokenizer, configeratin and wrapper class

    lam = 0.5

    if len(inn2)!=5000:
        inn2 = inn2 + inn3[:(5000-len(inn2))]

    filtered_list = [all_test_pred[i] for i in inn2]
    my_dict = {key: value for key, value in zip(inn2, filtered_list)}

    combined_dict = dict()

    for element in inn2:

        if y_pred[element] == 0:
            value1 = 1 - my_dict[element]
            value2 = pr_dict[element][0]
        else:
            value1 = my_dict[element]
            value2 = pr_dict[element][1]
        combined_dict[element] = lam * value1 + (1-lam) * value2


    if ch == '1' or ch == '3':
        combined_dict1 = {k: v for k, v in sorted(combined_dict.items(), key=lambda item: item[1], reverse=True)}
    else:
        combined_dict1 = {k: v for k, v in sorted(combined_dict.items(), key=lambda item: item[1])}

    top_dict = {k: combined_dict1[k] for k in list(combined_dict1)[:query_size]}
    top_keys = top_dict.keys()
    keys_list = list(top_keys)

    new_key = 0

    new_dict = dict()

    for key, value in pr_dict.items():
        if key not in keys_list:
            new_dict[new_key] = value
            new_key += 1

    print(new_dict)

    pr_dict = new_dict

    for keys in keys_list:
        new_row = random_samples_combined_rest.loc[keys]
        random_samples_combined = random_samples_combined.append(new_row)
        random_samples_combined_rest = random_samples_combined_rest.drop(keys)

    random_samples_combined = random_samples_combined.reset_index(drop=True)
    random_samples_combined_rest = random_samples_combined_rest.reset_index(drop=True)
    new_df = new_df.reset_index(drop=True)
    bert_df.to_csv("bert.csv", index=False)
    opt_df.to_csv("opt.csv", index=False)
    torch.save(model.state_dict(), path)
    cou = cou + 1

#Final Finetuning and Evaluation
training_subset_1_df = random_samples_combined
training_subset_rest_df = random_samples_combined_rest

training_subset_1 = BinaryDataset(training_subset_1_df, tokenizer1, max_len)
training_subset_rest = BinaryDataset(training_subset_rest_df, tokenizer1, max_len)

training_parameters = {
    'shuffle': True,
    'batch_size': batch_size,
    'num_workers': workers
}
testing_parameters = {
    'shuffle': False,
    'batch_size': batch_size,
    'num_workers': workers
}

training_loader_subset_1 = DataLoader(training_subset_1, **training_parameters)
training_loader_subset_rest = DataLoader(training_subset_rest, **testing_parameters)

model = DistilBERTClass()
model.to(device)
optimizer = torch.optim.Adam(params = model.parameters(), lr=LR)

for epoch in range(epochs):
    training(epoch, training_loader_subset_1)

torch.save(model.state_dict(), path)

test_set = BinaryDataset(df_test, tokenizer1, max_len, eval_mode = True)
test_loader = DataLoader(test_set, **testing_parameters)

all_test_pred= test(model, test_loader)
output_array = all_test_pred.cpu().numpy()
binary_array = (output_array >= 0.5).astype(int)
simple_array = binary_array.flatten()

print("Performance on the test set using BERT")
y_true = test_set.data['toxic'].values
y_pred = simple_array
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
precision1 = precision_score(y_true, y_pred, average='weighted')
recall1 = recall_score(y_true, y_pred, average='weighted')
f11 = f1_score(y_true, y_pred, average='weighted')
conf_matrix = confusion_matrix(y_true, y_pred)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Precision avg:", precision1)
print("Recall avg:", recall1)
print("F1 Score avg:", f11)
print("Confusion Matrix:\n", conf_matrix)

torch.save(model.state_dict(), path)
bert_df = bert_df.append({"iteration": 'test', "accuracy": accuracy,  "precision": precision, "recall": recall, 'F1':f1 }, ignore_index=True)
bert_df.to_csv("bert.csv", index=False)
opt_df.to_csv("opt.csv", index=False)