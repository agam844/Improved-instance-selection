import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertModel
from torch import cuda
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
device = torch.device('cuda' if cuda.is_available() else 'cpu')

#Training Dataset
file_path = 'train.csv'
df = pd.read_csv(file_path)

#Pre-processing
df['toxic'] = df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].max(axis=1)
df.drop(columns=['severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'], inplace=True)

df["comment_text"] = df["comment_text"].str.lower()
df["comment_text"] = df["comment_text"].str.replace("\xa0", " ", regex=False).str.split().str.join(" ")

MAX_LEN = 512
TRAIN_BATCH_SIZE = 32
EPOCHS = 1
LEARNING_RATE = 1e-05
NUM_WORKERS = 2
path = '/scratch/2780992k/WL_HC2.pth'

#Custom class for pre-processing before text can be fed into DistilBERT
class MultiLabelDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_len: int, eval_mode: bool = False):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.text = dataframe.comment_text
        self.eval_mode = eval_mode
        if self.eval_mode is False:
            #self.targets = self.data.toxic
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

QUERY_SIZE = 5000
NUM_AL_ITERATIONS = 5

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', truncation=True, do_lower_case=True)
training_set = MultiLabelDataset(df, tokenizer, MAX_LEN)

num_samples_subset_1 = 50000

#Dividing the training data into T0 and T1
training_subset_1_df = df.sample(n=num_samples_subset_1, random_state=42)
training_subset_1_df.reset_index(drop=True, inplace=True)
training_subset_rest_df = df.drop(training_subset_1_df.index)
training_subset_rest_df.reset_index(drop=True, inplace=True)

training_subset_1 = MultiLabelDataset(training_subset_1_df, tokenizer, MAX_LEN)
training_subset_rest = MultiLabelDataset(training_subset_rest_df, tokenizer, MAX_LEN)

#A new dataframe to store the results
columns = ['iteration', 'accuracy', 'precision', 'recall', 'f1', 'precision avg', 'recall avg', 'f1 avg']
WLHC_df = pd.DataFrame(columns=columns)


# Create data loaders for each subset
train_params_subset_1 = {
    'batch_size': TRAIN_BATCH_SIZE,
    'shuffle': True,
    'num_workers': NUM_WORKERS
}
train_params_subset_rest = {
    'batch_size': TRAIN_BATCH_SIZE,
    'shuffle': False,
    'num_workers': NUM_WORKERS
}

cou = 0

#method used to train the DIstilBERT model
def train(epoch, data_loader):
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

training_loader_subset_1 = DataLoader(training_subset_1, **train_params_subset_1)
training_loader_subset_rest = DataLoader(training_subset_rest, **train_params_subset_rest)

#Custom class to create a neural network
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

#initialising the model
model = DistilBERTClass()
model.to(device)
optimizer = torch.optim.Adam(params = model.parameters(), lr=LEARNING_RATE)

all_test_pred = []

#method to test the model
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

#Active learning loop
for iteration in range(5):

    model = DistilBERTClass()
    model.to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

    print(f"Active Learning Iteration {iteration + 1}")

    for epoch in range(EPOCHS):
        train(epoch, training_loader_subset_1)

    all_test_pred = test(model, training_loader_subset_rest)
    output_array = all_test_pred.cpu().numpy()
    binary_array = (output_array >= 0.5).astype(int)
    simple_array = binary_array.flatten()

    print("Performance on the test set (subset 2):")
    y_true = training_subset_rest.data['toxic'].values
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

    WLHC_df = WLHC_df.append(
        {"iteration": cou, "accuracy": accuracy, "precision": precision, "recall": recall, 'f1': f1,
         'precision avg': precision1, 'recall avg': recall1, 'f1 avg': f11 },
        ignore_index=True
    )

    cou = cou + 1

    inn = []
    for i in range(len(y_pred)):
        if y_pred[i]!=y_true[i]:
            inn.append(i)

    filtered_list2 = []
    if len(inn) != 5000:
        inn2 = []
        for i in range(len(y_pred)):
            if y_pred[i] == y_true[i]:
                inn2.append(i)

        for index in inn2:
            if simple_array[index] == 0:
                filtered_list2.append(1-all_test_pred)
            else:
                filtered_list2.append(all_test_pred)
        my_dict2 = {key: value for key, value in zip(inn2, filtered_list2)}
        sorted_dict2 = {k: v for k, v in sorted(my_dict2.items(), key=lambda item: item[1])}
        # sorted_dict2 = {k: v for k, v in sorted(my_dict2.items(), key=lambda item: item[1], reverse=True)}

    filtered_list = []

    for index in inn:
        if simple_array[index] == 0:
            filtered_list.append(1 - all_test_pred)
        else:
            filtered_list.append(all_test_pred)

    my_dict = {key: value for key, value in zip(inn, filtered_list)}


    # sorted_dict = {k: v for k, v in sorted(my_dict.items(), key=lambda item: item[1])}
    sorted_dict = {k: v for k, v in sorted(my_dict.items(), key=lambda item: item[1], reverse=True)}
    top_dict = {k: sorted_dict[k] for k in list(sorted_dict)[:QUERY_SIZE]}
    top_keys = top_dict.keys()
    if len(inn) != 5000:
        top_dict2 = {k: sorted_dict2[k] for k in list(sorted_dict2)[:(QUERY_SIZE - len(inn))]}
        top_keys2 = top_dict2.keys()
        keys_list1 = list(top_keys)
        keys_list2 = list(top_keys2)
        keys_list = keys_list1 + keys_list2
    else:
        keys_list = list(top_keys)

    training_subset_1.add_from_other_dataset(training_subset_rest, keys_list)
    training_subset_rest.remove_by_index(keys_list)
    training_subset_rest.data = training_subset_rest.data.reset_index(drop=True)
    training_subset_1.data = training_subset_1.data.reset_index(drop=True)
    print('training subert ',len(training_subset_1))
    print('rest subert ', len(training_subset_rest))
    training_loader_subset_1 = DataLoader(training_subset_1, **train_params_subset_1)
    training_loader_subset_rest = DataLoader(training_subset_rest, **train_params_subset_rest)

#final finetuning
for epoch in range(EPOCHS):
    train(epoch, training_loader_subset_1)


#initialise and pre-process the test dataset
df_test = pd.read_csv('test.csv')
df_label = pd.read_csv('test_labels.csv')

df_label['sum_toxic_categories'] = df_label[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].sum(axis=1)
df_label = df_label[['sum_toxic_categories']]
df_label = df_label[df_label['sum_toxic_categories'] >= 0]
df_label['Toxic'] = (df_label['sum_toxic_categories'] > 0).astype(int)
df_label.drop('sum_toxic_categories', axis=1, inplace=True)
indices_to_keep = df_label.index.tolist()
df_test = df_test.iloc[indices_to_keep]
df_test['toxic'] = df_label['Toxic']

test_set = MultiLabelDataset(df_test, tokenizer, MAX_LEN, eval_mode = True)
testing_params = {'batch_size': TRAIN_BATCH_SIZE,
               'shuffle': False,
               'num_workers': 2
                }
test_loader = DataLoader(test_set, **testing_params)

#testing one the test dataset
all_test_pred = test(model, test_loader)
output_array = all_test_pred.cpu().numpy()
binary_array = (output_array >= 0.5).astype(int)
simple_array = binary_array.flatten()

print("Performance on the test set (subset 2):")
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

WLHC_df = WLHC_df.append(
    {"iteration": cou, "accuracy": accuracy, "precision": precision, "recall": recall, 'f1': f1,
    'precision avg': precision1, 'recall avg': recall1, 'f1 avg': f11 },
    ignore_index=True
)

#saving the results CSV file
WLHC_df.to_csv("WL_HC1.csv", index=False)