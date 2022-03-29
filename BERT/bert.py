import time

from joblib.logger import format_time
from sklearn.metrics import classification_report
from torch.nn import BCEWithLogitsLoss
from torch.nn.functional import cross_entropy
from BERT.ckickBERT_Arch import clickBERT_Arch
from Tools.files import readFile, writeFile
from transformers import BertTokenizer, BertModel, AutoModel
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from transformers import AutoModel, BertTokenizerFast

#MAKES AVERAGE OF ALL 14 HIDDEN UNITS
def bertEncoding(db_option):
    array = []
    i = 0
    #with Bert tokenizer or not
    database = readFile('./DataRetrieval/clean_database.json') if (db_option == 1) else readFile('./DataRetrieval/BERT_clean_database.json')
    #Load tokenizer part of the model
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    # Load pre-trained model (weights)
    model = BertModel.from_pretrained('bert-base-cased',
                                      output_hidden_states=True,  # Whether the model returns all hidden-states.
                                      )
    # Put the model in "evaluation" mode, meaning feed-forward operation.
    model.eval()
    for item in database["list"]:
        indexed_tokens = []
        segments_ids = []
        if db_option == 1:
            # PRepare array for tokenization
            marked_text = "[CLS] " + item["title"] + " [SEP]"
            # Tokenixe
            tokenized = tokenizer.tokenize(marked_text)
            # Topken to ids and segmentID
            indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized)
            segments_ids = [1] * len(
                tokenized)  # TODO: HACER UN CONTROL DE LAS ORACIONES PARA VER SI ALGUNA TIENE DOS PARTES
            # Convert inputs to PyTorch tensors
        else:
            indexed_tokens = item["title"]["tokens"]
            segments_ids = item["title"]["positions"]

        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])

        #Get an output from the model
        with torch.no_grad():
            outputs = model(tokens_tensor, segments_tensors)
            hidden_states = outputs[2]
            #Rearrange for easier understanding
            hidden_states = outputs[2]
            token_embeddings = torch.stack(hidden_states, dim=0)
            token_embeddings.size()
            token_embeddings = torch.squeeze(token_embeddings, dim=1)
            token_embeddings = token_embeddings.permute(1, 0, 2)
            #Queda como: [# tokens, # layers, # features]
            # `token_vecs` is a tensor with shape [22 x 768]
            token_vecs = hidden_states[-2][0]
            # Calculate the average of all token vectors.
            sentence_embedding = torch.mean(token_vecs, dim=0)

        array.insert(i,sentence_embedding.tolist()) #TODO: DELETE ARRAY
        database["list"][i]["title"]=sentence_embedding.tolist()
        i += 1

    print("Hem acabat")
    writeFile('./DataRetrieval/encoded_database.json', database)

def bertPreprocessing(database):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    i = 0
    for item in database["list"]:
        newtitle = tokenizer(item["title"])
        database["list"][i]["title"]={"tokens" : newtitle["input_ids"],
                                      "positions": newtitle["token_type_ids"],
                                      "attention_mask": newtitle["attention_mask"]
                                      }
        i+=1
    writeFile("./DataRetrieval/BERT_clean_database.json", database)

def plotTheFitting(history):
    from matplotlib import pyplot as plt
    figure, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

    # Left plot (Accuracy)
    ax1.plot(history.history['accuracy'])
    ax1.plot(history.history['val_accuracy'])
    ax1.title.set_text('Model Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend(['train', 'validation'])

    # Right plot (Loss)
    ax2.plot(history.history['loss'])
    ax2.plot(history.history['val_loss'])
    ax2.title.set_text('Model Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend(['train', 'validation'])

#DEPCRECATED
#MAKES AVERAGE OF LAST 4 HIDDEN UNITS
#TODO: Arreglarque pone NaN en sitios por algún motivo
def bertEmbeddings_bad():
    array = []
    i = 0
    database = readFile('./DataRetrieval/database.json')
    #Load tokenizer part of the model
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    # Load pre-trained model (weights)
    model = BertModel.from_pretrained('bert-base-cased',
                                      output_hidden_states=True,  # Whether the model returns all hidden-states.
                                      )
    # Put the model in "evaluation" mode, meaning feed-forward operation.
    model.eval()
    for item in database["list"]:
        #PRepare array for tokenization
        marked_text = "[CLS] " + item["title"] + " [SEP]"
        #Tokenixe
        tokenized = tokenizer.tokenize(marked_text)
        #Topken to ids and segmentID
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized)
        segments_ids = [1] * len(tokenized) #TODO: HACER UN CONTROL DE LAS ORACIONES PARA VER SI ALGUNA TIENE DOS PARTES
        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])

        #Get an output from the model
        with torch.no_grad():
            outputs = model(tokens_tensor, segments_tensors)
            hidden_states = outputs[2]
            #Rearrange for easier understanding
            hidden_states = outputs[2]
            token_embeddings = torch.stack(hidden_states, dim=0)
            token_embeddings.size()
            token_embeddings = torch.squeeze(token_embeddings, dim=1)
            token_embeddings = token_embeddings.permute(1, 0, 2)
            #Queda como: [# tokens, # layers, # features]
            # `token_vecs` is a tensor with shape [22 x 768]
            token_vecs = hidden_states[-2][0][10:14]
            # Calculate the average of all token vectors.
            sentence_embedding = torch.mean(token_vecs, dim=0)

        temp = sentence_embedding.tolist()

        array.insert(i,sentence_embedding.tolist()) #TODO: DELETE ARRAY
        database["list"][i]["title"]=sentence_embedding.tolist()
        i += 1

    print("Hem acabat")
    writeFile('./DataRetrieval/encoded_database.json', database)

def adjustSizeForTensors(array,val):
    # Miramos de poner todas las entrada de igual longitud
    # Buscamos la longitud máxima
    maxTitle = 0
    maxDesc = 0
    i=0
    for item in array:
        try:
            length = len(item)
            #print("L-"+str(length)+"\n")
            if length > maxTitle:
                maxTitle = length
        except:
            print("ERROR-->" + str(i))
        i += 1

    # Reajustamos según longitud máxima
    j=0
    for item in array:
        if len(item) < maxTitle:
            for i in range(len(item), maxTitle):
                item.insert(i, val)
            array[j]=item
        j+=1
    return array

def buildDataset(database, batch_size):
    titles = []
    ratings = []
    for item in database["list"]:
        titles.append(item["title"])
        ratings.append(item["rating"])

    #Split the database
    train_text, temp_text, train_labels, temp_labels = train_test_split(titles, ratings,
                                                                        random_state=2018,
                                                                        test_size=0.3)#stratify=ratings #FIXME: EL STRATIFY DA ERROR. MIRAR POR QUÉ Y CÓMO AFECTA.
    val_text, test_text, val_labels, test_labels = train_test_split(temp_text, temp_labels,
                                                                    random_state=2018,
                                                                    test_size=0.5) #,stratify=temp_labels #FIXME: EL STRATIFY DA ERROR. MIRAR POR QUÉ Y CÓMO AFECTA.
    from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
    train_seq = []
    train_mask = []
    train_y = []
    val_seq = []
    val_mask = []
    val_y = []
    test_seq = []
    test_mask = []
    test_y = []
    #Separamos para montar luego el tensorDataset de TRAIN
    train_y = train_labels
    for item in train_text:
      train_seq.append(item["tokens"])
      train_mask.append(item["attention_mask"])
    #Separamos para montar luego el tensorDataset de VALIDATION
    val_y = val_labels
    for item in val_text:
        val_seq.append(item["tokens"])
        val_mask.append(item["attention_mask"])
    test_y = test_labels
    for item in test_text:
        test_seq.append(item["tokens"])
        test_mask.append(item["attention_mask"])
    #TODO: PREPROCESAR Y PONER EN CADA TENSOR TODOS LOS ARRAYS AL MISMO TAMAÑO

    #Convertimos el train to tensors
    train_seq=  torch.tensor(adjustSizeForTensors(train_seq,0))
    train_mask= torch.tensor(adjustSizeForTensors(train_mask,1))
    train_y=    torch.tensor(train_y)
    #Convertimos el VAL to tensors
    val_seq =   torch.tensor(adjustSizeForTensors(val_seq,0))
    val_mask =  torch.tensor(adjustSizeForTensors(val_mask,1))
    val_y =     torch.tensor(val_y)

    test_seq = torch.tensor(adjustSizeForTensors(test_seq,0))
    test_mask = torch.tensor(adjustSizeForTensors(test_mask,0))
    test_y = torch.tensor(test_labels)
    # wrap tensors
    train_data = TensorDataset(train_seq, train_mask, train_y)
    # sampler for sampling the data during training
    train_sampler = RandomSampler(train_data)
    # dataLoader for train set
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    # wrap tensors
    val_data = TensorDataset(val_seq, val_mask, val_y)
    # sampler for sampling the data during training
    val_sampler = SequentialSampler(val_data)
    # dataLoader for validation set
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)

    for batch in train_dataloader:
        print(batch[0][0])
        print(batch[0].shape)
        print(batch[1][0])
        print(batch[1].shape)
        print(batch[2])
        print(batch[2].shape)
        break

    return train_data, train_sampler, train_dataloader, val_data, val_sampler, val_dataloader, train_labels, test_seq, test_mask, test_y

def expandModel(bert, device):
    #Bloqueamos los pesos de las capas preentrenadas.
    for param in bert.parameters():
        param.requires_grad = False

    # pass the pre-trained BERT to our define architecture
    model = clickBERT_Arch(bert)
    # push the model to GPU
    model = model.to(device)
    return model

def getOptimizer(model):
    #We go to make the optimizer
    # optimizer from hugging face transformers
    from transformers import AdamW
    # define the optimizer
    optimizer = AdamW(model.parameters(),
                      lr=1e-5)  # learning rate
    return optimizer

def fineTuningBert(dataset):
    batch_size = 32
    #Load database and device
    database = readFile(dataset)
    device = torch.device("cuda")
    #device = torch.device("cpu")
    # import BERT-base pretrained model
    bert = AutoModel.from_pretrained('bert-base-cased')

    train_data,\
    train_sampler,\
    train_dataloader,\
    val_data,\
    val_sampler,\
    val_dataloader,\
    train_labels,\
    test_seq,\
    test_mask,\
    test_y    = buildDataset(database, batch_size)
    model = expandModel(bert, device)
    optimizer = getOptimizer(model)
    from sklearn.utils.class_weight import compute_class_weight
    # compute the class weights
    class_weights = compute_class_weight(class_weight ='balanced',
                                         classes = np.unique(train_labels),
                                         y =  train_labels)
    print("Class Weights:", class_weights)
    # converting list of class weights to a tensor
    weights = torch.tensor(class_weights, dtype=torch.float)
    # push to GPU
    weights = weights.to(device)
    # define the loss function
    #cross_entropy = nn.NLLLoss(weight=weights)
    global cross_entropy
    cross_entropy= nn.NLLLoss(weight=weights)
    # number of training epochs
    epochs = 100


    # set initial loss to infinite
    best_valid_loss = float('inf')

    # empty lists to store training and validation loss of each epoch
    train_losses = []
    valid_losses = []

    # for each epoch
    for epoch in range(epochs):

        print('\n Epoch {:} / {:}'.format(epoch + 1, epochs))

        # train model
        train_loss, _ = train(model, train_dataloader, device, optimizer)

        # evaluate model
        valid_loss, _ = evaluate(model, val_dataloader, device)

        # save the best model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), './BERT/saved_weights.pt')

        # append training and validation loss
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        print(f'\nTraining Loss: {train_loss:.3f}')
        print(f'Validation Loss: {valid_loss:.3f}')
    #avg_loss, total_preds =train(model, train_dataloader, device, optimizer)
    #avg_loss, total_preds =evaluate(model, val_dataloader,device)

    test(model,test_seq, device, test_mask, test_y)

def train(model, train_dataloader, device, optimizer):
    model.train()
    total_loss, total_accuracy = 0, 0

    # empty list to save model predictions
    total_preds = []

    # iterate over batches
    for step, batch in enumerate(train_dataloader):

        # progress update after every 50 batches.
        if step % 50 == 0 and not step == 0:
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(train_dataloader)))

        # push the batch to gpu
        batch = [r.to(device) for r in batch]
        sent_id, mask, labels = batch

        # clear previously calculated gradients
        model.zero_grad()

        # get model predictions for the current batch
        preds = model(sent_id, mask)
        # compute the loss between actual and predicted values
        loss = cross_entropy(preds, labels)


        # add on to the total loss
        total_loss = total_loss + loss.item()

        # backward pass to calculate the gradients
        loss.backward()

        # clip the the gradients to 1.0. It helps in preventing the exploding gradient problem
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # update parameters
        optimizer.step()

        # model predictions are stored on GPU. So, push it to CPU
        preds = preds.detach().cpu().numpy()

        # append the model predictions
        total_preds.append(preds)

    # compute the training loss of the epoch
    avg_loss = total_loss / len(train_dataloader)

    # predictions are in the form of (no. of batches, size of batch, no. of classes).
    # reshape the predictions in form of (number of samples, no. of classes)
    total_preds = np.concatenate(total_preds, axis=0)

    # returns the loss and predictions
    return avg_loss, total_preds

def evaluate(model, val_dataloader,device):
    print("\nEvaluating...")

    # deactivate dropout layers
    model.eval()

    total_loss, total_accuracy = 0, 0

    # empty list to save the model predictions
    total_preds = []

    # iterate over batches
    for step, batch in enumerate(val_dataloader):

        # Progress update every 50 batches.
        t0 = time.time()
        if step % 50 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            elapsed = format_time(time.time() - t0)

            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(val_dataloader)))

        # push the batch to gpu
        batch = [t.to(device) for t in batch]

        sent_id, mask, labels = batch

        # deactivate autograd
        with torch.no_grad():

            # model predictions
            preds = model(sent_id, mask)
            # compute the validation loss between actual and predicted values
            #loss = cross_entropy(preds, labels)
            loss = cross_entropy(preds, labels)

            total_loss = total_loss + loss.item()

            preds = preds.detach().cpu().numpy()

            total_preds.append(preds)

    # compute the validation loss of the epoch
    avg_loss = total_loss / len(val_dataloader)

    # reshape the predictions in form of (number of samples, no. of classes)
    total_preds = np.concatenate(total_preds, axis=0)

    return avg_loss, total_preds

def test(model,test_seq, device, test_mask, test_y):
    # load weights of best model
    path = './BERT/saved_weights.pt'
    model.load_state_dict(torch.load(path))
    with torch.no_grad():
        preds = model(test_seq.to(device), test_mask.to(device))
        preds = preds.detach().cpu().numpy()
        preds = np.argmax(preds, axis=1)
        print(classification_report(test_y, preds))