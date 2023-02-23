import sys
import numpy as np
import conlleval

from common_hf import encode, label_encode, write_result
from common_hf import argument_parser
from common_hf import read_conll, process_sentences, get_labels
from common_hf import get_predictions, get_predictions2

from transformers import RobertaTokenizer, RobertaForTokenClassification, RobertaConfig
from transformers import BertConfig, BertForTokenClassification
from transformers import BertTokenizer, MegatronBertForTokenClassification, MegatronBertConfig
from transformers import Trainer, TrainingArguments

import scipy
import torch
import math

class NERDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def main(argv):

    argparser = argument_parser()
    args = argparser.parse_args(argv[1:])
    seq_len = args.max_seq_length    # abbreviation
    train_words, train_tags = read_conll(args.train_data)
    test_words, test_tags = read_conll(args.test_data)

    results = []
    print(args.ner_model_dir)
    label_list = get_labels(train_tags)
    tag_map = { l: i for i, l in enumerate(label_list) }
    inv_tag_map = { v: k for k, v in tag_map.items() }

    if 'mega' in args.model_name.lower():
        config = MegatronBertConfig.from_pretrained(args.model_name)
        c_dict = config.to_dict()
        c_dict["id2label"] = inv_tag_map
        c_dict["label2id"] = tag_map
        c_dict["num_labels"] = len(label_list)
        config.update(c_dict)
        tokenizer = BertTokenizer.from_pretrained(args.model_name,config=config,do_lower_case = False)
        model = MegatronBertForTokenClassification.from_pretrained(args.model_name,config=config)

    elif 'roberta' in args.model_name.lower():
        config = RobertaConfig.from_pretrained(args.model_name)
        c_dict = config.to_dict()
        c_dict["id2label"] = inv_tag_map
        c_dict["label2id"] = tag_map
        c_dict["num_labels"] = len(label_list)
        config.update(c_dict)
        tokenizer = RobertaTokenizer.from_pretrained(args.model_name,config=config,do_lower_case = False)
        model = RobertaForTokenClassification.from_pretrained(args.model_name,config=config)

    else:
        config = BertConfig.from_pretrained(args.model_name,cache_dir=args.cache_dir)
        c_dict = config.to_dict()
        c_dict["id2label"] = inv_tag_map
        c_dict["label2id"] = tag_map
        c_dict["num_labels"] = len(label_list)
        config.update(c_dict)
        tokenizer = BertTokenizer.from_pretrained(args.model_name,config=config, do_lower_case = False)
        model = BertForTokenClassification.from_pretrained(args.model_name,config=config)
    

    train_data = process_sentences(train_words, train_tags, tokenizer, seq_len, args.predict_position)
    test_data = process_sentences(test_words, test_tags, tokenizer, seq_len, args.predict_position)
    train_x, _ = encode(train_data.combined_tokens, tokenizer, seq_len)
    test_x, _ = encode(test_data.combined_tokens, tokenizer, seq_len)
    train_y, train_weights = label_encode(train_data.combined_labels, tag_map, seq_len)
    test_y, test_weights = label_encode(test_data.combined_labels, tag_map, seq_len)

    train_encodings = {"input_ids" : torch.from_numpy(train_x),
             "token_type_ids" : torch.from_numpy(np.zeros(train_x.shape, dtype=int)),
             "attention_mask" : torch.from_numpy(train_weights)}

    test_encodings = {"input_ids" : torch.from_numpy(test_x),
            "token_type_ids" : torch.from_numpy(np.zeros(test_x.shape, dtype=int)),
            "attention_mask" : torch.from_numpy(test_weights)}
   
    train_dataset = NERDataset(train_encodings, np.squeeze(train_y).tolist())
    test_dataset = NERDataset(test_encodings, np.squeeze(test_y).tolist())

    if args.batch_size > 8:
        max_batch_size = 8  #without running to OOM on hardware with larger models
        accumulation = math.ceil(args.batch_size/max_batch_size)
    else:
        max_batch_size = args.batch_size
        accumulation = 1

    training_args = TrainingArguments(
        output_dir="output",          # output directory
        num_train_epochs=args.num_train_epochs,              # total number of training epochs
        per_device_train_batch_size=max_batch_size,  # batch size per device during training
        per_device_eval_batch_size=max_batch_size,   # batch size for evaluation
	    gradient_accumulation_steps = accumulation, 	
        learning_rate=args.learning_rate,
        adam_epsilon=1e-6,
        max_grad_norm=1.0,
        warmup_ratio=args.warmup_proportion,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir='./hf_logs',            # directory for storing logs
        logging_steps=200,
	    save_strategy='no'
    )


    trainer = Trainer(
        model=model,                         # 
	    args=training_args,        
	    train_dataset=train_dataset,         # training dataset
        eval_dataset=test_dataset             # evaluation dataset
    )

    trainer.train()

    if args.ner_model_dir is not None:
        trainer.save_model(args.ner_model_dir)
        tokenizer.save_vocabulary(args.ner_model_dir)

    
    model_out = trainer.predict(test_dataset)
    probs = scipy.special.softmax(model_out.predictions, axis=-1)
    preds = np.argmax(probs, axis=-1)

    results = []
    m_names = []


    # First tag then vote
    pr_ensemble, pr_test_first = get_predictions(preds, test_data.tokens, test_data.sentence_numbers)
    # Accumulate probabilities, then vote
    prob_ensemble, _ = get_predictions2(probs, test_data.tokens, test_data.sentence_numbers)
    ens = [pr_ensemble, prob_ensemble, pr_test_first]    
    method_names = ['CMV','CMVP','F']

    for i, ensem in enumerate(ens):
        ensemble = []
        for j,pred in enumerate(ensem):
            ensemble.append([inv_tag_map[t] for t in pred])
        output_file = "output/{}-{}.tsv".format(args.output_file, method_names[i])
        lines_ensemble, _ = write_result(
                output_file, test_data.words, test_data.lengths,
                test_data.tokens, test_data.labels, ensemble, tokenizer)
        print("Model trained: ", args.model_name)
        print("Seq-len: ", args.max_seq_length)
        print("Learning rate: ", args.learning_rate)
        print("Batch Size: ", args.batch_size)
        print("Epochs: ", args.num_train_epochs)
        print("Training data: ", args.train_data)
        print("Testing data: ", args.test_data)
        print("")
        print("Results with {}".format(method_names[i]))
        c = conlleval.evaluate(lines_ensemble)
        print("")
        conlleval.report(c)
        results.append([conlleval.metrics(c)[0].prec, conlleval.metrics(c)[0].rec, conlleval.metrics(c)[0].fscore])
        m_names.extend(method_names)


    result_file = "results/results-{}.csv".format(args.output_file)
    with open(result_file, 'w+') as f:
        for i, line in enumerate(results):
            params = "{},{},{},{},{},{},{},{},{},{}".format(args.output_file,
                                            args.max_seq_length, 
                                            args.model_name, 
                                            args.num_train_epochs, 
                                            args.learning_rate,
                                            args.batch_size,
                                            args.predict_position,
                                            args.train_data,
                                            args.test_data,
                                            m_names[i])
            f.write(params)
            for item in line:
                f.write(",{}".format(item))
            f.write('\n') 

    for i in results:
        print(i)
    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv))
