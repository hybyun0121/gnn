from dgl.data import FB15k237Dataset
import os
import pickle

def main():
    file_path = os.getcwd()
    data_path = os.path.join(file_path, 'data/FB15k-237')
    with open(f'{data_path}/train_triples_2i.pkl', 'rb') as handle:
        train_triples = pickle.load(handle)
    with open(f'{data_path}/train_ans_2i.pkl', 'rb') as handle:
        train_ans = pickle.load(handle)
    # print(type(train_triples))
    # print(len(train_triples))
    print(list(train_ans.items())[0])


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
