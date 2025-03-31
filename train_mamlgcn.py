import torch.optim
from torch.utils.data import DataLoader, Dataset
import warnings
from utils_syn import *
import pickle
from mamlgcn import *
import data_syn
torch.set_num_threads(int(2))
warnings.filterwarnings('ignore')


def train(train_batches, data_train, model, device):
    best_loss = 1
    best_r2 = 0
    total_preds_train = torch.Tensor().to(device)
    total_labels_train = torch.Tensor().to(device)
    for step in range(1, train_batches + 1):

        model.train()
        # do training
        x_support_set, x_target = data_train.get_train_batch(augment=False)  # [50,50,4] [50,40,4]
        b, spc, t = np.shape(x_support_set)
        support_set_samples_ = x_support_set.reshape(b, spc, t)  # [50,50,4]
        bt, spct, tt = np.shape(x_target)
        target_samples_ = x_target.reshape(bt, spct, tt)  # [50,40,4]
        support_target_samples = np.concatenate([support_set_samples_, target_samples_], axis=1)  # [50,90,4]
        b, s, t = np.shape(support_target_samples)
        support_target_samples = support_target_samples.reshape(b * s, t)  # [4500,4]
        Num_samples = len(support_target_samples)
        print("data over", Num_samples)
        train_data = TestbedDataset(support_target_samples, drug_features, cell_features, codes)
        train_data1 = TestbedDataset1(support_target_samples, drug_features, cell_features, codes)
        train_loader = DataLoader(train_data, batch_size=Num_samples, shuffle=False)
        train_loader1 = DataLoader(train_data1, batch_size=Num_samples, shuffle=False)
        for batch_idx, data in enumerate(train_loader):
            for batch_idx1, data1 in enumerate(train_loader1):
                if batch_idx1 == batch_idx:
                    data = data.to(device)
                    data1 = data1.to(device)
                    query_output, query_target = model.run_batch(data, data1, batch_size, True)
                    total_preds_train = torch.cat((total_preds_train, query_output), 0)
                    total_labels_train = torch.cat((total_labels_train, query_target), 0)

        if step % 50 == 0:
            print(total_labels_train.size())
            train_total_preds = total_preds_train.cpu().detach().numpy().flatten()
            train_total_labels = total_labels_train.cpu().detach().numpy().flatten()
            total_c_spearman = spearman(train_total_labels, train_total_preds)
            total_c_loss = mse(train_total_labels, train_total_preds)
            total_c_rmse = rmse(train_total_labels, train_total_preds)
            total_c_pearson = pearson(train_total_labels, train_total_preds)
            total_c_r2 = r2_score(train_total_labels, train_total_preds)

            print()
            print('=' * 50)
            print("train Epoch: {} --- Meta train Loss: {:4.4f}".format(step, total_c_loss), total_c_rmse,
                  total_c_spearman, total_c_pearson, total_c_r2)
            print('=' * 50)
            save_statistics(experiment_nameT, [step, total_c_loss, total_c_rmse, total_c_spearman,
                                               total_c_pearson, total_c_r2])
            total_preds_train = torch.Tensor().to(device)
            total_labels_train = torch.Tensor().to(device)

        if step % 100 == 0:  # val model
            total_preds = []  # 使用列表来存储预测值
            total_labels = []  # 使用列表来存储真实值
            for val_step in range(total_val_batches):
                x_support_set, x_target = data_train.get_test_batch(augment=False)
                b, spc, t = np.shape(x_support_set)
                support_set_samples_ = x_support_set.reshape(b, spc, t)
                bt, spct, tt = np.shape(x_target)
                target_sample_ = x_target.reshape(bt, spct, tt)
                support_target_samples = np.concatenate([support_set_samples_, target_sample_], axis=1)
                b, s, t = np.shape(support_target_samples)
                support_target_samples = support_target_samples.reshape(b * s, t)
                Num_samples = len(support_target_samples)
                print(Num_samples)
                val_data = TestbedDataset(support_target_samples, drug_features, cell_features, codes)
                val_data1 = TestbedDataset1(support_target_samples, drug_features, cell_features, codes)
                val_loader = DataLoader(val_data, batch_size=Num_samples, shuffle=False)
                val_loader1 = DataLoader(val_data1, batch_size=Num_samples, shuffle=False)
                # val result
                for batch_idx, data in enumerate(val_loader):
                    for batch_idx1, data1 in enumerate(val_loader1):
                        if batch_idx1 == batch_idx:
                            data = data.to(device)
                            data1 = data1.to(device)
                            val_output, val_target = model.test_batch(data, data1, b, False)
                            # 将预测和真实值添加到列表中
                            total_preds.append(val_output.cpu().detach().numpy())
                            total_labels.append(val_target.cpu().detach().numpy())

            total_preds = np.concatenate(total_preds, axis=0)
            total_labels = np.concatenate(total_labels, axis=0)
            print(total_preds.size)  # [40*20*20]
            total_c_spearman = spearman(total_labels, total_preds)
            total_c_loss = mse(total_labels, total_preds)
            total_c_rmse = rmse(total_labels, total_preds)
            total_c_pearson = pearson(total_labels, total_preds)
            total_c_r2 = r2_score(total_labels, total_preds)

            # save checkpoint
            if total_c_loss < best_loss or total_c_r2 > best_r2:
                best_loss = total_c_loss
                best_r2 = total_c_r2
                model_name = '%dk_%4.4f_model' % (step, total_c_loss)
                # defined model name
                state = {'step': step, 'state_dict': model.state_dict()}
                if not os.path.exists('saved_model/saved_mamlgcn/'):
                    os.makedirs('saved_model/saved_mamlgcn/', exist_ok=False)
                save_path = "saved_model/saved_mamlgcn/{}_{}.pth".format(model_name, step)
                torch.save(state, save_path)

            model.train()

            print()
            print('=' * 50)  #
            print("Validation Epoch: {} --- Meta val Loss: {:4.4f}".format(step, total_c_loss), total_c_rmse,
                  total_c_spearman, total_c_pearson, total_c_r2)
            print('=' * 50)
            print()
            print('Saving checkpoint %s...' % (experiment_nameTE))
            save_statistics(experiment_nameTE, [step, total_c_loss, total_c_rmse, total_c_spearman, total_c_pearson, total_c_r2])


if __name__ == '__main__':
    # writer = SummaryWriter('./Result_ful_syn_rest')
    fpath = 'data/sample_features/'
    codes = pickle.load(open(fpath + 'codes_cell.p', 'rb'))
    drug_features = pickle.load(open(fpath + 'drug_feature_cell.p', 'rb'))
    cell_features = pickle.load(open(fpath + 'cell_feature_900.p', 'rb'))

    cuda_name = 'cuda:1'

    lr = 0.001
    total_epochs = 40000
    batch_size = 50
    samples_support = 5
    samples_query = 40
    total_val_batches = 30
    embed = 128

    method = 'train 50-40 ,21 cell ,layer,dim=1,inner learning rate = 0.1,dim=128'
    device = cuda_name if torch.cuda.is_available() else "cpu"

    logs_path = 'one_shot_outputs/'
    experiment_nameT = f'mamlgcn_shot_train_{samples_query}qs_{samples_support}ss_{lr}'
    logs = "{}way{}shot , with {} tasks, test_batch is{},method is {} ".format(samples_support,
                                                                                        samples_query, batch_size,
                                                                                        total_val_batches,
                                                                                        method)
    save_statistics(experiment_nameT, ["Experimental details: {}".format(logs)])
    save_statistics(experiment_nameT, ["epoch", "train_c_loss", "train_c_rmse",
                                       "train_c_spearman", "train_c_pearson", "train_c_r2"])
    experiment_nameTE = f'mamlgcn_shot_val_{samples_query}qs_{samples_support}ss_{lr}'
    save_statistics(experiment_nameTE, ["Experimental details: {}".format(logs)])
    save_statistics(experiment_nameTE, ["epoch", "val_c_loss", "val_c_rmse",
                                        "val_c_spearman", "val_c_pearson", "val_c_r2"])

    save_model_name = 'maml_shot_' + str(samples_query) + 'qs_' + str(samples_support) + 'ss' + str(
        batch_size)

    print(device)
    mini = data_syn.MiniCellDataSet(batch_size=batch_size, samples_support=samples_support, samples_query=samples_query)
    HyperSynergy_model = HyperSynergy(num_support=samples_support, num_query=samples_query).to(device)
    print('-------------------------------you are you, i am me---------------------------')
    sum_ = 0
    for name, param in HyperSynergy_model.named_parameters():
        mul = 1
        for size_ in param.shape:
            mul *=size_
        sum_ +=mul
        print('%14s: %s' % (name, param.shape))
    print('parameters number: ', sum_)
    print('------------------------------you are you, i am me !------------------------------')

    model_path = 'pretrain_representation_model/pretrain_representation_few_zero_setting.model'

    pretrained_dict = torch.load(model_path, map_location=device)
    # read MMN's params
    model_dict = HyperSynergy_model.extractor.state_dict()
    # read same params in two model
    state_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
    # update
    model_dict.update(state_dict)
    # load part of model params
    HyperSynergy_model.extractor.load_state_dict(model_dict)
    # freeze
    for p in HyperSynergy_model.extractor.parameters():

        p.requires_grad = False
    # Train
    print("-------------------begin train ----------------------")
    train(total_epochs, mini, HyperSynergy_model, device)
