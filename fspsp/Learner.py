from DenseEDModel import DenseEDNet
from VggED_base import VggEDBase
from dataloader import Mydataset
from torch import tensor
from base_network import BaseNetwork
from utils import *
from torch.utils import data
import torch
import logging
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
import random
import math
from model import *
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_task_seq(salmap_path, meta_batch):
    # img_task_path = [os.path.join(img_path, task) for task in os.listdir(img_path)]
    task_path = [os.path.join(salmap_path, task) for task in os.listdir(salmap_path)]
    # fixation_path =

    # for mac
    for i in range(len(task_path)):
        if task_path[i].split('/')[-1] == '.DS_Store':
            task_path.remove(task_path[i])
            break

    task_path = random.sample(task_path, meta_batch)

    return task_path

class MetaLearner(object):
    def __init__(self, meta_updates, meta_batch, second_meta_batch, lr_alpha, lr_beta, save_path, shot):
        self.meta_updates = meta_updates
        self.meta_batch = meta_batch
        self.second_meta_batch = second_meta_batch
        # self.loss_function = loss_function
        self.lr_alpha = lr_alpha
        self.lr_beta = lr_beta
        self.save_path = save_path
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        self.shot = shot

        # self.network = DenseEDNet()
        self.network = VGGModel()
        # if isinstance(self.network, nn.DataParallel):
        #     self.network = self.network.module
        # self.network = nn.DataParallel(self.network)
        self.network.to(device)
        #self.network = nn.DataParallel(self.network)
        network_dict = self.network.state_dict()
        # weights_path = '/home/wzq/PycharmProjects/new_weights/s/deconv_context_bs10-16-24_1e-14.pth'
        weights_path = r'C:\Users\18817\Documents\PHcode\simplenet\saved_models\model1.pt'
        #weights_path = r'C:\Users\18817\Documents\PHcode\simplenet\saved_models\salicon_densenet.pt'
        #weights_path = r'D:\PHcode\second-2\simplenet_vgg\saliconresnet.pt'
        # weights_path = '/home/wzq/Second_project/weights/epoch_216'
        # weights_path = '/home/wzq/PycharmProjects/new-master/vgg16_conv.pth'
        pretrained_dict = torch.load(weights_path)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in network_dict}
        network_dict.update(pretrained_dict)
        self.network.load_state_dict(network_dict)
        #self.network.load_state_dict(self.weights['state_dict'])
        self.fast_network = BaseNetwork(self.lr_alpha, self.meta_batch)
        # if isinstance(self.fast_network, torch.nn.DataParallel):
        #     self.fast_network = self.fast_network.module
        # self.fast_network = nn.DataParallel(self.fast_network)
        self.fast_network.to(device)
        #self.fast_network = nn.DataParallel(self.fast_network)

        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr_beta)

    def meta_update(self, sal_maps_val, meta_gradients):
        logging.info('===> Updating network')
        data_val = Mydataset(sal_maps_val,
                             images_path=r'C:\Users\18817\Documents\PHcode\fspsp\PSM\all_images_release')
        dataloader_val = data.DataLoader(data_val, batch_size=1, shuffle=True)
        img, sal_map = dataloader_val.__iter__().next()
        _, loss = forward_pass(self.network, img, sal_map)

        gradients = {g: sum(d[g] for d in meta_gradients) for g in meta_gradients[0].keys()}
        # logging.info('===> Gradients updated: {}'.format(gradients))

        # hook
        hooks = []
        for key, value in self.network.named_parameters():
            def get_closure():
                k = key

                def replace_grad(grad):
                    return gradients[k]

                return replace_grad

            # if 'dense' not in key:
            if 'deconv' in key:

                hooks.append(value.register_hook(get_closure()))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        for hook in hooks:
            hook.remove()

    def test(self):
        #test_network = DenseEDNet()

        test_iterations = 10
        loss_test_avg, CC_test_avg = 0.0, 0.0
        print("** Testing meta network for {} iterations".format(test_iterations))

        for _ in range(5):
            test_network = VGGModel()
            loss_mtra, KLD_mtra, CC_mtra, SIM_mtra, loss_mval, KLD_mval, CC_mval, SIM_mval = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            test_network.to(device)
            #test_network.load_state_dict(torch.load('./pnasweights_5_10_10way5shot/epoch_481.pt'))
            test_network.load_state_dict(torch.load('./vggweights_6_14_5way5shot/vgg1_epoch_481.pt'))
            test_optimizer = torch.optim.SGD(test_network.parameters(), lr=self.lr_beta)
            test_task_path = [
                r'C:\Users\18817\Documents\PHcode\fspsp\Personalized Saliency Dataset\fixation_map_30\test\Sub_5',
                r'C:\Users\18817\Documents\PHcode\fspsp\Personalized Saliency Dataset\fixation_map_30\test\Sub_10',
                r'C:\Users\18817\Documents\PHcode\fspsp\Personalized Saliency Dataset\fixation_map_30\test\Sub_15',
                r'C:\Users\18817\Documents\PHcode\fspsp\Personalized Saliency Dataset\fixation_map_30\test\Sub_20',
                r'C:\Users\18817\Documents\PHcode\fspsp\Personalized Saliency Dataset\fixation_map_30\test\Sub_30']
            sal_maps_names = os.listdir(
                r'C:\Users\18817\Documents\PHcode\fspsp\Personalized Saliency Dataset\fixation_map_30\test\Sub_15')
            sal_maps_names.sort()
            sal_maps_names_all = []
            sal_maps_names_tra = sal_maps_names[0:50]
            sal_maps_names_val = sal_maps_names[50:100]

            # for i in range(1600):
            #     if (i + 1) % 4 == 0:
            #         sal_maps_names_val.append(sal_maps_names[i])
            #     else:
            #         sal_maps_names_tra.append(sal_maps_names[i])
            # random.shuffle(sal_maps_names_tra)
            # fix_imgs_tra = random.sample(sal_maps_names_tra, self.shot)  # 随机取5张图片
            # for i in range(50):
            #     sal_maps_names_all.append(sal_maps_names[i])
            random.shuffle(sal_maps_names_tra)
            fix_imgs_tra = sal_maps_names_tra[0: self.shot]  # 随机取5张图片
            sal_maps_names_val = sal_maps_names[50: 100]

            sal_maps_tra = [os.path.join(test_task_path[_], sal_map) for sal_map in fix_imgs_tra]
            sal_maps_val_all = [os.path.join(test_task_path[_], sal_map) for sal_map in sal_maps_names_val]
            #sal_maps_val = sal_maps_val_all[0: self.shot]
            print(sal_maps_tra)
            data_tra = Mydataset(sal_maps_tra,
                                 images_path=r'C:\Users\18817\Documents\PHcode\fspsp\Personalized Saliency Dataset\all_images_release')
            dataloader_tra = data.DataLoader(data_tra, batch_size=3, shuffle=True)
            data_val = Mydataset(sal_maps_val_all,
                                 images_path=r'C:\Users\18817\Documents\PHcode\fspsp\Personalized Saliency Dataset\all_images_release')
            dataloader_val = data.DataLoader(data_val, batch_size=3, shuffle=True)
            # for i in range(10):
            loss_test_avg, CC_test_avg, KLD_test_avg, SIM_test_avg, AUC_test_avg, NSS_test_avg = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            loss_test, CC_test,KLD,SIM,NSS = 0.0, 0.0, 0.0, 0.0, 0.0
            # sal_maps_test_path1 = [os.path.join(test_task_path[0], task) for task in sal_maps_names_val]
            # sal_maps_test_path2 = [os.path.join(test_task_path[1], task) for task in sal_maps_names_val]
            # sal_maps_test_path3 = [os.path.join(test_task_path[2], task) for task in sal_maps_names_val]
            # sal_maps_test_path4 = [os.path.join(test_task_path[3], task) for task in sal_maps_names_val]
            # sal_maps_test_path5 = [os.path.join(test_task_path[4], task) for task in sal_maps_names_val]
            # sal_maps_test_path = sal_maps_test_path1 + sal_maps_test_path2 + sal_maps_test_path3 + sal_maps_test_path4 + sal_maps_test_path5
            # print(sal_maps_test_path)
            for idx, datas in enumerate(dataloader_tra):
                _input, _target = datas[0], datas[1]
                #print('strat')
                _, loss = forward_pass(test_network, _input, _target)
                test_optimizer.zero_grad()
                loss.backward()
                test_optimizer.step()

            test_network.eval()

            with torch.no_grad():
                for index, (img, sal_map) in enumerate(dataloader_val):
                    img = img.to(device)
                    sal_map = sal_map.to(device)
                    out = test_network(img)
                    #print(out.shape,sal_map.shape)
                    loss_test = loss_function(out, sal_map)
                    CC_test = cc(out, sal_map)
                    KLD = kldiv(out, sal_map)
                    SIM = similarity(out, sal_map)
                    NSS = nss(out, sal_map)
                    #AUC = auc_judd(out, sal_map)

                    loss_test_avg += loss_test.item()
                    CC_test_avg += CC_test.item()
                    KLD_test_avg += KLD.item()
                    SIM_test_avg += SIM.item()
                    NSS_test_avg += NSS.item()
                    #AUC_test_avg +=AUC.item()

                loss_test_avg /= len(dataloader_val)
                CC_test_avg /= len(dataloader_val)
                KLD_test_avg /= len(dataloader_val)
                SIM_test_avg /= len(dataloader_val)
                NSS_test_avg /= len(dataloader_val)
                #AUC_test_avg /= len(dataloader_val)

            print('(Meta-testing) Evaluation loss: {} CC: {} KLD: {} SIM: {} NSS: {}'.format(loss_test_avg, CC_test_avg, KLD_test_avg, SIM_test_avg, NSS_test_avg))
            del test_network

    def meta_test_on_meta_train(self):
        # test_network = DenseEDNet()
        test_network = PNASModel()
        loss_mtra, KLD_mtra, CC_mtra, SIM_mtra, loss_mval, KLD_mval, CC_mval, SIM_mval = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        test_network.to(device)
        #test_network.load_state_dict(torch.load('./pnasweights/epoch_96.pt'))
        test_iterations = 10
        loss_test_avg, CC_test_avg = 0.0, 0.0
        print("** Testing meta network for {} iterations".format(test_iterations))

        for _ in range(5):
            test_network.copy_weights(self.network)
            # for param in test_network.frontend.parameters():
            #     param.requires_grad = False
            test_optimizer = torch.optim.SGD(test_network.parameters(), lr=self.lr_beta)
            test_task_path = [
                r'C:\Users\18817\Documents\PHcode\fspsp\Personalized Saliency Dataset\fixation_map_30\test\Sub_15',
                r'C:\Users\18817\Documents\PHcode\fspsp\Personalized Saliency Dataset\fixation_map_30\test\Sub_30',
                r'C:\Users\18817\Documents\PHcode\fspsp\Personalized Saliency Dataset\fixation_map_30\test\Sub_5',
                r'C:\Users\18817\Documents\PHcode\fspsp\Personalized Saliency Dataset\fixation_map_30\test\Sub_10',
                r'C:\Users\18817\Documents\PHcode\fspsp\Personalized Saliency Dataset\fixation_map_30\test\Sub_20']
            sal_maps_names = os.listdir(
                r'C:\Users\18817\Documents\PHcode\fspsp\Personalized Saliency Dataset\fixation_map_30\test\Sub_15')
            sal_maps_names.sort()
            sal_maps_names_tra = []
            sal_maps_names_val = []
            for i in range(1600):
                if (i + 1) % 4 == 0:
                    sal_maps_names_val.append(sal_maps_names[i])
                else:
                    sal_maps_names_tra.append(sal_maps_names[i])
            random.shuffle(sal_maps_names_tra)
            fix_imgs_tra = random.sample(sal_maps_names_tra, self.shot)  # 随机取5张图片
            sal_maps_tra = [os.path.join(test_task_path[_], sal_map) for sal_map in fix_imgs_tra]
            sal_maps_val_all = [os.path.join(test_task_path[_], sal_map) for sal_map in sal_maps_names_val]
            sal_maps_val = sal_maps_val_all[0: self.shot]
            print(sal_maps_tra)
            data_tra = Mydataset(sal_maps_tra,
                                 images_path=r'C:\Users\18817\Documents\PHcode\fspsp\Personalized Saliency Dataset\all_images_release')
            dataloader_tra = data.DataLoader(data_tra, batch_size=3, shuffle=True)
            data_val = Mydataset(sal_maps_val,
                                 images_path=r'C:\Users\18817\Documents\PHcode\fspsp\Personalized Saliency Dataset\all_images_release')
            dataloader_val = data.DataLoader(data_val, batch_size=3, shuffle=True)
            for i in range(10):
                loss_test_avg, CC_test_avg = 0.0, 0.0
                # sal_maps_test_path1 = [os.path.join(test_task_path[0], task) for task in sal_maps_names_val]
                # sal_maps_test_path2 = [os.path.join(test_task_path[1], task) for task in sal_maps_names_val]
                # sal_maps_test_path3 = [os.path.join(test_task_path[2], task) for task in sal_maps_names_val]
                # sal_maps_test_path4 = [os.path.join(test_task_path[3], task) for task in sal_maps_names_val]
                # sal_maps_test_path5 = [os.path.join(test_task_path[4], task) for task in sal_maps_names_val]
                # sal_maps_test_path = sal_maps_test_path1 + sal_maps_test_path2 + sal_maps_test_path3 + sal_maps_test_path4 + sal_maps_test_path5
                # print(sal_maps_test_path)
                for idx, datas in enumerate(dataloader_tra):
                    _input, _target = datas[0], datas[1]
                    # print('strat')
                    _, loss = forward_pass(test_network, _input, _target)
                    test_optimizer.zero_grad()
                    loss.backward()
                    test_optimizer.step()

                test_network.eval()

                with torch.no_grad():
                    for index, (img, sal_map) in enumerate(dataloader_val):
                        img = img.to(device)
                        sal_map = sal_map.to(device)
                        out = test_network(img)

                        loss_test = loss_function(out, sal_map)
                        CC_test = cc(out, sal_map)
                        loss_test_avg += loss_test.item()
                        CC_test_avg += CC_test.item()

                    loss_test_avg /= len(dataloader_val)
                    CC_test_avg /= len(dataloader_val)

                print('(Meta-testing) Evaluation loss: {} CC: {}'.format(loss_test_avg, CC_test_avg))
        del test_network




    def train(self):
        loss_mtra, KLD_mtra, CC_mtra, SIM_mtra, loss_mval, KLD_mval, CC_mval, SIM_mval = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        for index, epoch in enumerate(range(self.meta_updates)):
            print("===> Training epoch: {}/{}".format(index + 1, self.meta_updates))
            logging.info("===> Training epoch: {}/{}".format(index + 1, self.meta_updates))
            #self.meta_test_on_meta_train()
            task_seq = get_task_seq(
                r'C:\Users\18817\Documents\PHcode\fspsp\Personalized Saliency Dataset\fixation_map_30\train', self.meta_batch)
            print(task_seq)
            sal_maps_names = os.listdir(r'C:\Users\18817\Documents\PHcode\fspsp\Personalized Saliency Dataset\fixation_map_30\test\Sub_15')
            sal_maps_names.sort()
            sal_maps_names_all = []
            sal_maps_names_tra = sal_maps_names[0:50]
            sal_maps_names_val = sal_maps_names[50:100]



            # for i in range(1600):
            #     if (i + 1) % 4 == 0:
            #         sal_maps_names_val.append(sal_maps_names[i])
            #     else:
            #         sal_maps_names_tra.append(sal_maps_names[i])
            # random.shuffle(sal_maps_names_tra)
            # fix_imgs_tra = random.sample(sal_maps_names_tra, self.shot)#随机取5张图片
            # for i in range(50):
            #     sal_maps_names_all.append(sal_maps_names[i])
            random.shuffle(sal_maps_names_tra)
            #random.shuffle(sal_maps_names_valall)
            fix_imgs_tra = sal_maps_names_tra[0: self.shot]#随机取5张图片
            #sal_maps_names_val = sal_maps_names_valall[0: self.shot]
            #sal_maps_names_val = sal_maps_names_all[self.shot: 50]
            # print('task_seq: ', len(task_seq))
            meta_gradients = []
            meta_gradients2 = []
            c_loss = []
            loss_tra, KLD_tra, CC_tra, SIM_tra, loss_val, KLD_val, CC_val, SIM_val = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            for i in range(1):
                print("===> Training for meta_batch {}".format(i))
                for index, _ in enumerate(range(self.meta_batch)):
                    print("==> Training for person {}/{}".format(index + 1, self.meta_batch))
                    logging.info("===> Training for person {}/{}".format(index + 1, self.meta_batch))

                    task = task_seq[index]
                    # print('tasks: ', task)

                    sal_maps_tra = [os.path.join(task, sal_map) for sal_map in fix_imgs_tra]
                    random.shuffle(sal_maps_names_val)
                    sal_maps_val_all = [os.path.join(task, sal_map) for sal_map in sal_maps_names_val]

                    # random.shuffle(sal_maps)

                    sal_maps_val = sal_maps_val_all[0: self.shot]
                    # random.shuffle(sal_maps_tra)


                    self.fast_network.copy_weights(self.network)
                    #self.fast_network = nn.DataParallel(self.fast_network)
                    self.fast_network.to(device)
                    # print(sal_maps_tra)
                    # print(sal_maps_val)

                    metrics, grad, loss = self.fast_network.forward(sal_maps_tra, sal_maps_val)

                    (loss_tra_post, KLD_tra_post, CC_tra_post, SIM_tra_post, loss_val_post, KLD_val_post, CC_val_post,
                     SIM_val_post) = metrics
                    meta_gradients.append(grad)
                    c_loss.append(loss.cpu())

                    loss_tra += loss_tra_post
                    KLD_tra += KLD_tra_post
                    CC_tra += CC_tra_post
                    SIM_tra += SIM_tra_post
                    loss_val += loss_val_post
                    KLD_val += KLD_val_post
                    CC_val += CC_val_post
                    SIM_val += SIM_val_post

            #print(c_loss)
            self.meta_update(sal_maps_val, meta_gradients)
            temp = []
            for i in range(self.second_meta_batch):
                temp.append(c_loss.index(max(c_loss)))
                c_loss[c_loss.index(max(c_loss))] = tensor(0.0)
            temp.sort()
            #print(temp)
            for index in range(self.second_meta_batch):
                print("==> Training for person {}/{}".format(index + 1, 2))
                task = task_seq[temp[index]]
                sal_maps_tra = [os.path.join(task, sal_map) for sal_map in fix_imgs_tra]
                random.shuffle(sal_maps_names_val)
                sal_maps_val_all = [os.path.join(task, sal_map) for sal_map in sal_maps_names_val]
                sal_maps_val = sal_maps_val_all[0: self.shot]
                self.fast_network.copy_weights(self.network)
                self.fast_network.to(device)
                #self.fast_network = nn.DataParallel(self.fast_network)
                metrics2, grad2, loss2 = self.fast_network.forward(sal_maps_tra, sal_maps_val)
                (loss_tra_post, KLD_tra_post, CC_tra_post, SIM_tra_post, loss_val_post, KLD_val_post, CC_val_post,
                 SIM_val_post) = metrics2
                meta_gradients2.append(grad2)
                loss_tra += loss_tra_post
                KLD_tra += KLD_tra_post
                CC_tra += CC_tra_post
                SIM_tra += SIM_tra_post
                loss_val += loss_val_post
                KLD_val += KLD_val_post
                CC_val += CC_val_post
                SIM_val += SIM_val_post

            self.meta_update(sal_maps_val, meta_gradients2)


            # Evaluating the model
            # test_task_path = ['/home/wzq/Second_project/PSM_dataset/fixation_map_30_release/test/Sub_15',
            #                   '/home/wzq/Second_project/PSM_dataset/fixation_map_30_release/test/Sub_30',
            #                   '/home/wzq/Second_project/PSM_dataset/fixation_map_30_release/test/Sub_5',
            #                   '/home/wzq/Second_project/PSM_dataset/fixation_map_30_release/test/Sub_10',
            #                   '/home/wzq/Second_project/PSM_dataset/fixation_map_30_release/test/Sub_20']
            test_task_path = [r'C:\Users\18817\Documents\PHcode\fspsp\Personalized Saliency Dataset\fixation_map_30\test\Sub_15',
                              r'C:\Users\18817\Documents\PHcode\fspsp\Personalized Saliency Dataset\fixation_map_30\test\Sub_30',
                              r'C:\Users\18817\Documents\PHcode\fspsp\Personalized Saliency Dataset\fixation_map_30\test\Sub_5',
                              r'C:\Users\18817\Documents\PHcode\fspsp\Personalized Saliency Dataset\fixation_map_30\test\Sub_10',
                              r'C:\Users\18817\Documents\PHcode\fspsp\Personalized Saliency Dataset\fixation_map_30\test\Sub_20']
            if epoch % 10 == 0 and epoch >=449:
                loss_test_avg, CC_test_avg = 0.0, 0.0
                print("==> Evaluating the model at: {}".format(epoch + 1))
                # logging.info("==> Evaluating the model at: {}".format(epoch + 1))
                # sal_maps_test_path1 = [os.path.join(test_task_path[0], task) for task in sal_maps_names_val]
                # sal_maps_test_path2 = [os.path.join(test_task_path[1], task) for task in sal_maps_names_val]
                # sal_maps_test_path3 = [os.path.join(test_task_path[2], task) for task in sal_maps_names_val]
                # sal_maps_test_path4 = [os.path.join(test_task_path[3], task) for task in sal_maps_names_val]
                # sal_maps_test_path5 = [os.path.join(test_task_path[4], task) for task in sal_maps_names_val]
                # sal_maps_test_path = sal_maps_test_path1 + sal_maps_test_path2 + sal_maps_test_path3 + sal_maps_test_path4 + sal_maps_test_path5
                # data_test = Mydataset(sal_maps_test_path,
                #                      images_path=r'C:\Users\18817\Documents\PHcode\fspsp\Personalized Saliency Dataset\all_images_release')
                # dataloader_test = data.DataLoader(data_test, batch_size=1, shuffle=False)
                data_test = Mydataset(sal_maps_val_all,
                                      images_path=r'C:\Users\18817\Documents\PHcode\fspsp\Personalized Saliency Dataset\all_images_release')
                dataloader_test = data.DataLoader(data_test, batch_size=1, shuffle=False)
                print(len(dataloader_test))
                # test_network = DenseEDNet()
                test_network = VGGModel()
                #test_network = nn.DataParallel(test_network)
                test_network.to(device)

                test_network.copy_weights(self.network)

                test_network.eval()
                mse = nn.MSELoss()

                with torch.no_grad():
                    for index, (img, sal_map) in enumerate(dataloader_test):
                        img = img.to(device)
                        sal_map = sal_map.to(device)
                        out = test_network(img)
                        #print(sal_map.size(), out.size())
                        loss_test = loss_function(out, sal_map)
                        #loss_test = loss_function(out, sal_map)
                        #print(loss_test)
                        CC_test = cc(out, sal_map)
                        #print(CC_test)
                        loss_test_avg += loss_test.item()
                        CC_test_avg += CC_test.item()

                    loss_test_avg /= len(dataloader_test)
                    CC_test_avg /= len(dataloader_test)

                print('Evaluation loss: {} CC: {}'.format(loss_test_avg, CC_test_avg))
                logging.info('Evaluation loss: {} CC: {}'.format(loss_test_avg, CC_test_avg))

                if epoch == 450:
                    best_loss = loss_test_avg
                else:
                    if loss_test_avg <= best_loss:
                        best_loss = loss_test_avg
                        torch.save(self.network.state_dict(),
                                   '{}/vgg1_epoch_{}.pt'.format(self.save_path, epoch + 1))
                        print('Saving model at {}'.format(epoch + 1))
                        logging.info('Saving model at {}'.format(epoch + 1))

    def train_maxcc50(self):
        for index, epoch in enumerate(range(self.meta_updates)):
            print("===> Training epoch: {}/{}".format(index + 1, self.meta_updates))
            logging.info("===> Training epoch: {}/{}".format(index + 1, self.meta_updates))
            task_seq = get_task_seq(
                 r'C:\Users\18817\Documents\PHcode\fspsp\PSM\fixation_map_30_release\train', self.meta_batch)
            print(task_seq)
            sal_maps_names = os.listdir(r'D:\PHcode\second-2\test5_max_cc\20')
            sal_maps_names.sort()
            sal_maps_names_tra = sal_maps_names
            random.shuffle(sal_maps_names_tra)

            fix_imgs_tra = sal_maps_names_tra[0: self.shot]  # 随机取5张图片
            sal_maps_names_val = sal_maps_names_tra[self.shot: 50]

            meta_gradients = []
            meta_gradients2 = []
            c_loss = []
            loss_tra, KLD_tra, CC_tra, SIM_tra, loss_val, KLD_val, CC_val, SIM_val = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

            #print("===> Training for meta_batch {}".format(epoch + 1))
            for index, _ in enumerate(range(self.meta_batch)):
                print("==> Training for person {}/{}".format(index + 1, self.meta_batch))
                logging.info("===> Training for person {}/{}".format(index + 1, self.meta_batch))

                task = task_seq[index]

                sal_maps_tra = [os.path.join(task, sal_map) for sal_map in fix_imgs_tra]
                random.shuffle(sal_maps_names_val)
                sal_maps_val_all = [os.path.join(task, sal_map) for sal_map in sal_maps_names_val]

                #sal_maps_val = sal_maps_val_all[0: self.shot]
                sal_maps_val = sal_maps_val_all

                self.fast_network.copy_weights(self.network)
                self.fast_network.to(device)

                metrics, grad, loss = self.fast_network.forward(sal_maps_tra, sal_maps_val)

                (loss_tra_post, KLD_tra_post, CC_tra_post, SIM_tra_post, loss_val_post, KLD_val_post, CC_val_post,
                 SIM_val_post) = metrics
                meta_gradients.append(grad)
                c_loss.append(loss.cpu())

                loss_tra += loss_tra_post
                KLD_tra += KLD_tra_post
                CC_tra += CC_tra_post
                SIM_tra += SIM_tra_post
                loss_val += loss_val_post
                KLD_val += KLD_val_post
                CC_val += CC_val_post
                SIM_val += SIM_val_post

            self.meta_update(sal_maps_val, meta_gradients)
            temp = []
            for i in range(self.second_meta_batch):
                temp.append(c_loss.index(max(c_loss)))
                c_loss[c_loss.index(max(c_loss))] = tensor(0.0)
            temp.sort()
            for index in range(self.second_meta_batch):
                print("==> Training for person {}/{}".format(index + 1, 2))
                task = task_seq[temp[index]]
                sal_maps_tra = [os.path.join(task, sal_map) for sal_map in fix_imgs_tra]
                # random.shuffle(sal_maps_names_val)
                sal_maps_val_all = [os.path.join(task, sal_map) for sal_map in sal_maps_names_val]
                #sal_maps_val = sal_maps_val_all[0: self.shot]
                sal_maps_val = sal_maps_val_all

                self.fast_network.copy_weights(self.network)
                self.fast_network.to(device)
                # self.fast_network = nn.DataParallel(self.fast_network)
                metrics2, grad2, loss2 = self.fast_network.forward(sal_maps_tra, sal_maps_val)
                (loss_tra_post, KLD_tra_post, CC_tra_post, SIM_tra_post, loss_val_post, KLD_val_post, CC_val_post,
                 SIM_val_post) = metrics2
                meta_gradients2.append(grad2)
                loss_tra += loss_tra_post
                KLD_tra += KLD_tra_post
                CC_tra += CC_tra_post
                SIM_tra += SIM_tra_post
                loss_val += loss_val_post
                KLD_val += KLD_val_post
                CC_val += CC_val_post
                SIM_val += SIM_val_post

            self.meta_update(sal_maps_val, meta_gradients2)

            if epoch >= 449:
                loss_test_avg, CC_test_avg = 0.0, 0.0
                print("==> Evaluating the model at: {}".format(epoch + 1))

                data_test = Mydataset(sal_maps_val_all,
                                      images_path=r'C:\Users\18817\Documents\PHcode\fspsp\PSM\all_images_release')
                dataloader_test = data.DataLoader(data_test, batch_size=1, shuffle=False)
                print(len(dataloader_test))
                # test_network = DenseEDNet()
                test_network = VGGModel()
                test_network.to(device)
                test_network.copy_weights(self.network)

                test_network.eval()

                with torch.no_grad():
                    for index, (img, sal_map) in enumerate(dataloader_test):
                        img = img.to(device)
                        sal_map = sal_map.to(device)
                        out = test_network(img)

                        loss_test = loss_function(out, sal_map)
                        CC_test = cc(out, sal_map)
                        loss_test_avg += loss_test.item()
                        CC_test_avg += CC_test.item()

                    loss_test_avg /= len(dataloader_test)
                    CC_test_avg /= len(dataloader_test)

                print('Evaluation loss: {} CC: {}'.format(loss_test_avg, CC_test_avg))
                logging.info('Evaluation loss: {} CC: {}'.format(loss_test_avg, CC_test_avg))

                if epoch == 449:
                    best_loss = loss_test_avg
                else:
                    if loss_test_avg <= best_loss:
                        best_loss = loss_test_avg
                        torch.save(self.network.state_dict(),
                                   '{}/vgg2_epoch_{}.pt'.format(self.save_path, epoch + 1))
                        print('Saving model at {}'.format(epoch + 1))
                        logging.info('Saving model at {}'.format(epoch + 1))

    def save_maxcc50(self):
        path_a = './result/result_s_cc_sim_kld_45/'
        if not os.path.exists(path_a):
            os.mkdir(path_a)
        for _ in range(5):
            test_iterations = 10
            test_network = VGGModel()
            loss_mtra, KLD_mtra, CC_mtra, SIM_mtra, loss_mval, KLD_mval, CC_mval, SIM_mval = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            test_network.to(device)
            test_network.load_state_dict(torch.load('./vggweights_11_17_5way5shot/vgg2_epoch_472.pt'))

            test_optimizer = torch.optim.SGD(test_network.parameters(), lr=0.005)
            test_task_path = [
                  r'C:\Users\18817\Documents\PHcode\second-2\test_s_max_cc\5',
                  r'C:\Users\18817\Documents\PHcode\second-2\test_s_max_cc\12',
                  r'C:\Users\18817\Documents\PHcode\second-2\test_s_max_cc\20',
                  r'C:\Users\18817\Documents\PHcode\second-2\test_s_max_cc\22',
                  r'C:\Users\18817\Documents\PHcode\second-2\test_s_max_cc\30']
            sal_maps_names = os.listdir(
                  r'C:\Users\18817\Documents\PHcode\second-2\test_s_max_cc\20')

            # test_task_path = [
            #     '/home/lxh/PHcode/second-2/train/train2_max50_cc/5/',
            #     '/home/lxh/PHcode/second-2/train/train2_max50_cc/12/',
            #     '/home/lxh/PHcode/second-2/train/train2_max50_cc/20/',
            #     '/home/lxh/PHcode/second-2/train/train2_max50_cc/22/',
            #     '/home/lxh/PHcode/second-2/train/train2_max50_cc/30/']
            # sal_maps_names = os.listdir(
            #     '/home/lxh/PHcode/second-2/train/train2_max50_cc/20')
            sal_maps_names.sort()
            sal_maps_names_all = []

            # sal_maps_names_tra = sal_maps_names[0:50]
            # fix_imgs_tra = sample(sal_maps_names_tra, 5)
            # for ii in range(len(sal_maps_names_tra)):
            #     for j in sal_maps_names_tra:
            #         if j in fix_imgs_tra:
            #             sal_maps_names_tra.remove(j)
            # sal_maps_names_val = sal_maps_names_tra

            sal_maps_names_tra = sal_maps_names
            gg = _
            fix_imgs_tra = sal_maps_names_tra[(gg * 5):(5 + gg * 5)]
            del sal_maps_names_tra[(gg * 5):(5 + gg * 5)]
            sal_maps_names_val = sal_maps_names_tra

            # sal_maps_names_tra = sal_maps_names
            # gg = _
            # fix_imgs_tra = sal_maps_names_tra[(gg * 5):(5 + gg * 5)]
            # sal_maps_names_val = sal_maps_names_tra[25:50]
            #
            for i in range(test_iterations):
                # test_task_path = [
                #     '/home/lxh/PHcode/second-2/train_max50_cc/3/',
                #     '/home/lxh/PHcode/second-2/train_max50_cc/5/',
                #     '/home/lxh/PHcode/second-2/train_max50_cc/10/',
                #     '/home/lxh/PHcode/second-2/train_max50_cc/12/',
                #     '/home/lxh/PHcode/second-2/train_max50_cc/15/',
                #     '/home/lxh/PHcode/second-2/train_max50_cc/18/',
                #     '/home/lxh/PHcode/second-2/train_max50_cc/20/',
                #     '/home/lxh/PHcode/second-2/train_max50_cc/22/',
                #     '/home/lxh/PHcode/second-2/train_max50_cc/25/',
                #     '/home/lxh/PHcode/second-2/train_max50_cc/30/']
                # sal_maps_names = os.listdir(
                #     '/home/lxh/PHcode/second-2/train_max50_cc/15')
                # sal_maps_names.sort()
                #
                # sal_maps_names_tra = sal_maps_names
                # random.shuffle(sal_maps_names_tra)
                #
                # gg = 0
                # fix_imgs_tra = sal_maps_names_tra[(gg * 5):(5 + gg * 5)]
                # del sal_maps_names_tra[(gg * 5):(5 + gg * 5)]
                # sal_maps_names_val = sal_maps_names_tra

                sal_maps_tra = [os.path.join(test_task_path[_], sal_map) for sal_map in fix_imgs_tra]
                sal_maps_val_all = [os.path.join(test_task_path[_], sal_map) for sal_map in sal_maps_names_val]
                print(sal_maps_tra)
                data_tra = Mydataset(sal_maps_tra,
                                     images_path=r'C:\Users\18817\Documents\PHcode\fspsp\PSM\all_images_release')
                dataloader_tra = data.DataLoader(data_tra, batch_size=self.shot, shuffle=True)


                for idx, datas in enumerate(dataloader_tra):
                    _input, _target = datas[0], datas[1]
                    # print('strat')
                    t, loss = forward_pass(test_network, _input, _target)
                    test_optimizer.zero_grad()
                    loss.backward()
                    test_optimizer.step()

            test_network.eval()
            test_sub = ['5','12','20','22','30']

            path = os.path.join(path_a, test_sub[_])
            if not os.path.exists(path):
                os.mkdir(path)

            # meta-save
            # test_task_path = [
            #     '/home/lxh/PHcode/second-2/train_max50_cc/3/',
            #     '/home/lxh/PHcode/second-2/train_max50_cc/5/',
            #     '/home/lxh/PHcode/second-2/train_max50_cc/10/',
            #     '/home/lxh/PHcode/second-2/train_max50_cc/12/',
            #     '/home/lxh/PHcode/second-2/train_max50_cc/15/',
            #     '/home/lxh/PHcode/second-2/train_max50_cc/18/',
            #     '/home/lxh/PHcode/second-2/train_max50_cc/20/',
            #     '/home/lxh/PHcode/second-2/train_max50_cc/22/',
            #     '/home/lxh/PHcode/second-2/train_max50_cc/25/',
            #     '/home/lxh/PHcode/second-2/train_max50_cc/30/']
            # sal_maps_names = os.listdir(
            #     '/home/lxh/PHcode/second-2/train_max50_cc/15')


            # sal_maps_names.sort()
            # sal_maps_names_tra = sal_maps_names
            # fix_imgs_tra = sal_maps_names_tra[(0 + _ * 5):(5 + _ * 5)]
            # del sal_maps_names_tra[(0 + _ * 5):(5 + _ * 5)]
            # sal_maps_names_val = sal_maps_names_tra
            # sal_maps_val_all = [os.path.join(test_task_path[_], sal_map) for sal_map in sal_maps_names_val]

            print('val--------------{}:'.format(_))
            print(sal_maps_val_all)
            data_val = Mydataset(sal_maps_val_all,
                                 images_path=r'C:\Users\18817\Documents\PHcode\fspsp\PSM\all_images_release')

            dataloader_val = data.DataLoader(data_val, batch_size=1, shuffle=True)
            with torch.no_grad():
                for index, (img, sal_map, name) in enumerate(dataloader_val):
                    print(name[0])
                    #name = sal_maps_names_val[index].split('/')[-1]
                    savepath = os.path.join(path,name[0])
                    #print(savepath)
                    img = img.to(device)

                    out = test_network(img)
                    out = out.cpu().squeeze(0).numpy()
                    #print(out.shape)
                    out = cv2.resize(out, (640,360))

                    out = torch.FloatTensor(blur(out))

                    img_save(out, savepath, normalize=True)
            del test_network

    def test_maxcc50(self):
        for _ in range(5):
            test_iterations = 11
            loss_test_avg, CC_test_avg = 0.0, 0.0
            print("** Testing meta network for {} iterations".format(test_iterations))
            cc_all, kld_all, sim_all, nss_all, auc_all = [], [], [], [], []
            test_network = VGGModel()
            loss_mtra, KLD_mtra, CC_mtra, SIM_mtra, loss_mval, KLD_mval, CC_mval, SIM_mval = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            test_network.to(device)
            test_network.load_state_dict(torch.load('./vggweights_11_18_5way1shot/vgg2_epoch_479.pt'))
            test_optimizer = torch.optim.SGD(test_network.parameters(), lr=0.005)
            test_task_path = [
                r'D:\PHcode\second-2\test5_max_cc\5',
                r'D:\PHcode\second-2\test5_max_cc\12',
                r'D:\PHcode\second-2\test5_max_cc\20',
                r'D:\PHcode\second-2\test5_max_cc\22',
                r'D:\PHcode\second-2\test5_max_cc\30']
            sal_maps_names = os.listdir(
                r'D:\PHcode\second-2\test5_max_cc\20')

            sal_maps_names.sort()
            sal_maps_names_all = []

            sal_maps_names_tra = sal_maps_names

            # gg = _
            # fix_imgs_tra = sal_maps_names_tra[(gg * 5):(5 + gg * 5)]
            # del sal_maps_names_tra[(gg * 5):(5 + gg * 5)]
            # sal_maps_names_val = sal_maps_names_tra

            gg = _
            fix_imgs_tra = sal_maps_names_tra[(gg * 5):(1 + gg * 5)]
            del sal_maps_names_tra[(gg * 5):(1 + gg * 5)]
            sal_maps_names_val = sal_maps_names_tra

            for i in range(test_iterations):
                sal_maps_tra = [os.path.join(test_task_path[_], sal_map) for sal_map in fix_imgs_tra]
                sal_maps_val_all = [os.path.join(test_task_path[_], sal_map) for sal_map in sal_maps_names_val]
                # sal_maps_val = sal_maps_val_all[0: self.shot]
                print(len(sal_maps_names_val), sal_maps_tra)
                data_tra = Mydataset(sal_maps_tra,
                                     images_path=r'C:\Users\18817\Documents\PHcode\fspsp\PSM\all_images_release')
                dataloader_tra = data.DataLoader(data_tra, batch_size=5, shuffle=True)
                data_val = Mydataset(sal_maps_val_all,
                                     images_path=r'C:\Users\18817\Documents\PHcode\fspsp\PSM\all_images_release')
                dataloader_val = data.DataLoader(data_val, batch_size=5, shuffle=True)
                # for i in range(10):
                loss_test_avg, CC_test_avg, KLD_test_avg, SIM_test_avg, AUC_test_avg, NSS_test_avg = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                if i >= 1:
                    for idx, datas in enumerate(dataloader_tra):
                        _input, _target = datas[0], datas[1]
                        # print('strat')
                        xx, loss = forward_pass(test_network, _input, _target)
                        test_optimizer.zero_grad()
                        loss.backward()
                        test_optimizer.step()

                test_network.eval()
                if i == 0:
                    with torch.no_grad():
                        for index, (img, sal_map) in enumerate(dataloader_val):
                            # print(name)
                            img = img.to(device)
                            sal_map = sal_map.to(device)
                            out = test_network(img)
                            # print(out.shape,sal_map.shape)
                            loss_test = loss_function(out, sal_map)
                            CC_test = cc(out, sal_map)
                            KLD = kldiv(out, sal_map)
                            SIM = similarity(out, sal_map)
                            NSS = nss(out, sal_map)
                            AUC = auc_judd(out, sal_map)
                            # print('(Meta-testing) Evaluation loss: {} CC: {} KLD: {} SIM: {} NSS: {} '.format(loss_test, CC_test, KLD, SIM, NSS))
                            loss_test_avg += loss_test.item()
                            CC_test_avg += CC_test.item()
                            KLD_test_avg += KLD.item()
                            SIM_test_avg += SIM.item()
                            NSS_test_avg += NSS.item()
                            AUC_test_avg += AUC.item()

                        loss_test_avg /= len(dataloader_val)
                        CC_test_avg /= len(dataloader_val)
                        KLD_test_avg /= len(dataloader_val)
                        SIM_test_avg /= len(dataloader_val)
                        NSS_test_avg /= len(dataloader_val)
                        AUC_test_avg /= len(dataloader_val)

                    print('(Meta-testing) Evaluation loss: {} CC: {} KLD: {} SIM: {} NSS: {} AUC: {}'.format(loss_test_avg,
                                                                                                             CC_test_avg,
                                                                                                             KLD_test_avg,
                                                                                                             SIM_test_avg,
                                                                                                             NSS_test_avg,
                                                                                                             AUC_test_avg))
                if i == 10:
                    with torch.no_grad():
                        for index, (img, sal_map) in enumerate(dataloader_val):
                            # print(name)
                            img = img.to(device)
                            sal_map = sal_map.to(device)
                            out = test_network(img)
                            # print(out.shape,sal_map.shape)
                            loss_test = loss_function(out, sal_map)
                            CC_test = cc(out, sal_map)
                            KLD = kldiv(out, sal_map)
                            SIM = similarity(out, sal_map)
                            NSS = nss(out, sal_map)
                            AUC = auc_judd(out, sal_map)
                            # print('(Meta-testing) Evaluation loss: {} CC: {} KLD: {} SIM: {} NSS: {} '.format(loss_test, CC_test, KLD, SIM, NSS))
                            loss_test_avg += loss_test.item()
                            CC_test_avg += CC_test.item()
                            KLD_test_avg += KLD.item()
                            SIM_test_avg += SIM.item()
                            NSS_test_avg += NSS.item()
                            AUC_test_avg += AUC.item()

                        loss_test_avg /= len(dataloader_val)
                        CC_test_avg /= len(dataloader_val)
                        KLD_test_avg /= len(dataloader_val)
                        SIM_test_avg /= len(dataloader_val)
                        NSS_test_avg /= len(dataloader_val)
                        AUC_test_avg /= len(dataloader_val)

                    print('(Meta-testing) Evaluation loss: {} CC: {} KLD: {} SIM: {} NSS: {} AUC: {}'.format(
                        loss_test_avg,
                        CC_test_avg,
                        KLD_test_avg,
                        SIM_test_avg,
                        NSS_test_avg,
                        AUC_test_avg))
                cc_all.append(CC_test_avg)
                kld_all.append(KLD_test_avg)
                sim_all.append(SIM_test_avg)
                nss_all.append(NSS_test_avg)

            del test_network


