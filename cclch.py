import copy
import os
import random
import sys
import time

import math
import scipy.io as scio
import torch
from torch.optim import SGD
from torch.utils.data import DataLoader

from dataset.Dataset import get_dataset, TCDataset, COCODataset
from experiments.eval import retrivel
from loss import IRD_loss
from loss.ecmh_loss import object_function
from loss.supconloss import SupConLoss
from utils.data_augmentation import augmentation
from utils.data_mining import load_mat
from utils.get_alldata_coco import get_all_data


def cclch(cluster_list, old_sample, train_sample, eval_sample, image_model, text_model, lr, image_dir, transform,
          batch_size, cuda, bit, map, best_epoch, epochs, logger, num_works, temp, all_index, all_num, data_set):
    logger.info("cclch-COCO")
    temp_model_image1 = copy.deepcopy(image_model)
    temp_model_text1 = copy.deepcopy(text_model)
    temp_model_image2 = image_model
    temp_model_text2 = text_model
    temp_train = train_sample
    temp_eval = eval_sample

    logger.info(f"Number of clusters: {len(cluster_list)}")
    i = 0
    all_index=old_sample
    cluster_index = []

    while i < len(cluster_list):
        cluster_index = list(set(cluster_list[i]).difference(set(all_index)))
        while len(cluster_index) < batch_size or len(
                random.sample(cluster_index, int(math.floor(len(cluster_index) * 0.2)))) < batch_size or len(
            random.sample(cluster_index, int(math.floor(len(cluster_index) * 0.2)))) < batch_size:
            logger.info(f"Cluster {i + 1} does not have enough samples!!!")
            i = i + 1
            if i >= len(cluster_list):
                break
            cluster_index = cluster_index + list(set(cluster_list[i]).difference(set(all_index)))


        image_model_new = temp_model_image1
        text_model_new = temp_model_text1
        image_model_old = temp_model_image2
        text_model_old = temp_model_text2
        optimizer_image = SGD(image_model_new.parameters(), lr)
        optimizer_text = SGD(text_model_new.parameters(), lr)

        train_cluster_samples = random.sample(cluster_index, int(math.floor(len(cluster_index) * 0.8)))
        eval_cluster_samples = list(set(cluster_index) - set(train_cluster_samples))
        replay_samples = random.sample(temp_train, int(math.floor(len(temp_train) * 0.2)))
        train_cluster_samples = train_cluster_samples + replay_samples
        all_eval = temp_eval + eval_cluster_samples

        str_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
        logger.info(
            f"{str_time}  Training class:{i + 1} cluster  Sample number:{len(cluster_index)}  Replay_sample_number:{len(replay_samples)}")

        if data_set == "MIRFLICKR":
            _, _, full_text, full_label = load_mat(image_dir)
            train_cluster_dataset = get_dataset(image_dir, full_text, full_label, transform, batch_size,
                                                train_cluster_samples, temp)
            eval_cluster_dataset = get_dataset(image_dir, full_text, full_label, transform, batch_size,
                                               eval_cluster_samples, temp)
            temp_eval_dataset = get_dataset(image_dir, full_text, full_label, transform, batch_size, temp_eval, temp)
            all_eval_dataset = get_dataset(image_dir, full_text, full_label, transform, batch_size, all_eval, temp)
        elif data_set == "TC12":
            dataset_path = os.path.join(image_dir, "IAPRTC_12.mat")
            iapr_data = scio.loadmat(dataset_path)
            full_image = iapr_data['I_tr']
            full_text = iapr_data["T_tr"]
            full_label = iapr_data["L_tr"]
            train_cluster_dataset = TCDataset(full_image[train_cluster_samples], full_text[train_cluster_samples],
                                              full_label[train_cluster_samples], batch_size)
            eval_cluster_dataset = TCDataset(full_image[eval_cluster_samples], full_text[eval_cluster_samples],
                                             full_label[eval_cluster_samples], batch_size)
            temp_eval_dataset = TCDataset(full_image[temp_eval], full_text[temp_eval], full_label[temp_eval],
                                          batch_size)
            all_eval_dataset = TCDataset(full_image[all_eval], full_text[all_eval], full_label[all_eval], batch_size)

        elif data_set == "COCO":
            full_name, full_text, full_label = get_all_data(image_dir)
            train_image_name, train_text, train_label = [], [], []
            eval_image_name, eval_text, eval_label = [], [], []
            temp_eval_name, temp_eval_text, temp_eval_label = [], [], []
            all_eval_name, all_eval_text, all_eval_label = [], [], []
            for index in train_cluster_samples:
                index = str(index)
                train_image_name.append(full_name[index])
                train_text.append(full_text[index])
                train_label.append(full_label[index])
            for index in eval_cluster_samples:
                index = str(index)
                eval_image_name.append(full_name[index])
                eval_text.append(full_text[index])
                eval_label.append(full_label[index])
            for index in temp_eval:
                index = str(index)
                temp_eval_name.append(full_name[index])
                temp_eval_text.append(full_text[index])
                temp_eval_label.append(full_label[index])
            for index in all_eval:
                index = str(index)
                all_eval_name.append(full_name[index])
                all_eval_text.append(full_text[index])
                all_eval_label.append(full_label[index])
            train_cluster_dataset = COCODataset(image_dir, train_image_name, train_text, train_label, transform,
                                                batch_size)
            eval_cluster_dataset = COCODataset(image_dir, eval_image_name, eval_text, eval_label, transform,
                                               batch_size)
            temp_eval_dataset = COCODataset(image_dir, temp_eval_name, temp_eval_text, temp_eval_label, transform,
                                            batch_size)
            all_eval_dataset = COCODataset(image_dir, all_eval_name, all_eval_text, all_eval_label, transform,
                                           batch_size)
        train_cluster_dataloader = DataLoader(train_cluster_dataset, batch_size, shuffle=True, pin_memory=True,
                                              drop_last=True, num_workers=num_works)

        loss, IRD_loss_image, IRD_loss_text, supcl_loss_image, supcl_loss_text = 0, 0, 0, 0, 0
        current_temp = 0.2
        past_temp = 0.01
        # temp = 0.07

        image_model_old.train()
        text_model_old.train()
        image_model_new.train()
        text_model_new.train()

        for epoch in range(epochs):
            since = time.time()
            for data in train_cluster_dataloader:
                augmented_images, augmented_texts, augmented_labels = augmentation(data)
                index = data["index"].numpy()
                image = data["image"]
                text = data["text"]
                label = data["label"]
                if cuda:
                    image = image.cuda()
                text = text.cuda()
                label = label.cuda()
                augmented_images = augmented_images.cuda()
                augmented_texts = augmented_texts.cuda()
                augmented_labels = augmented_labels.cuda()
                image = torch.cat([image, augmented_images], dim=0)
                text = torch.cat([text, augmented_texts], dim=0)
                label = torch.cat([label, augmented_labels], dim=0)

                image_features = image_model_old(image)
                text_features = text_model_old(text)

                image_features1 = image_model_new(image)
                text_features1 = text_model_new(text)

                # logits1, logits2 = IRD_loss.ird_loss_current(image_features, text_features, current_temp=current_temp)
                # IRD_loss_image, IRD_loss_text = IRD_loss.ird_loss_past(image_features1, text_features1,
                #                                                        distill_power=1.0, past_temp=past_temp,
                #                                                        logits1_image=logits1, logits1_text=logits2)
                #
                # supcl_loss_image = SupConLoss(image_features1, label).cuda()
                # supcl_loss_text = SupConLoss(text_features1, label).cuda()
                # loss = IRD_loss_image + IRD_loss_text + supcl_loss_image + supcl_loss_text
                # loss = loss / (len(train_cluster_samples) * batch_size)

                # IRD
            #     supcl_loss_image = SupConLoss(image_features1, label).cuda()
            #     supcl_loss_text = SupConLoss(text_features1, label).cuda()
            #
            #     loss = supcl_loss_image + supcl_loss_text
            #     loss = loss / (len(train_cluster_samples) * batch_size)
            #
            #     optimizer_image.zero_grad()
            #     optimizer_text.zero_grad()
            #     loss.requires_grad_(True)
            #     loss.backward()
            #     optimizer_image.step()
            #     optimizer_text.step()
            #
            # time_elapsed = time.time() - since
            #
            # loss_str = "Epoch: [%3d/%3d], " % (epoch + 1, epochs)
            # loss_str += "Loss:" + " {}".format(torch.mean(loss)) + "  " + "\t"
            # # loss_str += "IRD_loss:" + " {}".format(torch.mean(IRD_loss_image + IRD_loss_text)) + "  " + "\t"
            # loss_str += "SupCl_loss:" + " {}".format(torch.mean(supcl_loss_image + supcl_loss_text)) + "  " + "\t"
            # loss_str += "  " + "Time:" + "{}".format(
            #     time_elapsed) + "\t"
            # logger.info(loss_str)
            # retrivel(temp_eval_dataset, image_model_new, text_model_new, batch_size, cuda, bit, map, best_epoch, epoch,
            #          num_works, all_num, flag="old")
            # retrivel(eval_cluster_dataset, image_model_new, text_model_new, batch_size, cuda, bit, map, best_epoch,
            #          epoch, num_works, all_num, flag="new")
            # retrivel(all_eval_dataset, image_model, text_model, batch_size, cuda, bit, map, best_epoch, epoch,
            #          num_works, all_num, flag="all")
            # logger.info(
            #     "old_task_map:\nI2T_old:{}    T2I_old:{}\nI2T_new:{}    T2I_new:{}\nI2T_all:{}    T2I_all:{}".format(
            #         map["I2T_old"], map["T2I_old"], map["I2T"], map["T2I"], map["I2T_all"], map["T2I_all"]))
            # if (epoch + 1) % 5 == 0:
            #     logger.info(
            #         "Best_MAP:I2T_max_old:{}    T2I_max_old:{}    Best_epoch_old:{}\nI2T_max:{}     T2I_max:{}     Best_epoch:{}\nI2T_max_all:{}    T2I_max_all:{}    Best_epoch_all:{}".format(
            #             map["I2T_max_old"], map["T2I_max_old"], best_epoch["best_old"], map["I2T_max"], map["T2I_max"],
            #             best_epoch["best"], map["I2T_max_all"], map["T2I_max_all"], best_epoch["best_all"]))
            # sys.stdout.flush()
        logger.info(
            "Best_MAP:I2T_max_old:{}    T2I_max_old:{}    Best_epoch_old:{}\nI2T_max:{}     T2I_max:{}     Best_epoch:{}\nI2T_max_all:{}    T2I_max_all:{}    Best_epoch_all:{}".format(
                map["I2T_max_old"], map["T2I_max_old"], best_epoch["best_old"], map["I2T_max"], map["T2I_max"],
                best_epoch["best"], map["I2T_max_all"], map["T2I_max_all"], best_epoch["best_all"]))
        logger.info(f"The {i + 1} task has been learned！！！")
        temp_model_image1 = copy.deepcopy(image_model_new)
        all_index = list(set(all_index + cluster_index))
        temp_model_text1 = copy.deepcopy(text_model_new)
        temp_model_text2 = text_model_new
        temp_model_image2 = image_model_new
        temp_train = train_cluster_samples
        temp_eval = eval_cluster_samples
        i = i + 1
