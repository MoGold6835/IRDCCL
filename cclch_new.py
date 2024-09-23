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
from experiments.eval import retrivel, do_eval_extend
from loss import IRD_loss
from loss.supconloss import SupConLoss
from utils.data_augmentation import augmentation
from utils.data_mining import load_mat
from utils.get_alldata_coco import get_all_data


def cclch_new(train_data, query_data, database_data, database_all, image_model, text_model, lr, image_dir,
              batch_size, cuda, bit, max_map, best_epoch, epochs, logger, num_works, all_num, data_set):
    temp_model_image1 = copy.deepcopy(image_model)
    temp_model_text1 = copy.deepcopy(text_model)
    temp_model_image2 = image_model
    temp_model_text2 = text_model

    i = 0
    while i < len(train_data):
        image_model_new = temp_model_image1
        text_model_new = temp_model_text1
        image_model_old = temp_model_image2
        text_model_old = temp_model_text2
        optimizer_image = SGD(image_model_new.parameters(), lr["img"])
        optimizer_text = SGD(text_model_new.parameters(), lr["txt"])
        train_cluster_dataloader = DataLoader(train_data[i], batch_size=batch_size, shuffle=True, num_workers=num_works,
                                              pin_memory=True, drop_last=True)

        loss, IRD_loss_image, IRD_loss_text, supcl_loss_image, supcl_loss_text = 0, 0, 0, 0, 0
        current_temp = 0.2
        past_temp = 0.01
        # temp = 0.07

        image_model_old.eval()
        text_model_old.eval()
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

                logits1, logits2 = IRD_loss.ird_loss_current(image_features, text_features, current_temp=current_temp)
                IRD_loss_image, IRD_loss_text = IRD_loss.ird_loss_past(image_features1, text_features1,
                                                                       distill_power=1.0, past_temp=past_temp,
                                                                       logits1_image=logits1, logits1_text=logits2)

                supcl_loss_image = SupConLoss(image_features1, label)
                supcl_loss_text = SupConLoss(text_features1, label)
                loss = IRD_loss_image + IRD_loss_text + supcl_loss_image + supcl_loss_text
                loss = loss / (len(train_data[i]) * batch_size)

                optimizer_image.zero_grad()
                optimizer_text.zero_grad()
                loss.requires_grad_(True)
                loss.backward()
                optimizer_image.step()
                optimizer_text.step()

            time_elapsed = time.time() - since

            loss_str = "Epoch: [%3d/%3d], " % (epoch + 1, epochs)
            loss_str += "Loss:" + " {}".format(torch.mean(loss)) + "  " + "\t"
            loss_str += "IRD_loss:" + " {}".format(torch.mean(IRD_loss_image + IRD_loss_text)) + "  " + "\t"
            loss_str += "SupCl_loss:" + " {}".format(torch.mean(supcl_loss_image + supcl_loss_text)) + "  " + "\t"
            loss_str += "  " + "Time:" + "{}".format(
                time_elapsed) + "\t"
            logger.info(loss_str)

            do_eval_extend(query_data[0], query_data[1], database_data[0], database_data[1], database_all,
                           image_model_new, text_model_new, bit, cuda, batch_size, epoch, max_map, best_epoch,
                           logger, num_works, all_num)
            if (epoch + 1) % 5 == 0:
                logger.info("#" * 50)
                logger.info("#" * 50)
                logger.info("Old_Tasks:")
                logger.info("Best_MAP_q:  image2text_map_q:{}     text2image_map_q:{}     Best_epoch_q:  {}".format(
                    max_map["old"]["max_mapi2t_q"],
                    max_map["old"]["max_mapt2i_q"],
                    best_epoch["old"]["best_epoch_q"]))
                logger.info("Best_MAP_d:  image2text_map_d:{}     text2image_map_d:{}     Best_epoch_d:  {}".format(
                    max_map["old"]["max_mapi2t_d"],
                    max_map["old"]["max_mapt2i_d"],
                    best_epoch["old"]["best_epoch_d"]))
                logger.info("#" * 50)
                logger.info("New_Tasks:")
                logger.info("Best_MAP_q:  image2text_map_q:{}     text2image_map_q:{}     Best_epoch_q:  {}".format(
                    max_map["new"]["max_mapi2t_q"],
                    max_map["new"]["max_mapt2i_q"],
                    best_epoch["new"]["best_epoch_q"]))
                logger.info("Best_MAP_d:  image2text_map_d:{}     text2image_map_d:{}     Best_epoch_d:  {}".format(
                    max_map["new"]["max_mapi2t_d"],
                    max_map["new"]["max_mapt2i_d"],
                    best_epoch["new"]["best_epoch_d"]))
                logger.info("#" * 50)
                logger.info("All_Tasks:")
                logger.info("Best_MAP_q:  image2text_map_q:{}     text2image_map_q:{}     Best_epoch_q:  {}".format(
                    max_map["all"]["max_mapi2t_q"],
                    max_map["all"]["max_mapt2i_q"],
                    best_epoch["all"]["best_epoch_q"]))
                logger.info("Best_MAP_d:  image2text_map_d:{}     text2image_map_d:{}     Best_epoch_d:  {}".format(
                    max_map["all"]["max_mapi2t_d"],
                    max_map["all"]["max_mapt2i_d"],
                    best_epoch["all"]["best_epoch_d"]))
                logger.info("#" * 50)
                logger.info("#" * 50)
            sys.stdout.flush()
        logger.info("#" * 50)
        logger.info("#" * 50)
        temp_model_image1 = copy.deepcopy(image_model_new)
        temp_model_text1 = copy.deepcopy(text_model_new)
        temp_model_text2 = text_model_new
        temp_model_image2 = image_model_new
        i = i + 1
