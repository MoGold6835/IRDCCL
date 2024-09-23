import os
import sys
import time

import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from experiments.eval import do_eval_extend
from loss.objet_function import object_function
from loss.supconloss import SupConLoss
from utils.calculate import calc_neighbor
from utils.data_augmentation import augmentation


def co2l_s(image_model_old, text_model_old, image_model_new, text_model_new, replay_data, train_data,
           batch_size, cuda, optimizer_img, optimizer_txt, query_data_old, query_data, database_data_old,
           database_data_new, database_data, epochs, bit, max_map, best_epoch, logger, num_works, all_num,
           checkpoint_dir):
    train_dataloader = DataLoader(train_data, batch_size=batch_size, drop_last=True, shuffle=True,
                                  num_workers=num_works, pin_memory=True)
    train_num = len(all_data)
    logger.info("Starting train new data!!!")
    logger.info(
        f"Replay old data num:{len(replay_data)}  New data number:{len(train_data[1])}   Train data number:{len(all_data)}")

    current_temp = 0.2
    past_temp = 0.01

    train_L = torch.cat([train_data[1].get_all_label(), replay_data.get_all_label()], dim=0)
    ones = torch.ones(batch_size, 1)
    ones_ = torch.ones(train_num - batch_size, 1)
    new_train_image_buffer = torch.randn(train_num, bit)
    new_train_text_buffer = torch.randn(train_num, bit)
    B = torch.sign(new_train_text_buffer + new_train_image_buffer)

    if cuda:
        image_model_old = image_model_old.cuda()
        text_model_old = text_model_old.cuda()
        image_model_new = image_model_new.cuda()
        text_model_new = text_model_new.cuda()
        train_L = train_L.cuda()
        ones = ones.cuda()
        ones_ = ones_.cuda()
        new_train_image_buffer = new_train_image_buffer.cuda()
        new_train_text_buffer = new_train_text_buffer.cuda()
        B = B.cuda()

    scheduler_new_image = StepLR(optimizer_img, step_size=20, gamma=0.1)
    scheduler_new_text = StepLR(optimizer_txt, step_size=20, gamma=0.1)

    loss_dict = {"loss": 0, "IRD_loss_image": 0, "IRD_loss_text": 0, "supcl_loss_image": 0, "supcl_loss_text": 0,
                 "cmh_loss_image": 0, "cmh_loss_text": 0}

    image_model_new.train()
    text_model_new.train()
    image_model_old.eval()
    text_model_old.eval()

    for epoch in range(epochs):
        since = time.time()
        for data in train_dataloader:
            augmented_images, augmented_texts, augmented_labels = augmentation(data)
            index = data["index"].numpy()
            image = data["image"]
            label = data["label"]
            if cuda:
                image = image.cuda()
                label = label.cuda()
                augmented_images = augmented_images.cuda()
                augmented_labels = augmented_labels.cuda()
            image1 = torch.cat([image, augmented_images], dim=0)
            labels = torch.cat([label, augmented_labels], dim=0)

            optimizer_img.zero_grad()
            image_features = image_model_old(image1)
            image_features1 = image_model_new(image1)
            logits = image_model_new(image)
            new_train_image_buffer[index, :] = logits.data

            unupdated_ind = np.setdiff1d(range(train_num), index)
            S = calc_neighbor(label, train_L[index, :])
            theta_x = 1.0 / 2 * torch.matmul(logits.t(), new_train_text_buffer[index, :])
            logloss_1 = -torch.sum(torch.mul(S, theta_x) - torch.log(1.0 + torch.exp(theta_x)))
            quantization_1 = -torch.sum(torch.pow(B[index, :] - logits, 2))
            balance_1 = torch.sum(
                torch.pow(
                    torch.matmul(logits, ones) + torch.matmul(new_train_image_buffer[unupdated_ind, :].t(), ones_),
                    2))

            # logloss_1, quantization_1 = object_function(logits, label, new_train_image_buffer,
            #                                             new_train_text_buffer, index, train_num, train_L,
            #                                             ones, ones_, B)

            # ird_current
            image_sim = torch.div(torch.matmul(image_features1, image_features1.T), current_temp)
            logits_mask = torch.scatter(
                torch.ones_like(image_sim),
                1,
                torch.arange(image_sim.size(0)).view(-1, 1).cuda(non_blocking=True),
                0
            )
            logits_max1, _ = torch.max(image_sim * logits_mask, dim=1, keepdim=True)
            features1_sim = image_sim - logits_max1.detach()
            row_size = features1_sim.size(0)
            logits_current = torch.exp(features1_sim[logits_mask.bool()].view(row_size, -1)) / torch.exp(
                features1_sim[logits_mask.bool()].view(row_size, -1)).sum(dim=1, keepdim=True)

            # ird_past
            with torch.no_grad():
                past_image_sim = torch.div(torch.matmul(image_features, image_features.T), past_temp)
                logits_mask = torch.scatter(
                    torch.ones_like(past_image_sim),
                    1,
                    torch.arange(past_image_sim.size(0)).view(-1, 1).cuda(non_blocking=True),
                    0
                )
                logits_max1, _ = torch.max(past_image_sim * logits_mask, dim=1, keepdim=True)
                features1_sim = past_image_sim - logits_max1.detach()
                row_size = features1_sim.size(0)
                logits_past = torch.exp(features1_sim[logits_mask.bool()].view(row_size, -1)) / torch.exp(
                    features1_sim[logits_mask.bool()].view(row_size, -1)).sum(dim=1, keepdim=True)
            ird_loss_image = (-logits_past * torch.log(logits_current)).sum().mean()

            # Suploss
            sup_loss_image = SupConLoss(image_features1, labels)

            loss_image = logloss_1 + 1 * quantization_1 + 1 * balance_1 + ird_loss_image + sup_loss_image
            loss_image = loss_image / float(batch_size * all_num)

            loss_image.requires_grad_(True)
            loss_image.backward()
            optimizer_img.step()
            scheduler_new_image.step()
        for data in train_dataloader:
            augmented_images, augmented_texts, augmented_labels = augmentation(data)
            index = data["index"].numpy()
            text = data["text"]
            label = data["label"]
            if cuda:
                label = label.cuda()
                text = text.cuda()
                augmented_texts = augmented_texts.cuda()
                augmented_labels = augmented_labels.cuda()
            text1 = torch.cat([text, augmented_texts], dim=0)
            labels = torch.cat([label, augmented_labels], dim=0)

            optimizer_txt.zero_grad()
            text_features = text_model_old(text1)
            text_features1 = text_model_new(text1)
            logits2 = text_model_new(text)
            new_train_text_buffer[index, :] = logits2.data
            # cmh_loss
            unupdated_ind = np.setdiff1d(range(train_num), index)
            S = calc_neighbor(label, train_L[index, :])
            theta_y = 1.0 / 2 * torch.matmul(new_train_image_buffer[index, :].t(), logits2)
            logloss_2 = -torch.sum(torch.mul(S.t(), theta_y) - torch.log(1.0 + torch.exp(theta_y)))
            quantization_2 = torch.sum(torch.pow((new_train_text_buffer[index, :] - logits2), 2))
            balance_2 = torch.sum(
                torch.pow(
                    torch.matmul(logits2, ones) + torch.matmul(new_train_text_buffer[unupdated_ind, :].t(), ones_),
                    2))

            # logloss_2, quantization_2 = object_function(logits2, label, new_train_image_buffer,
            #                                             new_train_text_buffer, index, train_num, train_L,
            #                                             ones, ones_, B)

            # ird_current
            text_sim = torch.div(torch.matmul(text_features1, text_features1.T), current_temp)
            logits_mask = torch.scatter(
                torch.ones_like(text_sim),
                1,
                torch.arange(text_sim.size(0)).view(-1, 1).cuda(non_blocking=True),
                0
            )
            logits_max1, _ = torch.max(text_sim * logits_mask, dim=1, keepdim=True)
            features1_sim = text_sim - logits_max1.detach()
            row_size = features1_sim.size(0)
            logits_current = torch.exp(features1_sim[logits_mask.bool()].view(row_size, -1)) / torch.exp(
                features1_sim[logits_mask.bool()].view(row_size, -1)).sum(dim=1, keepdim=True)

            # ird_past
            with torch.no_grad():
                past_image_sim = torch.div(torch.matmul(text_features, text_features.T), past_temp)
            logits_mask = torch.scatter(
                torch.ones_like(past_image_sim),
                1,
                torch.arange(past_image_sim.size(0)).view(-1, 1).cuda(non_blocking=True),
                0
            )
            logits_max1, _ = torch.max(past_image_sim * logits_mask, dim=1, keepdim=True)
            features1_sim = past_image_sim - logits_max1.detach()
            row_size = features1_sim.size(0)
            logits_past = torch.exp(features1_sim[logits_mask.bool()].view(row_size, -1)) / torch.exp(
                features1_sim[logits_mask.bool()].view(row_size, -1)).sum(dim=1, keepdim=True)
            ird_loss_text = (-logits_past * torch.log(logits_current)).sum(1).mean()

            # SupConLoss
            sup_loss_text = SupConLoss(text_features1, labels)

            loss_text = logloss_2 + 1 * quantization_2 + 1 * balance_2 + ird_loss_text + sup_loss_text
            loss_text = loss_text / (batch_size * all_num)

            loss_text.requires_grad_(True)
            loss_text.backward()
            optimizer_txt.step()
            scheduler_new_text.step()
        B = torch.sign(new_train_text_buffer + new_train_image_buffer)
        time_elapsed = time.time() - since
        loss_dict["loss"] = loss_image + loss_text
        loss_dict["IRD_loss_image"] = ird_loss_image
        loss_dict["IRD_loss_text"] = ird_loss_text
        loss_dict["supcl_loss_image"] = sup_loss_image
        loss_dict["supcl_loss_text"] = sup_loss_text
        loss_dict["cmh_loss_image"] = logloss_1 + 0.01 * quantization_1
        loss_dict["cmh_loss_text"] = logloss_2 + 0.01 * quantization_2

        loss_str = "epoch: [%3d/%3d]:" % (epoch + 1, epochs)
        loss_str += "Loss:" + " {}".format(torch.mean(loss_dict["loss"])) + "  " + "\t"
        loss_str += "IRD_loss:" + " {}".format(
            torch.mean(loss_dict["IRD_loss_image"] + loss_dict["IRD_loss_text"])) + "  " + "\t"
        loss_str += "SupCl_loss:" + " {}".format(
            torch.mean(loss_dict["supcl_loss_image"] + loss_dict["supcl_loss_text"])) + "  " + "\t"
        loss_str += "cmh_loss:" + " {}".format(
            torch.mean(loss_dict["cmh_loss_image"] + loss_dict["cmh_loss_text"])) + "  " + "\t"
        loss_str += "  " + "Time:" + "{}".format(time_elapsed) + "\t"
        logger.info(loss_str)

        do_eval_extend(query_data_old, query_data, database_data_old, database_data_new, database_data,
                       image_model_new, text_model_new, bit, cuda, batch_size, epoch, max_map, best_epoch, logger,
                       num_works, all_num, extensible=True)
        since = time.time()
        str2 = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(since))
        image_model_new.save_dict(os.path.join(checkpoint_dir, str2 + "64bit_new_image_model.pth"))
        text_model_new.save_dict(os.path.join(checkpoint_dir, str2 + "64bit_new_text_model.pth"))
        if (epoch + 1) % 5 == 0:
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
    logger.info("训练过程结束！")
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
