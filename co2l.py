import itertools
import os.path
import sys
import time
import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from dataset.Dataset import MyDataset
from experiments.eval import do_eval_extend
from loss import IRD_loss
from loss.objet_function import object_function
from loss.supconloss import SupConLoss
from utils.data_augmentation import augmentation


def co2l(image_model_old, text_model_old, image_model_new, text_model_new, replay_data, train_data,
         batch_size, cuda, optimizer_img, optimizer_txt, query_data_old, query_data, database_data_old,
         database_data_new, database_data, epochs, bit, max_map, best_epoch, logger, num_works, all_num,
         checkpoint_dir):
    all_data = train_data + replay_data
    new_dataloader = DataLoader(all_data, batch_size=batch_size, drop_last=True, shuffle=True,
                                num_workers=num_works, pin_memory=True)
    train_num = len(train_data[1]) + len(replay_data)
    logger.info("Starting train new data!!!")
    logger.info(
        f"Replay old data num:{len(replay_data)}  New data number:{len(train_data[1])}   Train data number:{len(all_data)}")
    if cuda:
        image_model_old = image_model_old.cuda()
        text_model_old = text_model_old.cuda()
        image_model_new = image_model_new.cuda()
        text_model_new = text_model_new.cuda()
    image_model_old.eval()
    text_model_old.eval()
    image_model_new.train()
    text_model_new.train()

    current_temp = 0.2
    past_temp = 0.01

    train_L = torch.cat([train_data.get_all_label(), replay_data.get_all_label()], dim=0)

    ones = torch.ones(batch_size, 1)
    # ones = torch.cat([ones, ones], dim=0)
    ones_ = torch.ones(train_num - batch_size, 1)
    new_train_image_buffer = torch.randn(train_num, bit)
    new_train_text_buffer = torch.randn(train_num, bit)

    B = torch.sign(new_train_text_buffer + new_train_image_buffer)
    if cuda:
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

    # 进行第二轮训练，获取最新的模型
    for epoch in range(epochs):
        since = time.time()
        for data in new_dataloader:
            augmented_images, augmented_texts, augmented_labels = augmentation(data)
            index = data["index"].numpy()
            image1 = data["image"]
            text1 = data["text"]
            labels1 = data["label"]
            if cuda:
                image1 = image1.cuda()
                text1 = text1.cuda()
                labels1 = labels1.cuda()
                augmented_images = augmented_images.cuda()
                augmented_texts = augmented_texts.cuda()
                augmented_labels = augmented_labels.cuda()

            image = torch.cat([image1, augmented_images], dim=0)
            text = torch.cat([text1, augmented_texts], dim=0)
            labels = torch.cat([labels1, augmented_labels], dim=0)

            image_features = image_model_old(image)
            text_features = text_model_old(text)

            image_features1 = image_model_new(image)
            text_features1 = text_model_new(text)
            optimizer_img.zero_grad()
            optimizer_txt.zero_grad()
            logits1, logits2 = IRD_loss.ird_loss_current(image_features1, text_features1, current_temp=current_temp)

            IRD_loss_image, IRD_loss_text = IRD_loss.ird_loss_past(image_features, text_features, distill_power=1.0,
                                                                   past_temp=past_temp, logits1_image=logits1,
                                                                   logits1_text=logits2)
            supcl_loss_image = SupConLoss(image_features1, labels)
            supcl_loss_text = SupConLoss(text_features1, labels)

            image_logits = image_model_new(image1)
            text_logits = text_model_new(text1)
            new_train_text_buffer[index, :] = text_logits.data
            new_train_image_buffer[index, :] = image_logits.data
            # logloss_1, quantization_1 = object_function(image_logits, labels1, new_train_image_buffer,
            #                                             new_train_text_buffer, index, train_num, train_L,
            #                                             ones, ones_, B)
            # logloss_2, quantization_2 = object_function(text_logits, labels1, new_train_image_buffer,
            #                                             new_train_text_buffer, index, train_num, train_L,
            #                                             ones, ones_, B)
            # cmh_loss_image = logloss_1 + 0.01 * quantization_1
            # cmh_loss_text = logloss_2 + 0.01 * quantization_2

            loss = IRD_loss_image + IRD_loss_text + supcl_loss_image + supcl_loss_text + cmh_loss_image + cmh_loss_text
            loss = loss / (len(train_data) * batch_size)

            loss.requires_grad_(True)
            loss.backward()
            optimizer_img.step()
            optimizer_txt.step()
            scheduler_new_image.step()
            scheduler_new_text.step()
        B = torch.sign(new_train_image_buffer + new_train_text_buffer)
        time_elapsed = time.time() - since
        loss_dict["loss"] = loss
        loss_dict["IRD_loss_image"] = IRD_loss_image
        loss_dict["IRD_loss_text"] = IRD_loss_text
        loss_dict["supcl_loss_image"] = supcl_loss_image
        loss_dict["supcl_loss_text"] = supcl_loss_text
        loss_dict["cmh_loss_image"] = cmh_loss_image
        loss_dict["cmh_loss_text"] = cmh_loss_text
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
