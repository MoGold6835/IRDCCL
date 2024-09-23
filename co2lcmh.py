import os.path
import sys
import time
import torch
from torch.utils.data import DataLoader
from experiments.eval import do_eval_extend
from loss import IRD_loss
from loss.cmh_loss import CMH_loss
from loss.supconloss import SupConLoss
from utils.data_augmentation import augmentation


def co2lcmh(image_model_old, text_model_old, image_model_new, text_model_new, replay_data, train_data,
            batch_size, cuda, optimizer_img, optimizer_txt, query_data_old, query_data, database_data_old,
            database_data_new, database_data, epochs, bit, max_map, best_epoch, logger, num_works, all_num,
            checkpoint_dir):
    replay_data = replay_data + train_data
    replay_dataloader = DataLoader(replay_data, batch_size=batch_size, drop_last=True,
                                   shuffle=True, num_workers=num_works, pin_memory=True)
    if cuda:
        image_model_old = image_model_old.cuda()
        text_model_old = text_model_old.cuda()
        image_model_new = image_model_new.cuda()
        text_model_new = text_model_new.cuda()

    current_temp = 0.2
    past_temp = 0.01

    image_model_old.eval()
    text_model_old.eval()
    image_model_new.train()
    text_model_new.train()

    loss, IRD_loss_image, IRD_loss_text, supcl_loss_image, supcl_loss_text = 0, 0, 0, 0, 0

    # 进行第二轮训练，获取最新的模型
    for epoch in range(epochs):
        since = time.time()
        for data in replay_dataloader:
            augmented_images, augmented_texts, augmented_labels = augmentation(data)
            image = data["image"]
            text = data["text"]
            labels = data["label"]
            if cuda:
                image = image.cuda()
                text = text.cuda()
                labels = labels.cuda()
                augmented_images = augmented_images.cuda()
                augmented_texts = augmented_texts.cuda()
                augmented_labels = augmented_labels.cuda()
            image = torch.cat([image, augmented_images], dim=0)
            text = torch.cat([text, augmented_texts], dim=0)
            labels = torch.cat([labels, augmented_labels], dim=0)

            image_features = image_model_old(image)
            text_features = text_model_old(text)
            image_features1 = image_model_new(image)
            text_features1 = text_model_new(text)

            logits1, logits2 = IRD_loss.ird_loss_current(image_features, text_features, current_temp=current_temp)
            IRD_loss_image, IRD_loss_text = IRD_loss.ird_loss_past(image_features1, text_features1,
                                                                   distill_power=1.0, past_temp=past_temp,
                                                                   logits1_image=logits1, logits1_text=logits2)
            supcl_loss_image = SupConLoss(image_features1, labels)
            supcl_loss_text = SupConLoss(text_features1, labels)

            loss = IRD_loss_image + IRD_loss_text + supcl_loss_image + supcl_loss_text
            loss = loss / (len(train_data[1]) * batch_size)

            optimizer_img.zero_grad()
            optimizer_txt.zero_grad()
            loss.requires_grad_(True)
            loss.backward()
            optimizer_img.step()
            optimizer_txt.step()

        time_elapsed = time.time() - since

        loss_str = "epoch: [%3d/%3d], " % (epoch + 1, epochs)
        loss_str += "Loss:" + " {}".format(torch.mean(loss)) + "  " + "\t"
        loss_str += "IRD_loss:" + " {}".format(torch.mean(IRD_loss_image + IRD_loss_text)) + "  " + "\t"
        loss_str += "SupCl_loss:" + " {}".format(torch.mean(supcl_loss_image + supcl_loss_text)) + "  " + "\t"
        loss_str += "  " + "Time:" + "{}".format(
            time_elapsed) + "\t"
        logger.info(loss_str)
        do_eval_extend(query_data_old, query_data, database_data_old, database_data_new, database_data,
                       image_model_new, text_model_new, bit, cuda, batch_size, epoch, max_map, best_epoch, logger,
                       num_works, all_num, extensible=True)
        image_model_new.save_dict(os.path.join(checkpoint_dir, "64bit_new_image_model.pth"))
        text_model_new.save_dict(os.path.join(checkpoint_dir, "64bit_new_text_model.pth"))
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
