import os
import sys
import time
import torch
from torch.optim import SGD
from torch.utils.data import DataLoader

from loss.ecmh_loss import object_function
from models import cnnf, MLP
from utils.calculate import calc_map_k
# from utils.plot_func import plot_function


def do_eval(query_data, database_data_all, image_model, text_model, predict_model, bit, cuda, batch_size, epoch,
            checkpoint_dir, logger, save_dir, max_map, best_epoch, num_works, all_num, extensible=False):
    global database_label, database_image_feature, database_text_feature, text2image, image2text
    query_dataloader = DataLoader(query_data, batch_size=batch_size, drop_last=True, shuffle=False,
                                  num_workers=num_works, pin_memory=False)
    database_dataloader = DataLoader(database_data_all, batch_size=batch_size, drop_last=True, shuffle=False,
                                     num_workers=num_works, pin_memory=False)
    query_image_buffer = query_text_buffer = torch.empty(len(query_data), bit, dtype=torch.float)
    database_image_buffer = database_text_buffer = torch.empty(len(database_data_all), bit, dtype=torch.float)
    query_label_buffer, database_label_buffer = torch.empty(len(query_data), all_num,
                                                            dtype=torch.float), torch.empty(
        len(database_data_all), all_num, dtype=torch.float)
    since = time.time()
    if cuda:
        query_text_buffer = query_text_buffer.cuda()
        query_image_buffer = query_image_buffer.cuda()
        database_text_buffer = database_text_buffer.cuda()
        database_image_buffer = database_image_buffer.cuda()
        query_label_buffer = query_label_buffer.cuda()
        database_label_buffer = database_label_buffer.cuda()
    image_model.eval()
    text_model.eval()
    with torch.no_grad():
        for database_data in database_dataloader:
            index1 = database_data["index"].numpy()
            image1 = database_data["image"]
            text1 = database_data["text"]
            label1 = database_data["label"]
            if cuda:
                image1 = image1.cuda()
                text1 = text1.cuda()
                label1 = label1.cuda()
            image_hash1 = torch.sign(image_model(image1))
            text_hash1 = torch.sign(text_model(text1))
            database_image_buffer[index1, :] = image_hash1
            database_text_buffer[index1, :] = text_hash1
            database_label_buffer[index1, :] = label1
        for query_data in query_dataloader:
            index2 = query_data["index"].numpy()
            image2 = query_data["image"]
            text2 = query_data["text"]
            label2 = query_data["label"]
            if cuda is True:
                image2 = image2.cuda()
                text2 = text2.cuda()
                label2 = label2.cuda()
            image_hash2 = torch.sign(image_model(image2))
            text_hash2 = torch.sign(text_model(text2))
            query_image_buffer[index2, :] = image_hash2
            query_text_buffer[index2, :] = text_hash2
            query_label_buffer[index2, :] = label2

    retrieval_image = torch.cat([query_image_buffer, database_image_buffer], 0)
    retrieval_text = torch.cat([query_text_buffer, database_text_buffer], 0)
    retrieval_label = torch.cat([query_label_buffer, database_label_buffer], 0)

    image2image_d = calc_map_k(query_image_buffer, retrieval_image, query_label_buffer, retrieval_label)
    text2text_d = calc_map_k(query_text_buffer, retrieval_text, query_label_buffer, retrieval_label)
    image2text_d = calc_map_k(query_image_buffer, retrieval_text, query_label_buffer, retrieval_label)
    text2image_d = calc_map_k(query_text_buffer, retrieval_image, query_label_buffer, retrieval_label)

    image2image_q = calc_map_k(query_image_buffer, query_image_buffer, query_label_buffer,
                               query_label_buffer)
    text2text_q = calc_map_k(query_text_buffer, query_text_buffer, query_label_buffer, query_label_buffer)
    image2text_q = calc_map_k(query_image_buffer, query_text_buffer, query_label_buffer,
                              query_label_buffer)
    text2image_q = calc_map_k(query_text_buffer, query_image_buffer, query_label_buffer,
                              query_label_buffer)
    max_map["old"]["max_mapi2t_q"] = max(image2text_q, max_map["old"]["max_mapi2t_q"])
    max_map["old"]["max_mapi2i_q"] = max(image2image_q, max_map["old"]["max_mapi2i_q"])
    max_map["old"]["max_mapt2i_q"] = max(text2image_q, max_map["old"]["max_mapt2i_q"])
    max_map["old"]["max_mapt2t_q"] = max(text2text_q, max_map["old"]["max_mapt2t_q"])
    max_map["old"]["max_mapi2t_d"] = max(image2text_d, max_map["old"]["max_mapi2t_d"])
    max_map["old"]["max_mapi2i_d"] = max(image2image_d, max_map["old"]["max_mapi2i_d"])
    max_map["old"]["max_mapt2i_d"] = max(text2image_d, max_map["old"]["max_mapt2i_d"])
    max_map["old"]["max_mapt2t_d"] = max(text2text_d, max_map["old"]["max_mapt2t_d"])

    since2 = time.time()
    str2 = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(since2))
    if image2text_q + text2image_q >= max_map["old"]["max_mapt2i_q"] + max_map["old"]["max_mapi2t_q"]:
        max_map["old"]["max_mapt2i_q"] = text2image_q
        max_map["old"]["max_mapi2t_q"] = image2text_q
        best_epoch["old"]["best_epoch_q"] = epoch
        if extensible is False:
            image_model.save_dict(
                os.path.join(checkpoint_dir, str(bit) + "-" + "CNNF_query" + ".pth"))
            text_model.save_dict(
                os.path.join(checkpoint_dir, str(bit) + "-" + "MLP_query" + ".pth"))
        else:
            image_model.save_dict(
                os.path.join(checkpoint_dir, str(bit) + "-" + "CNNF_extensible_query" + ".pth"))
            text_model.save_dict(
                os.path.join(checkpoint_dir, str(bit) + "-" + "MLP_extensible_query" + ".pth"))
    if image2text_d + text2image_d >= max_map["old"]["max_mapt2i_d"] + max_map["old"]["max_mapi2t_d"]:
        max_map["old"]["max_mapt2i_d"] = text2image_d
        max_map["old"]["max_mapi2t_d"] = image2text_d
        best_epoch["old"]["best_epoch_d"] = epoch
        if extensible is False:
            image_model.save_dict(
                os.path.join(checkpoint_dir, str(bit) + "-" + "CNNF_database" + ".pth"))
            text_model.save_dict(
                os.path.join(checkpoint_dir, str(bit) + "-" + "MLP_database" + ".pth"))
        else:
            image_model.save_dict(
                os.path.join(checkpoint_dir, str(bit) + "-" + "CNNF_extensible_database" + ".pth"))
            text_model.save_dict(
                os.path.join(checkpoint_dir, str(bit) + "-" + "MLP_extensible_database" + ".pth"))
    image2text_q_ave = (image2image_q + image2text_q) / 2
    text2image_q_ave = (text2text_q + text2image_q) / 2
    image2text_d_ave = (image2text_d + image2image_d) / 2
    text2image_d_ave = (text2text_d + text2image_d) / 2
    time_elapsed_1 = time.time() - since
    # if extensible:
    #     plot_function(query_image_buffer, query_text_buffer, retrieval_image, retrieval_text,
    #                   query_label_buffer, retrieval_label, save_dir)
    logger.info("image2text_map_d:{}     text2image_map_d:{}     ValidateTime:{}".format(image2text_d_ave,
                                                                                         text2image_d_ave,
                                                                                         time_elapsed_1))
    logger.info("image2text_map_q:{}     text2image_map_q:{}     ValidateTime:{}".format(image2text_q_ave,
                                                                                         text2image_q_ave,
                                                                                         time_elapsed_1))


def do_eval_extend(query_data_old, query_data, database_data_old, database_data_new, database_data_all, image_model,
                   text_model, bit, cuda, batch_size, epoch, max_map, best_epoch, logger, num_works, all_num,
                   extensible=False):
    global database_label, database_image_feature, database_text_feature, text2image, image2text
    query_dataloader = DataLoader(query_data, batch_size=batch_size, drop_last=True, shuffle=False,
                                  num_workers=num_works, pin_memory=False)
    database_dataloader = DataLoader(database_data_all, batch_size=batch_size, drop_last=True, shuffle=False,
                                     num_workers=num_works, pin_memory=False)
    old_query_dataloader = DataLoader(query_data_old, batch_size=batch_size, drop_last=True, shuffle=False,
                                      num_workers=num_works, pin_memory=False)
    query_image_buffer = query_text_buffer = torch.empty(len(query_data), bit, dtype=torch.float)
    old_query_image_buffer = old_query_text_buffer = torch.empty(len(query_data_old), bit, dtype=torch.float)
    database_image_buffer = database_text_buffer = torch.empty(len(database_data_all), bit, dtype=torch.float)
    query_label_buffer, database_label_buffer = torch.empty(len(query_data), all_num, dtype=torch.float), torch.empty(
        len(database_data_all), all_num, dtype=torch.float)
    old_query_label_buffer, old_database_label_buffer, new_database_label_buffer = torch.empty(len(query_data_old),
        all_num,dtype=torch.float), torch.empty(len(database_data_old), all_num, dtype=torch.float), torch.empty(len(database_data_new), all_num,dtype=torch.float)
    since = time.time()
    if cuda:
        old_query_text_buffer = old_query_text_buffer.cuda()
        old_query_image_buffer = old_query_image_buffer.cuda()
        query_text_buffer = query_text_buffer.cuda()
        query_image_buffer = query_image_buffer.cuda()
        database_text_buffer = database_text_buffer.cuda()
        database_image_buffer = database_image_buffer.cuda()
        query_label_buffer = query_label_buffer.cuda()
        database_label_buffer = database_label_buffer.cuda()
        old_query_label_buffer = old_query_label_buffer.cuda()
    image_model.eval()
    text_model.eval()
    with torch.no_grad():
        for database_data in database_dataloader:
            index1 = database_data["index"].numpy()
            image1 = database_data["image"]
            text1 = database_data["text"]
            label1 = database_data["label"]
            if cuda:
                image1 = image1.cuda()
                text1 = text1.cuda()
                label1 = label1.cuda()
            image_hash1 = torch.sign(image_model(image1))
            text_hash1 = torch.sign(text_model(text1))
            database_image_buffer[index1, :] = image_hash1
            database_text_buffer[index1, :] = text_hash1
            database_label_buffer[index1, :] = label1
        for query_data in query_dataloader:
            index2 = query_data["index"].numpy()
            image2 = query_data["image"]
            text2 = query_data["text"]
            label2 = query_data["label"]
            if cuda is True:
                image2 = image2.cuda()
                text2 = text2.cuda()
                label2 = label2.cuda()
            image_hash2 = torch.sign(image_model(image2))
            text_hash2 = torch.sign(text_model(text2))
            query_image_buffer[index2, :] = image_hash2
            query_text_buffer[index2, :] = text_hash2
            query_label_buffer[index2, :] = label2
        for query_data in old_query_dataloader:
            index4 = query_data["index"].numpy()
            image4 = query_data["image"]
            text4 = query_data["text"]
            label4 = query_data["label"]
            if cuda is True:
                image4 = image4.cuda()
                text4 = text4.cuda()
                label4 = label4.cuda()
            image_hash4 = torch.sign(image_model(image4))
            text_hash4 = torch.sign(text_model(text4))
            old_query_image_buffer[index4, :] = image_hash4
            old_query_text_buffer[index4, :] = text_hash4
            old_query_label_buffer[index4, :] = label4
        do_retrival(old_query_image_buffer, old_query_text_buffer, old_query_label_buffer,
                    database_image_buffer, database_text_buffer, database_label_buffer,
                    since, epoch, max_map, best_epoch, logger, flag="Old Tasks")
        do_retrival(query_image_buffer, query_text_buffer, query_label_buffer, database_image_buffer,
                    database_text_buffer, database_label_buffer, since,
                    epoch, max_map, best_epoch, logger, flag="New Tasks")
        all_query_image_buffer = torch.cat([old_query_image_buffer, query_image_buffer], 0)
        all_query_text_buffer = torch.cat([old_query_text_buffer, query_text_buffer], 0)
        all_query_label_buffer = torch.cat([old_query_label_buffer, query_label_buffer], 0)
        do_retrival(all_query_image_buffer, all_query_text_buffer, all_query_label_buffer,
                    database_image_buffer, database_text_buffer, database_label_buffer,
                    since, epoch, max_map, best_epoch, logger, flag="All Tasks")


def do_retrival(query_image_buffer, query_text_buffer, query_label_buffer, database_image_buffer, database_text_buffer,
                database_label_buffer, since, epoch, max_map, best_epoch, logger, flag=""):
    # retrieval_image = torch.cat([query_image_buffer, database_image_buffer], 0)
    # retrieval_text = torch.cat([query_text_buffer, database_text_buffer], 0)
    # retrieval_label = torch.cat([query_label_buffer, database_label_buffer], 0)
    retrieval_image = database_image_buffer
    retrieval_text = database_text_buffer
    retrieval_label = database_label_buffer

    image2image_d = calc_map_k(query_image_buffer, retrieval_image, query_label_buffer, retrieval_label)
    text2text_d = calc_map_k(query_text_buffer, retrieval_text, query_label_buffer, retrieval_label)
    image2text_d = calc_map_k(query_image_buffer, retrieval_text, query_label_buffer, retrieval_label)
    text2image_d = calc_map_k(query_text_buffer, retrieval_image, query_label_buffer, retrieval_label)

    image2image_q = calc_map_k(query_image_buffer, query_image_buffer, query_label_buffer, query_label_buffer)
    text2text_q = calc_map_k(query_text_buffer, query_text_buffer, query_label_buffer, query_label_buffer)
    image2text_q = calc_map_k(query_image_buffer, query_text_buffer, query_label_buffer, query_label_buffer)
    text2image_q = calc_map_k(query_text_buffer, query_image_buffer, query_label_buffer, query_label_buffer)

    if flag == "Old Tasks":
        max_map["old"]["max_mapi2t_q"] = max(image2text_q, max_map["old"]["max_mapi2t_q"])
        max_map["old"]["max_mapi2i_q"] = max(image2image_q, max_map["old"]["max_mapi2i_q"])
        max_map["old"]["max_mapt2i_q"] = max(text2image_q, max_map["old"]["max_mapt2i_q"])
        max_map["old"]["max_mapt2t_q"] = max(text2text_q, max_map["old"]["max_mapt2t_q"])
        max_map["old"]["max_mapi2t_d"] = max(image2text_d, max_map["old"]["max_mapi2t_d"])
        max_map["old"]["max_mapi2i_d"] = max(image2image_d, max_map["old"]["max_mapi2i_d"])
        max_map["old"]["max_mapt2i_d"] = max(text2image_d, max_map["old"]["max_mapt2i_d"])
        max_map["old"]["max_mapt2t_d"] = max(text2text_d, max_map["old"]["max_mapt2t_d"])
        if image2text_q + text2image_q >= max_map["old"]["max_mapt2i_q"] + max_map["old"]["max_mapi2t_q"]:
            max_map["old"]["max_mapt2i_q"] = text2image_q
            max_map["old"]["max_mapi2t_q"] = image2text_q
            best_epoch["old"]["best_epoch_q"] = epoch
        if image2text_d + text2image_d >= max_map["old"]["max_mapt2i_d"] + max_map["old"]["max_mapi2t_d"]:
            max_map["old"]["max_mapt2i_d"] = text2image_d
            max_map["old"]["max_mapi2t_d"] = image2text_d
            best_epoch["old"]["best_epoch_d"] = epoch
        image2text_q_ave = (image2image_q + image2text_q) / 2
        text2image_q_ave = (text2text_q + text2image_q) / 2
        image2text_d_ave = (image2text_d + image2image_d) / 2
        text2image_d_ave = (text2text_d + text2image_d) / 2
        time_elapsed_1 = time.time() - since
        logger.info(f"The {flag} MAP as follows:")
        logger.info("image2text_map_d:{}     text2image_map_d:{}     ValidateTime:{}".format(image2text_d_ave,
                                                                                             text2image_d_ave,
                                                                                             time_elapsed_1))
        logger.info("image2text_map_q:{}     text2image_map_q:{}     ValidateTime:{}".format(image2text_q_ave,
                                                                                             text2image_q_ave,
                                                                                             time_elapsed_1))
    elif flag == "New Tasks":
        max_map["new"]["max_mapi2t_q"] = max(image2text_q, max_map["new"]["max_mapi2t_q"])
        max_map["new"]["max_mapi2i_q"] = max(image2image_q, max_map["new"]["max_mapi2i_q"])
        max_map["new"]["max_mapt2i_q"] = max(text2image_q, max_map["new"]["max_mapt2i_q"])
        max_map["new"]["max_mapt2t_q"] = max(text2text_q, max_map["new"]["max_mapt2t_q"])
        max_map["new"]["max_mapi2t_d"] = max(image2text_d, max_map["new"]["max_mapi2t_d"])
        max_map["new"]["max_mapi2i_d"] = max(image2image_d, max_map["new"]["max_mapi2i_d"])
        max_map["new"]["max_mapt2i_d"] = max(text2image_d, max_map["new"]["max_mapt2i_d"])
        max_map["new"]["max_mapt2t_d"] = max(text2text_d, max_map["new"]["max_mapt2t_d"])
        if image2text_q + text2image_q >= max_map["new"]["max_mapt2i_q"] + max_map["new"]["max_mapi2t_q"]:
            max_map["new"]["max_mapt2i_q"] = text2image_q
            max_map["new"]["max_mapi2t_q"] = image2text_q
            best_epoch["new"]["best_epoch_q"] = epoch
        if image2text_d + text2image_d >= max_map["new"]["max_mapt2i_d"] + max_map["new"]["max_mapi2t_d"]:
            max_map["new"]["max_mapt2i_d"] = text2image_d
            max_map["new"]["max_mapi2t_d"] = image2text_d
            best_epoch["new"]["best_epoch_d"] = epoch
        image2text_q_ave = (image2image_q + image2text_q) / 2
        text2image_q_ave = (text2text_q + text2image_q) / 2
        image2text_d_ave = (image2text_d + image2image_d) / 2
        text2image_d_ave = (text2text_d + text2image_d) / 2
        time_elapsed_1 = time.time() - since
        logger.info(f"The {flag} MAP as follows:")
        logger.info("image2text_map_d:{}     text2image_map_d:{}     ValidateTime:{}".format(image2text_d_ave,
                                                                                             text2image_d_ave,
                                                                                             time_elapsed_1))
        logger.info("image2text_map_q:{}     text2image_map_q:{}     ValidateTime:{}".format(image2text_q_ave,
                                                                                             text2image_q_ave,
                                                                                             time_elapsed_1))
    elif flag == "All Tasks":
        max_map["all"]["max_mapi2t_q"] = max(image2text_q, max_map["all"]["max_mapi2t_q"])
        max_map["all"]["max_mapi2i_q"] = max(image2image_q, max_map["all"]["max_mapi2i_q"])
        max_map["all"]["max_mapt2i_q"] = max(text2image_q, max_map["all"]["max_mapt2i_q"])
        max_map["all"]["max_mapt2t_q"] = max(text2text_q, max_map["all"]["max_mapt2t_q"])
        max_map["all"]["max_mapi2t_d"] = max(image2text_d, max_map["all"]["max_mapi2t_d"])
        max_map["all"]["max_mapi2i_d"] = max(image2image_d, max_map["all"]["max_mapi2i_d"])
        max_map["all"]["max_mapt2i_d"] = max(text2image_d, max_map["all"]["max_mapt2i_d"])
        max_map["all"]["max_mapt2t_d"] = max(text2text_d, max_map["all"]["max_mapt2t_d"])
        if image2text_q + text2image_q >= max_map["all"]["max_mapt2i_q"] + max_map["all"]["max_mapi2t_q"]:
            max_map["all"]["max_mapt2i_q"] = text2image_q
            max_map["all"]["max_mapi2t_q"] = image2text_q
            best_epoch["all"]["best_epoch_q"] = epoch
        if image2text_d + text2image_d >= max_map["all"]["max_mapt2i_d"] + max_map["all"]["max_mapi2t_d"]:
            max_map["all"]["max_mapt2i_d"] = text2image_d
            max_map["all"]["max_mapi2t_d"] = image2text_d
            best_epoch["all"]["best_epoch_d"] = epoch
        image2text_q_ave = (image2image_q + image2text_q) / 2
        text2image_q_ave = (text2text_q + text2image_q) / 2
        image2text_d_ave = (image2text_d + image2image_d) / 2
        text2image_d_ave = (text2text_d + text2image_d) / 2
        time_elapsed_1 = time.time() - since
        logger.info(f"The {flag} MAP as follows:")
        logger.info("image2text_map_d:{}     text2image_map_d:{}     ValidateTime:{}".format(image2text_d_ave,
                                                                                             text2image_d_ave,
                                                                                             time_elapsed_1))
        logger.info("image2text_map_q:{}     text2image_map_q:{}     ValidateTime:{}".format(image2text_q_ave,
                                                                                             text2image_q_ave,
                                                                                             time_elapsed_1))


def do_extensible(query_data_old, query_data, database_data_old, database_data_new, database_data, image_model_old,
                  text_model_old, checkpoint_dir, logger, batch_size, cuda, bit, epoch, train_extensible_data, max_map,
                  best_epoch, lr, beta_1, num_works, all_num):
    logger.info("Initializing model with optimal weights.")
    saved_model_cnnf = os.path.join(checkpoint_dir, str(bit) + "-" + "CNNF_query" + ".pth")
    save_model_mlp = os.path.join(checkpoint_dir, str(bit) + "-" + "MLP_query" + ".pth")
    image_model_old.load_state_dict(torch.load(saved_model_cnnf))
    text_model_old.load_state_dict(torch.load(save_model_mlp))
    train_extensible_dataloader = DataLoader(train_extensible_data, batch_size=batch_size, drop_last=True,
                                             shuffle=True, num_workers=num_works, pin_memory=True)
    logger.info("Generating new data features using old neural networks......")
    n1o_image_feature_buffer = torch.empty(20015, bit).cuda()
    n1o_text_feature_buffer = torch.empty(20015, bit).cuda()
    new_hash_buffer = torch.empty(20015, bit).cuda()
    if cuda:
        text_model_old = text_model_old.cuda()
        image_model_old = image_model_old.cuda()
    text_model_old.eval()
    image_model_old.eval()
    with torch.no_grad():
        for data in train_extensible_dataloader:
            index = data["index"].numpy()
            image = data["image"]
            text = data["text"]
            if cuda is True:
                image = image.cuda()
                text = text.cuda()
            # 使用旧数据训练出来的参数来生成新数据的特征
            n1o_image_features = image_model_old(image)
            n1o_text_features = text_model_old(text)
            n1o_image_feature_buffer[index, :] = n1o_image_features
            n1o_text_feature_buffer[index, :] = n1o_text_features
    str1 = """
    Successfully generated new data features using old neural networks！！！
    Randomly initializing model weights......
    """
    logger.info(str1)
    image_model_new = cnnf.get_cnnf(bit, pretrain=False)
    text_model_new = MLP.MLP(1386, bit, leakRelu=False)
    if cuda:
        image_model_new = image_model_new.cuda()
        text_model_new = text_model_new.cuda()
    opt_image = SGD(image_model_new.parameters(), lr=lr['img'])
    opt_text = SGD(text_model_new.parameters(), lr=lr['txt'])
    str2 = """
    Successfully initialized model weights randomly！！！
    Start extended data training......
    """
    logger.info(str2)
    for epochs in range(epoch):
        print("Epoch{}:".format(epochs + 1))
        since = time.time()
        image_model_new.eval()
        text_model_new.eval()
        with torch.no_grad():
            for train_data in train_extensible_dataloader:
                new_train_index1 = train_data["index"].numpy()
                new_train_image1 = train_data["image"]
                new_train_text1 = train_data["text"]
                if cuda:
                    new_train_image1 = new_train_image1.cuda()
                    new_train_text1 = new_train_text1.cuda()
                # 使用旧数据训练出来的参数来
                image_features = image_model_new(new_train_image1)
                text_features = text_model_new(new_train_text1)
                new_hash_buffer[new_train_index1, :] = torch.sign(image_features + text_features)
        image_model_new.train()
        text_model_new.train()
        for data in train_extensible_dataloader:
            opt_image.zero_grad()
            opt_text.zero_grad()
            new_train_index2 = data["index"]
            new_train_image2 = data["image"]
            new_train_text2 = data["text"]
            new_train_label2 = data["label"]
            if cuda:
                new_train_image2 = new_train_image2.cuda()
                new_train_text2 = new_train_text2.cuda()
                new_train_label2 = new_train_label2.cuda()
                new_train_index2 = new_train_index2.cuda()
            image_results = image_model_new(new_train_image2)
            text_results = text_model_new(new_train_text2)
            n1o_image_features_temp = torch.index_select(n1o_image_feature_buffer, 0, new_train_index2)
            n1o_text_features_temp = torch.index_select(n1o_text_feature_buffer, 0, new_train_index2)
            new_hash_temp = torch.index_select(new_hash_buffer, 0, new_train_index2)
            sim_new = torch.cdist(image_results, text_results).cuda()
            loss, dloss, closs, hloss = object_function(sim_new, image_results, n1o_image_features_temp,
                                                        text_results, n1o_text_features_temp,
                                                        new_train_label2,
                                                        new_hash_temp, beta_1, batch_size, bit)
            loss.requires_grad_(True)
            loss.backward()
            opt_image.step()
            opt_text.step()
        time_elapsed = time.time() - since
        loss_str = "epoch: [%3d/%3d], " % (epochs + 1, epoch)
        loss_str += "Loss:" + " {}".format(torch.mean(loss)) + "  " + "\t"
        loss_str += "Loss_d:" + " {}".format(torch.mean(dloss)) + "  " + "\t"
        loss_str += "Loss_c:" + " {}".format(torch.mean(closs)) + "  " + "\t"
        loss_str += "Loss_h:" + " {}".format(torch.mean(hloss)) + "  " + "Time:" + "{}".format(
            time_elapsed) + "\t"
        logger.info(loss_str)
        do_eval_extend(query_data_old, query_data, database_data_old, database_data_new, database_data,
                       image_model_new, text_model_new, bit, cuda, batch_size, epoch, max_map, best_epoch, logger,
                       num_works, all_num, extensible=True)
        sys.stdout.flush()
    logger.info("End of training process！！！")
    logger.info(
        "Best_MAP_q:  image2text_map_q:{}     text2image_map_q:{}     Best_epoch_q:  {}".format(
            max_map["old"]["max_mapi2t_q"],
            max_map["old"]["max_mapt2i_q"],
            best_epoch["old"]["best_epoch_q"] + 1))
    logger.info(
        "Best_MAP_d:  image2text_map_d:{}     text2image_map_d:{}     Best_epoch_q:  {}".format(
            max_map["old"]["max_mapi2t_d"],
            max_map["old"]["max_mapt2i_d"],
            best_epoch["old"]["best_epoch_d"] + 1))


def retrivel(query_data, image_model, text_model, batch_size, cuda, bit, map, best_epoch, epoch, num_works, all_num,
             flag):
    query_dataloader = DataLoader(query_data, batch_size, shuffle=True, pin_memory=True, drop_last=True,
                                  num_workers=num_works)
    image_hash_buffer = torch.zeros(len(query_data), bit)
    text_hash_buffer = torch.zeros(len(query_data), bit)
    label_buffer = torch.zeros(len(query_data), all_num)
    if cuda:
        image_hash_buffer = image_hash_buffer.cuda()
        text_hash_buffer = text_hash_buffer.cuda()
        label_buffer = label_buffer.cuda()
    image_model.eval()
    text_model.eval()
    with torch.no_grad():
        for data in query_dataloader:
            image = data["image"]
            text = data["text"]
            label = data["label"]
            index = data["index"].numpy()
            if cuda:
                image = image.cuda()
                text = text.cuda()
                label = label.cuda()
            image_results = image_model(image)
            text_results = text_model(text)
            image_hash = torch.sign(image_results)
            text_hash = torch.sign(text_results)
            image_hash_buffer[index, :] = image_hash
            text_hash_buffer[index, :] = text_hash
            label_buffer[index, :] = label
    image2image = calc_map_k(image_hash_buffer, image_hash_buffer, label_buffer, label_buffer).cuda()
    image2text = calc_map_k(image_hash_buffer, text_hash_buffer, label_buffer, label_buffer).cuda()
    text2text = calc_map_k(text_hash_buffer, text_hash_buffer, label_buffer, label_buffer).cuda()
    text2image = calc_map_k(text_hash_buffer, image_hash_buffer, label_buffer, label_buffer).cuda()

    ave_image2text = (image2image + image2text) / 2
    ave_text2image = (text2text + text2image) / 2
    if flag == "old":
        if ave_image2text + ave_text2image > map["I2T_old"] + map["T2I_old"]:
            map["I2T_max_old"] = ave_image2text
            map["T2I_max_old"] = ave_text2image
            best_epoch["best_old"] = epoch + 1
        map["I2T_old"] = ave_image2text
        map["T2I_old"] = ave_text2image
    elif flag == "new":
        if ave_image2text + ave_text2image > map["I2T"] + map["T2I"]:
            map["I2T_max"] = ave_image2text
            map["T2I_max"] = ave_text2image
            best_epoch["best"] = epoch + 1
        map["I2T"] = ave_image2text
        map["T2I"] = ave_text2image
    elif flag == "all":
        if ave_image2text + ave_text2image > map["I2T_all"] + map["T2I_all"]:
            map["I2T_max_all"] = ave_image2text
            map["T2I_max_all"] = ave_text2image
            best_epoch["best_all"] = epoch + 1
        map["I2T_all"] = ave_image2text
        map["T2I_all"] = ave_text2image