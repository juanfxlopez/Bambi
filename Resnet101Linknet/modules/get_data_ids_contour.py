from os import listdir
import pandas as pd

def get_ids_in_csv(inputs, labels, inter, contour, destination_name_inputs, destination_name_labels, destination_name_inter,destination_name_contour):
    list_ids_train_input = [f for f in listdir(inputs)]
    list_ids_train_labels = [f for f in listdir(labels)]
    list_ids_train_inter = [f for f in listdir(inter)]
    list_ids_train_contour = [f for f in listdir(contour)]

    list_ids_train_input = sorted(list_ids_train_input)
    list_ids_train_labels = sorted(list_ids_train_labels)
    list_ids_train_inter = sorted(list_ids_train_inter)
    list_ids_train_contour = sorted(list_ids_train_contour)

    dict_train_input = {"ids": list_ids_train_input}
    dict_train_labels = {"ids": list_ids_train_labels}
    dict_train_inter = {"ids": list_ids_train_inter}
    dict_train_contour = {"ids": list_ids_train_contour}

    del list_ids_train_input, list_ids_train_labels, list_ids_train_inter, list_ids_train_contour 
    train_input_df = pd.DataFrame(dict_train_input, index=None)
    train_labels_df = pd.DataFrame(dict_train_labels, index=None)
    train_inter_df = pd.DataFrame(dict_train_inter, index=None)
    train_contour_df = pd.DataFrame(dict_train_contour, index=None)
    train_input_df.to_csv(destination_name_inputs)
    train_labels_df.to_csv(destination_name_labels)
    train_inter_df.to_csv(destination_name_inter)
    train_contour_df.to_csv(destination_name_contour)

def get_ids_in_list(inputs):
    label_list = [f for f in listdir(inputs)]
    label_list = sorted(label_list)
    return label_list


get_ids_in_csv("../data/GenData/TrainData/images/", "../data/GenData/TrainData/labels/", "../data/GenData/TrainData/watershed/", "../data/GenData/TrainData/labels_inter_inv/", 
            "../data/GenData/train_input_ids.csv", "../data/GenData/train_labels_ids.csv", "../data/GenData/train_inter_ids.csv","../data/GenData/train_contour_ids.csv")