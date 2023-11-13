
import numpy as np
import torch
import math

from sklearn.metrics import (
    roc_auc_score,
    log_loss,
    mean_squared_error,
    accuracy_score,
    f1_score,
)
def harm_score(y_true, y_score, k=10):
    """Computing hit score metric at k.

    Args:
        y_true (np.ndarray): ground-truth labels.
        y_score (np.ndarray): predicted labels.

    Returns:
        np.ndarray: hit score.
    """
    ground_truth = np.where(np.array(y_true) == 1)[0]
    k = min(np.shape(ground_truth), k)
    # print("true", y_true)
    # print("score", y_score)
    # import pdb
    # pdb.set_trace()

    argsort = np.argsort(y_score)[::-1][:k]

    cnt = 0
    for idx in argsort:
        if idx in ground_truth:
            cnt = cnt + 1
        else:
            cnt = cnt - 0.5
    # print(cnt)
            # return 1
    return cnt / k
def mrr_score(y_true, y_score):
    """Computing mrr score metric.

    Args:
        y_true (np.ndarray): ground-truth labels.
        y_score (np.ndarray): predicted labels.
    
    Returns:
        np.ndarray: mrr scores.
    """
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true)


def ndcg_score(y_true, y_score, k=10):
    """Computing ndcg score metric at k.

    Args:
        y_true (np.ndarray): ground-truth labels.
        y_score (np.ndarray): predicted labels.

    Returns:
        np.ndarray: ndcg scores.
    """
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best
def dcg_score(y_true, y_score, k=10):
    """Computing dcg score metric at k.

    Args:
        y_true (np.ndarray): ground-truth labels.
        y_score (np.ndarray): predicted labels.

    Returns:
        np.ndarray: dcg scores.
    """
    k = min(np.shape(y_true)[-1], k)
    order = np.argsort(y_score)[::-1]
    # print(order)
    y_true = np.take(y_true, order[:k])
    gains = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)

def test_all_users(model, batch_size, item_num, test_data, test_data_user, user_pos, top_k):
    
    predictedIndices = []
    GroundTruth = []
    labels_group_u = {}
    preds_group_u = {}
    count = 0

    for user, item, label, vague_or_not, true_or_not, pos_lab, neg_lab in test_data:
        user = user.cuda()
        item = item.cuda()
        label = label.float().cuda()
        pred = model(user, item)
        # if (user not in labels_group_u) :
            
        if count == 0:
            predictions = pred
            labels = label
        else:
            predictions = torch.cat([predictions, pred], 0)
            labels = torch.cat([labels, label], 0)
        count = count + 1
    
    # print(len(labels))
    # import pdb
    # pdb.set_trace()
    AUC = cal_metric(labels.cpu().detach().numpy(), predictions.cpu().detach().numpy(), ['auc'])
    # predictedIndices = []
    # GroundTruth = []
    wauc = 0.0
    mrr = 0.0
    ndcg = 0.0
    harm = 0.0
    cnt = 0
    for u in test_data_user:
        # batch_num = item_num // batch_size
        batch_size = len(test_data_user[u])
        batch_user = torch.Tensor([u]*batch_size).long().cuda()
        # st, ed = 0, batch_size
        # print(batch_num)
        batch_item = torch.Tensor([i[0] for i in test_data_user[u]]).long().cuda()
        label = [i[1] for i in test_data_user[u]]
        pred = model(batch_user, batch_item)
        mrr = mrr + mrr_score(label, label)
        ndcg = ndcg + ndcg_score(label, pred.cpu().detach().numpy(), 10)
        harm = harm + harm_score(label, pred.cpu().detach().numpy(), 10)
        wauc = wauc + roc_auc_score(label, pred.cpu().detach().numpy())
        cnt = cnt + 1
    wauc = wauc / cnt
    mrr = mrr / cnt
    ndcg = ndcg / cnt
    harm = harm / cnt
        # for batch_index in range(batch_num):
        #     pred = model(batch_user, batch_item)
        #     # print(batch_index)
        #     if batch_index == 0:
        #         predictions = pred
        #     else:
        #         predictions = torch.cat([predictions, pred], 0)
        #     st, ed = st+batch_size, ed+batch_size
        # ed = ed - batch_size
        # batch_item = torch.Tensor([i for i in range(ed, item_num)]).long().cuda()
        # batch_user = torch.Tensor([u]*(item_num-ed)).long().cuda()
        # pred = model(batch_user, batch_item)
        # predictions = torch.cat([predictions, pred], 0)
    #     test_data_mask = [0] * item_num
    #     if u in user_pos:
    #         for i in user_pos[u]:
    #             test_data_mask[i] = -9999
    #     predictions = predictions + torch.Tensor(test_data_mask).float().cuda()
    #     _, indices = torch.topk(predictions, top_k[-1])
    #     indices = indices.cpu().numpy().tolist()
    #     predictedIndices.append(indices)
    #     GroundTruth.append(test_data_pos[u])
    # AUC['precision'], AUC['recall'], AUC['NDCG'], AUC['MRR'] = compute_acc(GroundTruth, predictedIndices, top_k)
    # return precision, recall, NDCG, MRR
    AUC['wauc'] = wauc
    AUC['ndcg'] = ndcg
    AUC['mrr'] = mrr
    AUC['harm'] = harm
    return AUC
    # return 
def cal_metric(labels, preds, metrics):
    """Calculate metrics,such as auc, logloss.
    
    FIXME: 
        refactor this with the reco metrics and make it explicit.
    """
    res = {}
    if not metrics:
        return res
    # import pdb 
    # pdb.set_trace()
    # print(labels)
    for metric in metrics:
        if metric == "auc":
            auc = roc_auc_score(np.asarray(labels), np.asarray(preds))
            res["auc"] = round(auc, 4)
        elif metric == "rmse":
            rmse = mean_squared_error(np.asarray(labels), np.asarray(preds))
            res["rmse"] = np.sqrt(round(rmse, 4))
        elif metric == "logloss":
            # avoid logloss nan
            preds = [max(min(p, 1.0 - 10e-12), 10e-12) for p in preds]
            logloss = log_loss(np.asarray(labels), np.asarray(preds))
            res["logloss"] = round(logloss, 4)
        elif metric == "acc":
            pred = np.asarray(preds)
            pred[pred >= 0.5] = 1
            pred[pred < 0.5] = 0
            acc = accuracy_score(np.asarray(labels), pred)
            res["acc"] = round(acc, 4)
        elif metric == "f1":
            pred = np.asarray(preds)
            pred[pred >= 0.5] = 1
            pred[pred < 0.5] = 0
            f1 = f1_score(np.asarray(labels), pred)
            res["f1"] = round(f1, 4)
        elif metric == "mean_mrr":
            mean_mrr = np.mean(
                [
                    mrr_score(each_labels, each_preds)
                    for each_labels, each_preds in zip(labels, preds)
                ]
            )
            res["mean_mrr"] = round(mean_mrr, 4)
        elif metric.startswith("ndcg"):  # format like:  ndcg@2;4;6;8
            ndcg_list = [1, 2]
            ks = metric.split("@")
            if len(ks) > 1:
                ndcg_list = [int(token) for token in ks[1].split(";")]
            for k in ndcg_list:
                ndcg_temp = np.mean(
                    [
                        ndcg_score(each_labels, each_preds, k)
                        for each_labels, each_preds in zip(labels, preds)
                    ]
                )
                res["ndcg@{0}".format(k)] = round(ndcg_temp, 4)
        elif metric.startswith("hit"):  # format like:  hit@2;4;6;8
            hit_list = [1, 2]
            ks = metric.split("@")
            if len(ks) > 1:
                hit_list = [int(token) for token in ks[1].split(";")]
            for k in hit_list:
                hit_temp = np.mean(
                    [
                        hit_score(each_labels, each_preds, k)
                        for each_labels, each_preds in zip(labels, preds)
                    ]
                )
                res["hit@{0}".format(k)] = round(hit_temp, 4)
        elif metric == "group_auc":
            group_auc = np.mean(
                [
                    roc_auc_score(each_labels, each_preds)
                    for each_labels, each_preds in zip(labels, preds)
                ]
            )
            res["group_auc"] = round(group_auc, 4)
        else:
            raise ValueError("not define this metric {0}".format(metric))
    return res
def compute_acc(GroundTruth, predictedIndices, topN):
    precision = [] 
    recall = [] 
    NDCG = [] 
    MRR = []
    import pdb
    for index in range(len(topN)):
        sumForPrecision = 0
        sumForRecall = 0
        sumForNdcg = 0
        sumForMRR = 0
        for i in range(len(predictedIndices)):  # for a user,
            if len(GroundTruth[i]) != 0:
                mrrFlag = True
                userHit = 0
                userMRR = 0
                dcg = 0
                idcg = 0
                idcgCount = len(GroundTruth[i])
                ndcg = 0
                hit = []
                # pdb.set_trace()
                for j in range(topN[index]):
                    if predictedIndices[i][j] in GroundTruth[i]:
                        # if Hit!
                        # pdb.set_trace()

                        dcg += 1.0/math.log(j + 2)
                        if mrrFlag:
                            userMRR = (1.0/(j+1.0))
                            mrrFlag = False
                        userHit += 1
                
                    if idcgCount > 0:
                        idcg += 1.0/math.log(j + 2)
                        idcgCount = idcgCount-1
                            
                if(idcg != 0):
                    ndcg += (dcg/idcg)
                # print(userHit / len(GroundTruth[i]))
                # pdb.set_trace()
                # if (userHit > 0) :
                #     pdb.set_trace()
                sumForPrecision += float(userHit) / topN[index]
                sumForRecall += float(userHit) / len(GroundTruth[i])               
                sumForNdcg += ndcg
                sumForMRR += userMRR
        
        precision.append(sumForPrecision / len(predictedIndices))
        recall.append(float(sumForRecall) / len(predictedIndices))
        NDCG.append(sumForNdcg / len(predictedIndices))
        MRR.append(sumForMRR / len(predictedIndices))
        
    return precision, recall, NDCG, MRR

