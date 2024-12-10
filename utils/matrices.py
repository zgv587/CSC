def cal_err(raw_sentence, pred_sentence, corr_sentence, limit_length=300):
    matrices = ["over_corr", "total_err", "true_corr"]
    char_level = {key: 0 for key in matrices}
    sent_level = {key: 0 for key in matrices}

    f1 = f2 = 0
    for i, c in enumerate(raw_sentence):
        if i >= limit_length:
            break
            
        pc, cc = pred_sentence[i], corr_sentence[i]

        if cc != c:
            char_level["total_err"] += 1
            char_level["true_corr"] += pc == cc
            f1 = 1
        elif pc != cc:
            char_level["over_corr"] += 1
            f2 = 1

    if f1:
        sent_level["true_corr"] += all(pred_sentence == corr_sentence)
        sent_level["total_err"] += f1
    sent_level["over_corr"] += f2

    return char_level, sent_level