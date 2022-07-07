def convert_sample_to_history_dialog(sample, with_knowledge=None):

    history_dialog = "History Dialogue:\n"
    for turn in sample["dialogue"]:
        history_dialog += f"{turn[0]}" +"\n"
        if turn[1] == "":
            ## NO GENERATION REQUIRED
            pass
        else:
            history_dialog += f"{turn[1]}" +"\n\n"
    # print("history:"+'\n'+history_dialog)
    return history_dialog

def convert_sample_to_base_dialog(sample):

    base_dialog = ""
    for turn in sample["Sample"]:
        base_dialog += sample["SPEAKER1"] + f"{turn[0]}" +"\n"
        if turn[1] == "":
            ## NO GENERATION REQUIRED
            pass
        else:
            base_dialog += sample["SPEAKER2"] + f"{turn[1]}" +"\n\n"
    return base_dialog
