from tqdm import tqdm
import pandas as pd


def augment_sentence(sentence, aug):
    """""" """
    Constructs a new sentence via text augmentation.

    Input:
        - sentence:     A string of text
        - aug:          An augmentation object defined by the nlpaug library

    Output:
        - A string of text that been augmented
    """ """"""
    return aug.augment(sentence)[0]


def augment_data(df, aug, target_count):
    """
    Takes a pandas DataFrame and augments its text data to a target count

    Input:
        - df:           pandas DataFrame with columns ['Text', 'Label']
        - aug:          nlpaug augmentation object
        - target_count: integer, number of samples per class after augmentation

    Output:
        - df:           DataFrame with augmented data appended and shuffled
    """
    augmented_dfs = [df.copy()]  # start with original df

    for category in tqdm(df["Label"].unique()):
        existing_text = df[df["Label"] == category]

        num_to_gen = target_count - len(existing_text)
        if num_to_gen <= 0:
            continue

        data_to_aug = existing_text.sample(n=num_to_gen, replace=True).copy()
        data_to_aug["Text"] = data_to_aug["Text"].apply(lambda x: aug.augment(x)[0])

        # append augmented data to list
        augmented_dfs.append(data_to_aug)

    # Concatenate all augmented data and shuffle
    augmented_df = pd.concat(augmented_dfs, ignore_index=True)
    return augmented_df.sample(frac=1, random_state=0)
