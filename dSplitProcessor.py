def process_dataset():
    # Load the IMDB dataset
    train_iter = IMDB(root=".data", split='train')  # Load the 'train' split

    # Convert to list for random splitting
    data = [(label, text) for label, text in train_iter]

    # Compute sizes for train and initial test splits
    total_size = len(data)
    train_size = total_size * 0.7
    val_size    = total_size * 0.2
    test_size = total_size * 0.1

    # Create train and test datasets with random split
    train_data, test_data,val_data = random_split(data, [train_size,val_size, test_size])
    #train_size =   25000+25000//split_1 or 50000*0.7
    #test_size   =   25000(1-1//split_1) or 50000*0.1
    #val_size   =   25000(8/10-2/5) or 50000*0.2
    # Split the test_data further into train_data, val_data, and test

    # Preprocess all datasets using label and text pipelines
    train_data = [
        (label_pipeline(label), text_pipeline(text))
        for label, text in train_data
    ]
    val_data = [
        (label_pipeline(label), text_pipeline(text))
        for label, text in val_data
    ]
    test_data = [
        (label_pipeline(label), text_pipeline(text))
        for label, text in test_data
    ]

    return train_data, val_data, test_data


# Process the dataset
train_data, val_data, test_data = process_dataset()

# Print sizes of each split
print(f"Training data size: {len(train_data)}")
print(f"Validation data size: {len(val_data)}")
print(f"Testing data size: {len(test_data)}"