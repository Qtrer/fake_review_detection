import os
import string
import pandas as pd
from sklearn.model_selection import train_test_split

DATASET_PATH = 'dataset/op_spam/'

hotel_name = []
reviewer_num = []
is_fake = []
review_type = []
is_positive = []
reviews = []
punctuation_num = []
capture_num = []
review_length = []

for filename in os.listdir(DATASET_PATH):
    file_list = filename.replace('.txt', '').split('_')
    if file_list[0] == 'd':
        is_fake.append('1')
        review_type.append('d')
    else:
        is_fake.append('0')
        review_type.append('t')
    if file_list[3] == 'p':
        is_positive.append('1')
    else:
        is_positive.append('0')
    hotel_name.append(file_list[1])
    reviewer_num.append(file_list[2])
    with open(DATASET_PATH + filename, 'r') as f:
        review = f.read()
        p_num = 0
        c_num = 0
        length = 0
        for c in review:
            length += 1
            if c in string.punctuation:
                p_num += 1
            if c.isupper():
                c_num += 1
        reviews.append(review)
        punctuation_num.append(str(p_num))
        capture_num.append(str(c_num))
        review_length.append(str(length))

data = {'hotel_name': hotel_name,
        'reviewer_num': reviewer_num,
        'is_fake': is_fake,
        'review_type': review_type,
        'is_positive': is_positive,
        'punctuation_num': punctuation_num,
        'capture_num': capture_num,
        'length': review_length,
        'reviews': reviews
        }
df = pd.DataFrame(data)
df.to_csv('dataset/op_spam.csv')
