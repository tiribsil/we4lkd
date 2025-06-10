import string

target_disease = 'acute myeloid leukemia'
folder_name = target_disease.lower().translate(str.maketrans('', '', string.punctuation)).replace(' ', '_')