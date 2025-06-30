import string

target_disease = 'acute myeloid leukemia'
normalized_target_disease = target_disease.lower().translate(str.maketrans('', '', string.punctuation)).replace(' ', '_')