import string

def get_target_disease():
    try:
        with open('target_disease.txt', 'r') as file:
            target_disease = file.read().strip()
    except FileNotFoundError:
        target_disease = None

    return target_disease

def set_target_disease(disease):
    with open('target_disease.txt', 'w') as file:
        file.write(disease)

def get_normalized_target_disease():
    return get_target_disease().lower().translate(str.maketrans('', '', string.punctuation)).replace(' ', '_')