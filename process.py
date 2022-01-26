# import libraries here
import cv2
import numpy as np
import collections
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense,Activation
from keras.optimizers import SGD
from keras.models import model_from_json
from sklearn import datasets
from sklearn.cluster import KMeans
from fuzzywuzzy import fuzz


def train_or_load_character_recognition_model(train_image_paths):
    """
    Procedura prima putanje do fotografija za obucavanje (dataset se sastoji iz razlicitih fotografija alfabeta)

    Procedura treba da istrenira model i da ga sacuva pod proizvoljnim nazivom. Kada se procedura pozove, ona treba da trenira model ako on nije istraniran, ili da ga samo ucita ako je prethodno
    istreniran

    :param train_image_paths: putanje do fotografija alfabeta
    :return: Objekat modela
    """
    # TODO - Istrenirati model ako vec nije istreniran, ili ga samo ucitati ako je vec istreniran
    all_letters = []
    for path in train_image_paths:
        image_color = load_image(path)
        img = prepare_image(image_color)
        selected_regions, letters, region_distances = select_roi(image_color.copy(), img)
        all_letters = all_letters + letters

    alphabet = ['A', 'B', 'C', 'Č', 'Ć', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S',
                'Š', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'Ž', 'a', 'b', 'c', 'č', 'ć', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
                'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 'š', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'ž']

    inputs = prepare_for_ann(all_letters)
    outputs = convert_output(alphabet)
    ann = create_ann()
    ann = train_ann(ann, inputs, outputs)

    return ann


def extract_text_from_image(trained_model, image_path, vocabulary):
    """
    Procedura prima objekat istreniranog modela za prepoznavanje znakova (karaktera), putanju do fotografije na kojoj
    se nalazi tekst za ekstrakciju i recnik svih poznatih reci koje se mogu naci na fotografiji.
    Procedura treba da ucita fotografiju sa prosledjene putanje, i da sa nje izvuce sav tekst koriscenjem
    openCV (detekcija karaktera) i prethodno istreniranog modela (prepoznavanje karaktera), i da vrati procitani tekst
    kao string.

    Ova procedura se poziva automatski iz main procedure pa nema potrebe dodavati njen poziv u main.py

    :param trained_model: <Model> Istrenirani model za prepoznavanje karaktera
    :param image_path: <String> Putanja do fotografije sa koje treba procitati tekst.
    :param vocabulary: <Dict> Recnik SVIH poznatih reci i ucestalost njihovog pojavljivanja u tekstu
    :return: <String>  Tekst procitan sa ulazne slike
    """
    extracted_text = ""
    # TODO - Izvuci tekst sa ulazne fotografije i vratiti ga kao string
    alphabet = ['A', 'B', 'C', 'Č', 'Ć', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S',
                'Š', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'Ž', 'a', 'b', 'c', 'č', 'ć', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
                'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 'š', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'ž']

    image_color = load_image(image_path)
    img = prepare_image(image_color)
    selected_regions, letters, distances = select_roi(image_color.copy(), img)
    if len(distances) >= 2:
        distances = np.array(distances).reshape(len(distances), 1)
        k_means = KMeans(n_clusters=2, max_iter=2000, tol=0.00001, n_init=10)
        k_means.fit(distances)

        inputs = prepare_for_ann(letters)
        results = trained_model.predict(np.array(inputs, np.float32))
        extracted_text = display_result(results, alphabet, k_means)

        for word in extracted_text.split():
            max = 0
            new_word = ""
            for dic_word in getList(vocabulary):
                value = fuzz.ratio(word, dic_word)
                if max < value:
                    max = value
                    new_word = dic_word

            extracted_text = extracted_text.replace(word, new_word)
    return extracted_text


##########################################################################
# image preparation
def prepare_image(image):
    image = cv2.GaussianBlur(image, (3, 3), 0)
    image = image_bin(image_gray(image))
    image = invert(image)
    image = erode(image)
    return image


def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)


def image_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def image_bin(image_gs):
    _, image_bin = cv2.threshold(image_gs, 215, 255, cv2.THRESH_OTSU)
    return image_bin


def invert(image):
    return 255-image


def dilate(image):
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    return cv2.dilate(image, kernel, iterations=1)


def erode(image):
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    return cv2.erode(image, kernel, iterations=1)
##########################################################################


##########################################################################
# regions
def resize_region(region):
    return cv2.resize(region, (28, 28), interpolation = cv2.INTER_NEAREST)


def select_roi(image_orig, image_bin):
    img, contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions_array = []
    first_round = []
    avg_h = 0

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 10 and h > 10:
            avg_h += h
            first_round.append(contour)

    avg_h = avg_h/len(first_round)

    for contour in first_round:
        x, y, w, h = cv2.boundingRect(contour)
        if h > avg_h - 20:
            x, y, w, h = cv2.boundingRect(contour)
            if y - 40 > 0:
                y = y - 40
                h = h + 40
            elif y - 25 > 0:
                y = y - 25
                h = h + 25
            region = image_bin[y:y+h+1,x:x+w+1];
            regions_array.append([resize_region(region), (x,y,w,h)])
            cv2.rectangle(image_orig,(x,y),(x+w,y+h),(0,255,0),2)

    regions_array = sorted(regions_array, key=lambda item: item[1][0])

    sorted_regions = [region[0] for region in regions_array]
    sorted_rectangles = [region[1] for region in regions_array]
    region_distances = []
    for index in range(0, len(sorted_rectangles)-1):
        current = sorted_rectangles[index]
        next_rect = sorted_rectangles[index+1]
        distance = next_rect[0] - (current[0]+current[2])
        region_distances.append(distance)

    return image_orig, sorted_regions, region_distances


##########################################################################
# priprema za neuronsku
def prepare_for_ann(regions):
    ready_for_ann = []
    for region in regions:
        ready_for_ann.append(matrix_to_vector(scale_to_range(region)))
    return ready_for_ann


def scale_to_range(image):
    return image/255


def matrix_to_vector(image):
    return image.flatten()
##########################################################################


def convert_output(outputs):
    return np.eye(len(outputs))


def winner(output):
    return max(enumerate(output), key=lambda x: x[1])[0]


def create_ann():
    ann = Sequential()
    ann.add(Dense(500, input_dim=784, activation='sigmoid'))
    ann.add(Dense(128, input_dim=500, activation='sigmoid'))
    ann.add(Dense(60, activation='sigmoid'))
    return ann


def train_ann(ann, X_train, y_train):
    X_train = np.array(X_train, np.float32)
    y_train = np.array(y_train, np.float32)

    # definisanje parametra algoritma za obucavanje
    sgd = SGD(lr=0.1, momentum=0.9)
    ann.compile(loss='mean_squared_error', optimizer=sgd)

    # obucavanje neuronske mreze
    ann.fit(X_train, y_train, epochs=1000, batch_size=1, verbose=1, shuffle=False)

    return ann


def serialize_ann(ann):
    model_json = ann.to_json()
    with open("serialization_folder/neuronska.json", "w") as json_file:
        json_file.write(model_json)
    ann.save_weights("serialization_folder/neuronska.h5")


def load_trained_ann():
    try:
        # Ucitaj JSON i kreiraj arhitekturu neuronske mreze na osnovu njega
        json_file = open('serialization_folder/neuronska.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        ann = model_from_json(loaded_model_json)
        # ucitaj tezine u prethodno kreirani model
        ann.load_weights("serialization_folder/neuronska.h5")
        print("Istrenirani model uspesno ucitan.")
        return ann
    except Exception as e:
        # ako ucitavanje nije uspelo, verovatno model prethodno nije serijalizovan pa nema odakle da bude ucitan
        return None


def display_result(outputs, alphabet, k_means):
    '''
    Funkcija određuje koja od grupa predstavlja razmak između reči, a koja između slova, i na osnovu
    toga formira string od elemenata pronađenih sa slike.
    Args:
        outputs: niz izlaza iz neuronske mreže.
        alphabet: niz karaktera koje je potrebno prepoznati
        kmeans: obučen kmeans objekat
    Return:
        Vraća formatiran string
    '''
    # Odrediti indeks grupe koja odgovara rastojanju između reči, pomoću vrednosti iz k_means.cluster_centers_
    w_space_group = max(enumerate(k_means.cluster_centers_), key = lambda x: x[1])[0]
    result = alphabet[winner(outputs[0])]
    for idx, output in enumerate(outputs[1:,:]):
        # Iterativno dodavati prepoznate elemente kao u vežbi 2, alphabet[winner(output)]
        # Dodati space karakter u slučaju da odgovarajuće rastojanje između dva slova odgovara razmaku između reči.
        # U ovu svrhu, koristiti atribut niz k_means.labels_ koji sadrži sortirana rastojanja između susednih slova.
        if (k_means.labels_[idx] == w_space_group):
            result += ' '
        result += alphabet[winner(output)]
    return result


def getList(dict):
    lista = []
    for key in dict.keys():
        lista.append(key)

    return lista