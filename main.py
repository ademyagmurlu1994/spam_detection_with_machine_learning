from Tokeniz import Tokenizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import utils
import pickle
tokenizer = Tokenizer(100)

def train(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    #print('X_train:', X_train, '\n\nX_Test: ', X_test, '\n\ny_train:', y_train, '\n\ny_test: ', y_test)
    gnb = GaussianNB()
    model = gnb.fit(X_train, y_train)
    return model


def test(model, messages, feature_vektor):
    X, _ = tokenizer.manipulate_raw_dataset(messages, feature_vektor)
    out = ["Spam Değil", "Spam"]
    y_pred = model.predict(X)

    for i, message in enumerate(messages, 0):
        print(message[0] + ": " + str(out[int(y_pred[i])]))

#%% Model save
def save_model(model):
	with open('trained_classifier.pkl', 'wb') as fid:
		pickle.dump(model, fid)

def load_model(path="trained_classifier.pkl"):
    with open(path, 'rb') as fid:
        loaded_model = pickle.load(fid)

    return loaded_model

if __name__ == "__main__":
    # O spam olmayan mesaj, 1: spam olan mesaj
    raw_data = [['Ücretsiz film izlemek için tıklayın', 1],
                 ['Evde internet sadece 40tl Bu Fırsatı kaçırmayın', 1],
                 ['Tebrikler Samsung 21s kazandınız', 1],
                 ['İphone 12pro kazandınız Hediyenizi almak için bu linke tıklayın', 1],
                 ['Ankara keçiörende arasalar çok ucuza bu fırsatı kaçırmayın. Hemen aşağıdaki linke tıklayın', 1],
                 ['Abi yarına buluşuruz', 0],
                 ['Dün neredeydin', 0],
                 ['Bugün işe geç geleceğim', 0],
                 ['Okula giderken çantanı al', 0],
                 ['İyi günler kardeşim', 0]]

    dict = tokenizer.texts_to_sequences(raw_data)
    sorted_dict = tokenizer.sort_dict(dict)  # kelimelerin sıklığını sırala
    feature_vektor = tokenizer.get_feature_with_dict(sorted_dict, 3)  # sıklığı en çok olan ilk 3 kelime
    print(feature_vektor)
    feature_vektor = utils.dictToArray(feature_vektor)  # feature dict'i arraya dönüştür,

    X, y = tokenizer.manipulate_raw_dataset(raw_data, feature_vektor)
    model = train(X, y)
    #model = load_model()
    #save_model(model)
    test(model, raw_data, feature_vektor)
