from nltk.corpus import stopwords
import dataclean
import pickle

path='data\\lg.pkl'
model = pickle.load(open(path, 'rb'))
path1='data\\count.pkl'
vector=pickle.load(open(path1,'rb'))
dc = dataclean.Dataclean()

def find(x):
    x = dc.lower_case_convertion(x)
    x = dc.remove_punctuation(x)
    x = dc.numtowords(x)
    x = dc.lower_case_convertion(x)
    x = dc.remove_html_tags_beautifulsoup(x)
    x = dc.remove_urls(x)
    x = dc.accented_to_ascii(x)
    x = dc.remove_extra_spaces(x)
    x = dc.remove_single_char(x)
    stop = stopwords.words('english')
    x= ' '.join([word for word in x.split() if word not in (stop)])
    x = dc.emoji_words(x)
    x = dc.lemmatization(x)
    x = [x]
    vect = vector.transform(x).toarray()
    my_prediction = model.predict(vect)
    val = my_prediction[0]
    label={0:'religion',     
    1:'age',                    
    2:'gender',                 
    3:'ethnicity',              
    4:'not_cyberbullying',      
    5:'other_cyberbullying'}
    msg = label[val]
    print(msg)

