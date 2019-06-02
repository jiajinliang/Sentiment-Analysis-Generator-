

import sentiment as sentimentinterface
import classify
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.switch_backend('agg')
import matplotlib.ticker as ticker
# %matplotlib inline
import copy
import math
import matplotlib as mpl
import six
import pandas as pd
import subprocess

import importlib

class Model:
    
    def __init__(self):
        self.sentiment, self.cls = self.train()
        

    def train(self):
        importlib.reload(sentimentinterface)
        print("Reading data")
        tarfname = "data/sentiment.tar.gz"
        sentiment = sentimentinterface.read_data(tarfname)

        sentiment.stop_words = sentimentinterface.generate_stop_words(sentiment, diff = 0.4)

        from sklearn.feature_extraction.text import CountVectorizer

        sentiment.cv = CountVectorizer(min_df = 3)
        sentiment.cv.fit_transform(sentiment.train_data)
        sentiment.mindf_stop_words = sentiment.cv.stop_words_
        sentiment.cv = CountVectorizer(max_df = 0.2)
        sentiment.cv.fit_transform(sentiment.train_data)
        sentiment.maxdf_stop_words = sentiment.cv.stop_words_
        sentiment.cv = CountVectorizer()
        sentiment.cv.fit_transform(sentiment.train_data)
        sentiment.training_set_vocabulary = sentiment.cv.vocabulary_

        sentimentinterface.vectorize_data(sentiment, stop_words = sentiment.stop_words, max_df = 0.2, min_df = 3)
        cls = classify.train_classifier(sentiment.trainX, sentiment.trainy, C = 3.7)

        classify.evaluate(sentiment.devX, sentiment.devy, cls, 'dev')
        # print("\nReading unlabeled data")
        # unlabeled = sentimentinterface.read_unlabeled(tarfname, sentiment)
        # print("Writing predictions to a file")
        # sentimentinterface.write_pred_kaggle_file(unlabeled, cls, "data/sentiment-pred.csv", sentiment)
        
        
        # Logistic Regression Interception
        self.intercept = copy.deepcopy(cls.intercept_)[0]

        # Vectorizer vocaulary list (ordered)
        cv = sentiment.count_vect.vocabulary_
        cv = [(v,w) for w,v in cv.items()]
        cv.sort()
        cv = [x[1] for x in cv]
        self.cv = cv

        return sentiment, cls

    def vectorize_sentence(self, sentence):
        return self.sentiment.count_vect.transform([sentence])

    def predict(self, sentence):
        sentence_vect = self.vectorize_sentence(sentence)
        result = self.cls.predict(sentence_vect)
        if result[0] == 0:
            res = "Prediction: NEGATIVE\n\n"
        else:
            res = "Prediction: POSITIVE\n\n"
        if sentence in self.sentiment.train_data:
            if self.sentiment.trainy[self.sentiment.train_data.index(sentence)] == 1:
                res += "Label:\tPOSITIVE\n"
            else:
                res += "Label:\tNEGATIVE\n"
        else:
            res += "Label:\tNOT AVAILABLE\n"
        print(res)
        return res

    def clean(self, s):
        from string import punctuation
        s_new = []
        s_ignored = []
        res = []
        for c in s:
            if c not in punctuation:
                s_new.append(c.lower())
            else:
                s_new.append(' ')

        s = ''.join(c for c in s_new)
        #s = [''.join(c for c in s if c not in punctuation)][0]

        l = s.split()

        for w in l:
            if w in self.sentiment.count_vect.vocabulary_:
                if w not in res:
                    res.append(w)
            else:
                s_ignored.append(w)
    #     l = [w for w in l if w in sentiment.count_vect.vocabulary_]

        return res, s_ignored

    def find_stop_words(self, s_ignored):
    #     sentence_vect = clean(sentence)
        unseen = []
        maxdf = []
        mindf = []
        oliver_algorithm = []
        for w in s_ignored:
            if w not in self.sentiment.training_set_vocabulary:
                unseen.append(w)
            if w in self.sentiment.maxdf_stop_words:
                maxdf.append(w)
            if w in self.sentiment.mindf_stop_words:
                mindf.append(w)
            if w in self.sentiment.stop_words:
                oliver_algorithm.append(w)

    #     print("Words being ignored due to not appearing in training set are: ")
        res = "WORDS BEING IGNORED WHEN VECTORIZING THE SENTENCE:\n\n"
        res += "Words being ignored due to not appearing in training set are: \n"
        if len(unseen) == 0:
    #         print("None\n")
            res += "None\n"
        else:
    #         print(unseen)
    #         print('')
            res += '[' + ', '.join(set(unseen)) + ']\n'

    #     print("Words being ignored due to mindf (unfrequent in corpus) are: ")
        res += "\nWords being ignored due to mindf (unfrequent in corpus) are: \n"
        if len(mindf) == 0:
    #         print("None\n")
            res += "None\n"
        else:
    #         print(mindf)
    #         print('')
            res += '[' + ', '.join(set(mindf)) + ']\n'

    #     print("Words being ignored due to maxdf (too frequent in corpus) are: ")
        res += "\nWords being ignored due to maxdf (too frequent in corpus) are: \n"
        if len(maxdf) == 0:
    #         print("None\n")
            res += "None\n"
        else:
    #         print(maxdf)
    #         print('')
            res += '[' + ', '.join(set(maxdf)) + ']\n'

    #     print("Words being ignored due to our algorithm are: ")
        res += "\nWords being ignored due to our algorithm are: \n"
        if len(oliver_algorithm) == 0:
    #         print("None\n")
            res += "None\n"
        else:
    #         print(oliver_algorithm)
    #         print('')
            res += '[' + ', '.join(set(oliver_algorithm)) + ']\n'
        return res

    def explain_coef(self):
        p_dict = {}
        n_dict = {}
        sentences = self.sentiment.count_vect.inverse_transform(self.sentiment.trainX)
        for counter in range(0, len(self.sentiment.train_labels)):
            if self.sentiment.train_labels[counter] == "POSITIVE":
                for w in sentences[counter]:
                    if w in p_dict:
                        p_dict[w] += 1
                    else:
                        p_dict[w] = 1
            else:
                for w in sentences[counter]:
                    if w in n_dict:
                        n_dict[w] += 1
                    else:
                        n_dict[w] = 1
        return p_dict, n_dict

    def find_coef(self, sentence_vect, tfidf_vect):
        stop_words = self.sentiment.stop_words
        cls = self.cls
        sentiment = self.sentiment
        p_dict, n_dict = self.explain_coef()

    #     sentence_vect = clean(sentence)
        word_list = []
        coef_list = []
        num_p_list = []
        num_n_list = []
        tfidf_list = []
        count_list = []
        contribution_list = []
        for word in sentence_vect:
            if word in sentiment.count_vect.vocabulary_:
    #             print(word,"\'s coef:\n", cls.coef_[0][sentiment.count_vect.vocabulary_[word]])
                word_list.append(word)
                coef = cls.coef_[0][sentiment.count_vect.vocabulary_[word]]
                coef_list.append(coef)
                tfidf = tfidf_vect.toarray()[0][sentiment.count_vect.vocabulary_[word]]
                tfidf_list.append(tfidf)
                contribution_list.append(coef*tfidf)
                vec = sentiment.cv.transform([self.sentence])
                count_list.append(vec.toarray()[0][sentiment.training_set_vocabulary[word]])
                if word in p_dict:
                    num_p = p_dict[word]
                else:
                    num_p = 0
                if word in n_dict:
                    num_n = n_dict[word]
                else:
                    num_n = 0
    #             print("Number of ",word,"in POSITIVE reviews: ",num_p,"\tNumber of ",word,"in NEGATIVE reviews: ",num_n,"\n")
                num_p_list.append(num_p)
                num_n_list.append(num_n)
        dic = {'_Feature_':word_list, '_Coefficient_':coef_list, '_POSITIVE_':num_p_list,
              '_NEGATIVE_':num_n_list, '_TFIDF_':tfidf_list, '_Count_':count_list,
              '_Contribution_':contribution_list}
        df = pd.DataFrame(dic)
        df = df[['_Feature_','_Contribution_','_Count_','_TFIDF_','_Coefficient_','_POSITIVE_','_NEGATIVE_']]
    #     print(df)
        return df

    def color_negative_red(self, val):
        """
        Takes a scalar and returns a string with
        the css property `'color: red'` for negative
        strings, black otherwise.
        """
        try:
            if float(val) < 0:
                color = 'red'
            elif float(val) > 0:
                color = 'green'
            else:
                color = 'black'
    #         color = 'red' if float(val) < 0 else 'black'
        except ValueError:
            color = 'black'

        return 'color: %s' % color

    def highlight_max(self, s):
        '''
        highlight the maximum in a Series yellow.
        '''
        is_max = s == s.max()
        return ['background-color: lightgreen' if v else '' for v in is_max]

    def highlight_min(self, s):
        '''
        highlight the maximum in a Series yellow.
        '''
        is_min = s == s.min()
        return ['background-color: yellow' if v else '' for v in is_min]


    def bar_chart(self, cd, title,ylabel):

        words = [w for w,_ in cd.items()]
        coefs = [cd[w] for w in words]
        l = [x for x in zip(coefs, words)]
        l.sort()
        words = [w for c,w in l]
        coefs = [c for c,w in l]

        fig = plt.figure(figsize=(30,15))
        colors = ['red' if c < 0 else 'blue' for c in coefs]
        plt.bar(words, coefs, color=colors)
        plt.xlabel('Word', fontsize=20)
        plt.ylabel(ylabel, fontsize=20)
        plt.xticks(fontsize=20, rotation=30)
        plt.yticks(fontsize=15)
        plt.title(title, fontsize = 30)
        plt.show()

        return fig

    def coef_list(self, words):
        d = {}
        for w in words:
            c = self.cls.coef_[0][self.sentiment.count_vect.vocabulary_[w]]
            d[w] = c

        return d

    def tfidf_x_coef(self, x):

        cd = {}
        for i,v in zip(x.indices, x.data):
            cd[self.cv[i]] = self.cls.coef_[0][i] * v

        return cd

    def prob(self, x):

        z = self.intercept
        for i,v in zip(x.indices, x.data):
            z += self.cls.coef_[0][i] * v

        pos = 1 / (1 + math.exp(-z))
        neg = 1 - pos

        return [neg,pos]

    def pie_chart(self, probs):

        mpl.rcParams['font.size'] = 20

        fig = plt.figure(figsize=(10,5))
        labels = ["Negative", "Positive"]
        colors = ['red', 'blue']
        explode = (0.1, 0)
        plt.pie(probs, labels=labels, colors=colors,shadow=True, explode=explode,
                autopct='%1.1f%%', startangle=-30)
        plt.axis('equal')
        plt.show()

        return fig

    def generate_graphs(self, sentence):

        cls = self.cls
        sentiment = self.sentiment
        
        figs = []
        x,y = self.clean(sentence)

        x_vec = sentiment.count_vect.transform([sentence])
        probs = self.prob(x_vec)
        f3 = self.pie_chart(probs)
        figs.append(f3)

        cd = self.coef_list(x)
        f1 = self.bar_chart(cd, 'Words with Coefficients', 'Coefficient')
        figs.append(f1)


        cd = self.tfidf_x_coef(x_vec)
        f2 = self.bar_chart(cd, 'Words with Contributions', 'Contribution')
        figs.append(f2)



        return figs

    def analysis(self, sentence):
        
        cls = self.cls
        sentiment = self.sentiment
        
        sentence_vect, s_ignored = self.clean(sentence)
        tfidf_vect = sentiment.count_vect.transform([sentence])
        res = self.find_stop_words(s_ignored)
        res += "\nRemaining words in the vec: \n["
        for w in sentence_vect:
            if w == sentence_vect[-1]:
                res += w
            else:
                res += w + ', '
        res += ']\n'
        print(res)
    #     print("Remaining words in the vec:")
    #     print(sentence_vect)
        df = self.find_coef(sentence_vect, tfidf_vect)
        decimals = pd.Series([4,4,4], index=['_Contribution_', '_TFIDF_', '_Coefficient_'])
        df = df.round(decimals)
        df_style = df.style.applymap(self.color_negative_red, subset=['_Coefficient_','_Contribution_']).\
            apply(self.highlight_max, subset=['_Coefficient_','_Contribution_']).\
            apply(self.highlight_min, subset=['_Coefficient_','_Contribution_'])
    #     df = df.style.apply(highlight_max)
    
        # set column width and centered text
        df_style = df_style.set_properties(**{'width': '100px', 'text-align': 'center'})
        
        return df,df_style,res


    # output = []
    # s = "input the index to use a sentence from training set:"
    # output.append(s)
    # print(s)
    # n = input()
    # sentence = sentiment.train_data[int(n)]
    # s = sentence
    # output.append(s)
    # print(s)
    # s = predict(sentence)
    # output.append(s)
    # print(s)
    # df, s = analysis(sentence)
    # output.append(s)
    # print(s)
    # figs = generate_graphs(sentence)
    # figs[0].savefig("3.png")
    # figs[1].savefig("4.png")
    # figs[2].savefig("5.png")

    # import imgkit
    # html = df.render()
    # options = {"xvfb": ""}
    # imgkit.from_string(html, "7.png", options=options)

    # print(len(output),output)

    def pipeline(self, input_string):
        
        self.sentence = input_string

        output = ['0']
        output.append('Input Sentence:\n\n'+input_string) # original sentence
        s = self.predict(input_string)
        output.append(s) # prediction
        df,df_style,s = self.analysis(input_string)

        output.append(3)
        output.append(s) # analysis
        
        figs = self.generate_graphs(input_string)
        figs[0].savefig("3.png")
        figs[1].savefig("6.png")
        figs[2].savefig("7.png")
        
        # dian's method using matplotlib to plot a DataFrame as table and 
        # decimals = pd.Series([3,5,5], index=['_Contribution_', '_TFIDF_', '_Coefficient_'])
        # df = df.round(decimals)
        # f = render_mpl_table(data=df, header_columns=0, col_width=2.5)
        # f.savefig('5.png')
        
        to_html(df_style,'table.html')
        subprocess.call(
            'wkhtmltoimage -f png --width 0 table.html 5.png', shell=True)

        # this library works fine on linux and jupyter, but not supporting windows
#         import imgkit
#         html = df.render()
#         options = {"xvfb": ""}
#         imgkit.from_string(html, "7.png")#, options=options)
        
        output.append(3)
        output.append(6)
        output.append(7)
        output.append(8)
        output.append(9)
        # output.append(10)

        
        dic = {0:'s',3:'3.png', 5:'5.png',6:'6.png', 7:'7.png', 8:'Wordcloud_pos_corp1.png', 9:'Wordcloud_neg_corp1.png'}
        return output, dic
    
def render_mpl_table(data, col_width=3.0, row_height=0.625, font_size=14,
                 header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                 bbox=[0, 0, 1, 1], header_columns=0,
                 ax=None, **kwargs):
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')

    mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, **kwargs)

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in  six.iteritems(mpl_table._cells):
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0]%len(row_colors) ])
    return ax.get_figure()

def to_html(df_style, path):
    f = open(path,'w')
    f.write(df_style.render())
    f.close()
    
if __name__ == "__main__":
    m = Model()
    o,d = m.pipeline(m.sentiment.train_data[123])
    print(o)
    print(d)