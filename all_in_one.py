

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
            res = "Prediction: NEGATIVE"
        else:
            res = "Prediction: POSITIVE"
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
        res = "Words being ignored due to not appearing in training set are: \n"
        if len(unseen) == 0:
    #         print("None\n")
            res += "None\n"
        else:
    #         print(unseen)
    #         print('')
            res += '['
            for w in unseen:
                if w == unseen[-1]:
                    res += w
                else:
                    res += w + ', '
            res += ']\n'

    #     print("Words being ignored due to mindf (unfrequent in corpus) are: ")
        res += "Words being ignored due to mindf (unfrequent in corpus) are: \n"
        if len(mindf) == 0:
    #         print("None\n")
            res += "None\n"
        else:
    #         print(mindf)
    #         print('')
            res += '['
            for w in mindf:
                if w == mindf[-1]:
                    res += w
                else:
                    res += w + ', '
            res += ']\n'

    #     print("Words being ignored due to maxdf (too frequent in corpus) are: ")
        res += "Words being ignored due to maxdf (too frequent in corpus) are: \n"
        if len(maxdf) == 0:
    #         print("None\n")
            res += "None\n"
        else:
    #         print(maxdf)
    #         print('')
            res += '['
            for w in maxdf:
                if w == maxdf[-1]:
                    res += w
                else:
                    res += w + ', '
            res += ']\n'

    #     print("Words being ignored due to our algorithm are: ")
        res += "Words being ignored due to our algorithm are: \n"
        if len(oliver_algorithm) == 0:
    #         print("None\n")
            res += "None\n"
        else:
    #         print(oliver_algorithm)
    #         print('')
            res += '['
            for w in oliver_algorithm:
                if w == oliver_algorithm[-1]:
                    res += w
                else:
                    res += w + ', '
            res += ']\n'
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
        import pandas as pd
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
        dic = {'Feature':word_list, 'Coef':coef_list, 'in POSITIVE':num_p_list,
              'in NEGATIVE':num_n_list, 'tfidf val':tfidf_list, 'Original Count':count_list,
              'Contribution':contribution_list}
        df = pd.DataFrame(dic)
        df = df[['Feature','Contribution','Original Count','tfidf val','Coef','in POSITIVE','in NEGATIVE']]
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

        fig = plt.figure(figsize=(15,15))
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

        fig = plt.figure(figsize=(5,5))
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
        res += "Remaining words in the vec: \n["
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
        df = df.style.applymap(self.color_negative_red, subset=['Coef','Contribution']).\
            apply(self.highlight_max, subset=['Coef','Contribution']).\
            apply(self.highlight_min, subset=['Coef','Contribution'])
    #     df = df.style.apply(highlight_max)
        return df,res


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
        output.append(input_string) # original sentence
        s = self.predict(input_string)
        output.append(s) # prediction
        df, s = self.analysis(input_string)
        output.append(s) # analysis
        
        figs = self.generate_graphs(input_string)
        figs[0].savefig("4.png")
        figs[1].savefig("5.png")
        figs[2].savefig("6.png")
        
        import imgkit
        html = df.render()
        options = {"xvfb": ""}
        imgkit.from_string(html, "7.png")#, options=options)
        
        output.append(4)
        output.append(5)
        output.append(6)
        output.append(7)
        
        dic = {4:'4.png', 5:'5.png', 6:'6.png', 7:'7.png'}
        return output, dic
    
if __name__ == "__main__":
    m = Model()
    o,d = m.pipeline("Excellent")
    print(o)
    print(d)