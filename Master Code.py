import pandas as pd
import re
from datetime import datetime
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn import metrics,svm,naive_bayes
from sklearn.preprocessing import LabelEncoder
from sklearn import decomposition, ensemble
# from scipy import sparse
# from sklearn.preprocessing import LabelEncoder

## Timer Start
t1 = datetime.now()
print("Program Started\nProgress-")

## Read train
train = pd.read_csv("D://Installs/Pycharm/Programs/Office/Projects/Spend ML/Master Code/Train File.csv",encoding="ISO-8859-1")
train['Commodity'].unique()
test = pd.read_csv("D://Installs/Pycharm/Programs/Office/Projects/Spend ML/Master Code/Test File.csv",encoding="ISO-8859-1")

# Fill the missing data
train['Short Text'] = train['Short Text'].fillna("Absent")
test['Short Text'] = test['Short Text'].fillna("Absent")

# test['Commodity'].isna().value_counts()

##
train['Commodity'].value_counts()
test['Commodity'].value_counts()

## Short Text Clean
lst =[]
for i in range(0,len(train)):
    lst.append(' '.join([w for w in re.sub(' +', ' ', ''.join([i for i in re.sub('[^A-Za-z0-9]+', ' ', train.loc[i,'Short Text'].upper()) if not i.isdigit()]).strip()).split() if len(w)>2]))

## Output
train['Clean Text'] = pd.Series(lst)

## Short Text Clean
lst =[]
for i in range(0,len(test)):
    lst.append(' '.join([w for w in re.sub(' +', ' ', ''.join([i for i in re.sub('[^A-Za-z0-9]+', ' ', test.loc[i,'Short Text'].upper()) if not i.isdigit()]).strip()).split() if len(w)>2]))

## Output
test['Clean Text'] = pd.Series(lst)
print("  Short Text Cleaned")
#train.to_csv("D://Installs/Pycharm/Programs/Office/Projects/Spend ML/Master Code/Train File_Vend_Mtrl_Short Text_out.csv")

##################################################### Vendor Classification ######################################################
train_vendor_vect = CountVectorizer(ngram_range=(1, 1),token_pattern=r'\b\w+\b', min_df=1,stop_words= 'english')
vend_vec = train_vendor_vect.fit_transform(train['Vendor Name'].str.upper().values.astype('U'))
df = pd.concat([train['Commodity'],pd.DataFrame(vend_vec.toarray(),columns=list(train_vendor_vect.get_feature_names()))],axis = 1)
df = df.groupby('Commodity').sum().T
Vendor = df.drop(df[(df['Cleaning equipment, supplies and Janitorial services'] < 3) & (df['Furniture and Furnishings'] < 3) & (df['Industrial Manufacturing and Processing Machinery and Accessories'] < 3)
                 & (df['Manufacturing Components and Supplies'] < 3) & (df['Miscellaneous'] < 3) & (df['Office Equipment and Accessories and Supplies'] < 3)
                 & (df['Tools and General Machinery'] < 3)].index)
Vendor['Sum'] = Vendor.sum(axis = 1)
Vendor_new = Vendor.iloc[:,0:].div(Vendor["Sum"], axis=0)
Vendor_list = Vendor_new[(Vendor_new['Cleaning equipment, supplies and Janitorial services'] > .9) | (Vendor_new['Furniture and Furnishings'] > .9) | (Vendor_new['Industrial Manufacturing and Processing Machinery and Accessories'] > .9)
        | (Vendor_new['Manufacturing Components and Supplies'] > .9) | (Vendor_new['Miscellaneous'] > .9) | (Vendor_new['Office Equipment and Accessories and Supplies'] > .9)
        | (Vendor_new['Tools and General Machinery'] > .9)]
# Vendor_list.to_excel("D://Installs/Pycharm/Programs/Office/Projects/Spend ML/Master Code/Vendor_list.xlsx")
# Vendor = train[['Commodity','Vendor Name']].pivot_table(index='Vendor Name', columns='Commodity',aggfunc=len, fill_value=0)

print("  Vendor List Created")
################################################### Matrl Grp Classification #####################################################
train_material_grp_vect = CountVectorizer(ngram_range=(1, 1),token_pattern=r'\b\w+\b', min_df=1,stop_words= 'english')
material_grp_vec = train_material_grp_vect.fit_transform(train['Material Group'].replace('-', '', regex=True))
df = pd.concat([train['Commodity'],pd.DataFrame(material_grp_vec.toarray(),columns=list(train_material_grp_vect.get_feature_names()))],axis = 1)
df = df.groupby('Commodity').sum().T
Material = df.drop(df[(df['Cleaning equipment, supplies and Janitorial services'] < 3) & (df['Furniture and Furnishings'] < 3) & (df['Industrial Manufacturing and Processing Machinery and Accessories'] < 3)
                 & (df['Manufacturing Components and Supplies'] < 3) & (df['Miscellaneous'] < 3) & (df['Office Equipment and Accessories and Supplies'] < 3)
                 & (df['Tools and General Machinery'] < 3)].index)
Material['Sum'] = Material.sum(axis = 1)
Material_new = Material.iloc[:,0:].div(Material["Sum"], axis=0)
#Material_list = Material_new[(Material_new['Building Materials'] > .9) | (Material_new['Civil Materials'] > .9) | (Material_new['Consulting & Miscellaneous'] > .9)]
Material_list = Material_new[(Material_new['Cleaning equipment, supplies and Janitorial services'] > .9) | (Material_new['Furniture and Furnishings'] > .9) | (Material_new['Industrial Manufacturing and Processing Machinery and Accessories'] > .9)
        | (Material_new['Manufacturing Components and Supplies'] > .9) | (Material_new['Miscellaneous'] > .9) | (Material_new['Office Equipment and Accessories and Supplies'] > .9)
        | (Material_new['Tools and General Machinery'] > .9)]
# Material_list.to_excel("D://Installs/Pycharm/Programs/Office/Projects/Spend ML/Master Code/Material_grp_list.xlsx")
# Material = train[['Commodity','Material Group']].pivot_table(index='Material Group', columns='Commodity',aggfunc=len, fill_value=0)

print("  Material List Created")
##################################################### Word frequency counter #####################################################

# Vocabulary- Using Count Vectorizer
vectorizer = CountVectorizer(ngram_range=(1, 1),token_pattern=r'\b\w+\b', min_df=1,stop_words= 'english')
keyword_vec = vectorizer.fit_transform(train['Clean Text'])
# vectorizer.get_feature_names()
# grp = sparse.csr_matrix(LabelEncoder().fit_transform(train['Commodity'].apply(lambda x: x.strftime('%j')))).T
# grp = sparse.csr_matrix(LabelEncoder().fit_transform(train['Commodity'])).astype('int64').T
# grp = sparse.csr_matrix(LabelEncoder().fit_transform(train['Commodity'])).astype('int64')
# arr = sparse.vstack(arr,grp)
# arr = vectorizer.fit_transform(train['Clean Text'].head(5))

# Merging of dataframes
df = pd.concat([train['Commodity'],pd.DataFrame(keyword_vec.toarray(),columns=list(vectorizer.get_feature_names()))],axis = 1)
df = df.groupby('Commodity').sum().T
df = df.drop(df[(df['Building Materials'] < 6) & (df['Civil Materials'] < 6) & (df['Consulting & Miscellaneous'] < 75)].index)
df = df.drop(df[(df['Cleaning equipment, supplies and Janitorial services'] < 3) & (df['Furniture and Furnishings'] < 3) & (df['Industrial Manufacturing and Processing Machinery and Accessories'] < 3)
                 & (df['Manufacturing Components and Supplies'] < 3) & (df['Miscellaneous'] < 3) & (df['Office Equipment and Accessories and Supplies'] < 3)
                 & (df['Tools and General Machinery'] < 3)].index)
df['Sum'] = df.sum(axis = 1)
df_new = df.iloc[:,0:].div(df["Sum"], axis=0)
# df_new.to_excel("D://Installs/Pycharm/Programs/Office/Projects/Spend ML/Master Code/Keywords.xlsx")

# Unigram Keywords for commodities
#Unigrams_list = df_new[(df_new['Building Materials'] > .75) | (df_new['Civil Materials'] > .75) | (df_new['Consulting & Miscellaneous'] > .75)]
Unigrams_list = df_new[(df_new['Cleaning equipment, supplies and Janitorial services'] > .75) | (df_new['Furniture and Furnishings'] > .75) | (df_new['Industrial Manufacturing and Processing Machinery and Accessories'] > .75)
                | (df_new['Manufacturing Components and Supplies'] > .75) | (df_new['Miscellaneous'] > .75) | (df_new['Office Equipment and Accessories and Supplies'] > .75)
                | (df_new['Tools and General Machinery'] > .75)]
# Unigrams_list.to_excel("D://Installs/Pycharm/Programs/Office/Projects/Spend ML/Master Code/Unigrams_list.xlsx")
print("  Unigram Keyword List Created")

# Bigram Keywords for commodities
l1 = set(df.loc[df.index.difference(Unigrams_list.index)].sort_values('Sum',ascending=False).head(200).index)
l2 = set(df.sort_values('Sum',ascending=False).head(30).index)
l3 = list(l1) + list(l2 - l1)
len(l3)

def co_occurance_matrix(input_text,top_words,window_size):
    co_occur = pd.DataFrame(index=top_words, columns=top_words)

    for row, nrow in zip(top_words, range(len(top_words))):
        for colm,ncolm in zip(top_words,range(len(top_words))):
            count = 0
            if row == colm:
                co_occur.iloc[nrow,ncolm] = count
            else:
                for single_essay in input_text:
                    essay_split = single_essay.split(" ")
                    max_len = len(essay_split)
                    top_word_index = [index for index, split in enumerate(essay_split) if row == split]
                    for index in top_word_index:
                        if index == 0:
                            count = count + essay_split[:window_size + 1].count(colm)
                        elif index == (max_len -1):
                            count = count + essay_split[-(window_size + 1):].count(colm)
                        else:
                            count = count + essay_split[index + 1 : (index + window_size + 1)].count(colm)
                            if index < window_size:
                                count = count + essay_split[: index].count(colm)
                            else:
                                count = count + essay_split[(index - window_size): index].count(colm)
                co_occur.iloc[nrow,ncolm] = count

    return co_occur

corpus = list(train.loc[:,'Clean Text'].str.lower())
words = l3
window_size = 10

result = co_occurance_matrix(corpus,words,window_size)

df_upperhalf = result.where(np.triu(np.ones(result.shape)).astype(np.bool))
df_upperhalf = df_upperhalf.stack().reset_index()
df_upperhalf.columns = ['Key1','Key2','Value']
df_upperhalf['Concat'] = df_upperhalf['Key1'].str.cat(df_upperhalf['Key2'], sep =" ")
# df_upperhalf.to_excel("D://Installs/Pycharm/Programs/Office/Projects/Spend ML/Master Code/Bigrams_list.xlsx")

Bigram_vocab = list(df_upperhalf[df_upperhalf['Value'] > 5]['Concat'])

vectorizer = CountVectorizer(ngram_range=(1, 8),token_pattern=r'\b\w+\b', min_df=1,stop_words= 'english',vocabulary=Bigram_vocab)
keyword_vec = vectorizer.fit_transform(train['Clean Text'])
df = pd.concat([train['Commodity'],pd.DataFrame(keyword_vec.toarray(),columns=list(vectorizer.get_feature_names()))],axis = 1)
df = df.groupby('Commodity').sum().T
df['Sum'] = df.sum(axis = 1)
df = df.sort_values('Sum',ascending = False)
df = df[df['Sum'] > 10]
df_new = df.iloc[:,0:].div(df["Sum"], axis=0)
Bigrams_list = df_new[(df_new['Cleaning equipment, supplies and Janitorial services'] > .75) | (df_new['Furniture and Furnishings'] > .75) | (df_new['Industrial Manufacturing and Processing Machinery and Accessories'] > .75)
                | (df_new['Manufacturing Components and Supplies'] > .75) | (df_new['Miscellaneous'] > .75) | (df_new['Office Equipment and Accessories and Supplies'] > .75)
                | (df_new['Tools and General Machinery'] > .75)]

print("  Bigram Keyword List Created")

# Final Features
features = pd.DataFrame()
features['Column Names'] = pd.Series(pd.concat([pd.Series(Vendor_list.index),pd.Series(Material_list.index),pd.Series(Unigrams_list.index)],ignore_index= True))
# features.to_csv("D://Installs/Pycharm/Programs/Office/Projects/Spend ML/Master Code/features.csv")

# Vectors for features
vendor_vect = pd.Series(Vendor_list.index)
matrl_grp_vect = pd.Series(Material_list.index)
Unigrams_vect = pd.Series(Unigrams_list.index)
Bigrams_vect = pd.Series(Bigrams_list.index)

##################################################################### Train File Creation ##################################################################
# Count vectorizer object
def mytokenizer(text):
    return text.split()

# Vector creation using vendor, material grp and keywords
# Vendor vectorizer
train_vendor_vect = CountVectorizer(vocabulary=vendor_vect,tokenizer=mytokenizer)
A = train_vendor_vect.transform(train['Vendor Name'].str.upper().values.astype('U'))
# df1_vendor = pd.concat([train,pd.DataFrame(list(X.toarray()))],axis=1)

# Material grp vectorizer
train_material_grp_vect = CountVectorizer(vocabulary=matrl_grp_vect,tokenizer= mytokenizer)
B = train_material_grp_vect.transform(train['Material Group'].str.upper().replace('-', '', regex=True))
# df1_Material_grp = pd.concat([train,pd.DataFrame(list(X.toarray()))],axis=1)

# Keyword vectorizer
train_keyword_vect = CountVectorizer(vocabulary=Unigrams_vect,tokenizer= mytokenizer)
C = train_keyword_vect.transform(train['Clean Text'])
# df1_keyword = pd.concat([train,pd.DataFrame(list(X.toarray()))],axis=1)

# Keyword vectorizer
train_keyword_vect = CountVectorizer(vocabulary=Bigrams_vect,tokenizer= mytokenizer)
D = train_keyword_vect.transform(train['Clean Text'])
# df1_keyword = pd.concat([train,pd.DataFrame(list(X.toarray()))],axis=1)


# Final Train file
# df1 = pd.concat([train['Vendor Name'],pd.DataFrame(list(A.toarray()))],axis=1)
# df2 = pd.concat([train['Material Group'],pd.DataFrame(list(B.toarray()))],axis=1)
# df3 = pd.concat([train['Clean Text'],pd.DataFrame(list(C.toarray()))],axis=1)
# df4 = pd.concat([train['Clean Text'],pd.DataFrame(list(D.toarray()))],axis=1)

df_train_out = pd.concat([train,pd.DataFrame(list(A.toarray())),pd.DataFrame(list(B.toarray())),pd.DataFrame(list(C.toarray())),pd.DataFrame(list(D.toarray()))],axis=1)
df_train_out = pd.concat([train,pd.DataFrame(list(A.toarray())),pd.DataFrame(list(C.toarray())),pd.DataFrame(list(D.toarray()))],axis=1)

print("  Train File Created")

# df1.to_excel("D://Installs/Pycharm/Programs/Office/Projects/Spend ML/Master Code/Train_file.xlsx")

# Frequency of words
# freq = zip(vectorizer.get_feature_names(),arr.sum(axis=0).tolist()[0])
# print(sorted(freq, key = lambda x: -x[1]))
#
# # To-array
# # arr.toarray()
# # vectorizer.transform(['astm astm aaa al lite aaa a dinnnrpin aaapp Something completely new.']).toarray()
#
# # # Vocabulary- Using Tf-Idf Vectorizer
# # from sklearn.feature_extraction.text import TfidfVectorizer
# # vectorizer = TfidfVectorizer()
# # arr = vectorizer.fit_transform(train['Clean Text'])
# # # arr = vectorizer.fit_transform(train['Clean Text'].head(8))
# # vectorizer.get_feature_names()
# #
# # # To-array
# arr.toarray()

##################################################################### Test File Creation ##################################################################
# Count vectorizer object
def mytokenizer(text):
    return text.split()

# Vector creation using vendor, material grp and keywords
# Vendor vectorizer
test_vendor_vect = CountVectorizer(vocabulary=vendor_vect,tokenizer=mytokenizer)
A = test_vendor_vect.transform(test['Vendor Name'].str.upper().values.astype('U'))
# df1_vendor = pd.concat([test,pd.DataFrame(list(A.toarray()))],axis=1)

# Material grp vectorizer
test_material_grp_vect = CountVectorizer(vocabulary=matrl_grp_vect,tokenizer= mytokenizer)
B = test_material_grp_vect.transform(test['Material Group'].str.upper().replace('-', '', regex=True))
# df1_Material_grp = pd.concat([test,pd.DataFrame(list(B.toarray()))],axis=1)

# Keyword vectorizer
test_keyword_vect = CountVectorizer(vocabulary=Unigrams_vect,tokenizer= mytokenizer)
C = test_keyword_vect.transform(test['Clean Text'])
# df1_keyword = pd.concat([test,pd.DataFrame(list(C.toarray()))],axis=1)

# Keyword vectorizer
test_keyword_vect = CountVectorizer(vocabulary=Bigrams_vect,tokenizer= mytokenizer)
D = test_keyword_vect.transform(test['Clean Text'])
# df1_keyword = pd.concat([test,pd.DataFrame(list(C.toarray()))],axis=1)

# Final test file
# df1 = pd.concat([test['Vendor Name'],pd.DataFrame(list(A.toarray()))],axis=1)
# df2 = pd.concat([test['Material Group'],pd.DataFrame(list(B.toarray()))],axis=1)
# df3 = pd.concat([test['Clean Text'],pd.DataFrame(list(C.toarray()))],axis=1)
# df_test_out = pd.concat([test,pd.DataFrame(list(A.toarray()))],axis=1)
df_test_out = pd.concat([test,pd.DataFrame(list(A.toarray())),pd.DataFrame(list(B.toarray())),pd.DataFrame(list(C.toarray())),pd.DataFrame(list(D.toarray()))],axis=1)
df_test_out = pd.concat([test,pd.DataFrame(list(A.toarray())),pd.DataFrame(list(C.toarray())),pd.DataFrame(list(D.toarray()))],axis=1)

print("  Test File Created")
# df4.to_excel("D://Installs/Pycharm/Programs/Office/Projects/Spend ML/Master Code/Train_file.xlsx")
# ########################################################### Model Implementation ##########################################

# X = df_train_out.iloc[0:,6:]
# y = LabelEncoder().fit_transform(df_train_out['Commodity'])
# train_X, test_X, train_y, test_y = train_test_split(X, y,test_size=.25)

le = LabelEncoder()
le.fit(list(train['Commodity'].str.upper().unique()))

train_X = df_train_out.iloc[0:,7:]
train_y = le.transform(df_train_out['Commodity'].str.upper())
test_X = df_test_out.iloc[0:,7:]
test_y = le.transform(df_test_out['Commodity'].str.upper())
test_y = df_test_out['Commodity'].str.upper()

# train_X.shape
# test_y.shape

print("Classification Models Running on the data -")

# # fit the training dataset on the SVM classifier
# classifier = svm.SVC()
# classifier.fit(train_X, train_y)
#
# # predict the labels on validation dataset
# predictions_svm = classifier.predict(test_X)
# print("SVM Completed")

# fit the training dataset on the NB classifier
Naive = naive_bayes.MultinomialNB()
Naive.fit(train_X, train_y)

# predict the labels on validation dataset
predictions_naive = Naive.predict(test_X)
print("Naive Bayes Completed")

# fit the training dataset on the Random_Forest classifier
random_forest = ensemble.RandomForestClassifier(n_estimators = 100)
random_forest.fit(train_X, train_y)

# predict the labels on validation dataset
predictions_rand_forest = random_forest.predict(test_X)
probab = random_forest.predict_proba(test_X)
print("Random Forest Completed")

# Output dataframe
Output = pd.DataFrame()
Output['Manual output'] = pd.Series(le.inverse_transform(test_y))
# Output['SVM output'] = pd.Series(list(predictions_svm))
Output['Naive Bayes output'] = pd.Series(list(le.inverse_transform(predictions_naive)))
Output['Random Forest output'] = pd.Series(list(le.inverse_transform(predictions_rand_forest)))

# Use accuracy_score function to get the accuracy
# print("SVM Accuracy Score -> ",metrics.accuracy_score(predictions_svm, test_y)*100)
print("Naive Bayes Accuracy Score -> ",metrics.accuracy_score(predictions_naive, test_y)*100)
print("Random forest Accuracy Score -> ",metrics.accuracy_score(predictions_rand_forest, test_y)*100)

# print(pd.crosstab(Output['Manual output'],Output['Naive Bayes output']))
# print(pd.crosstab(Output['Manual output'],Output['Random Forest output']))
Confusion_Mat = pd.DataFrame()
Confusion_Mat = pd.crosstab(Output['Manual output'],Output['Naive Bayes output'],margins= True)
Confusion_Mat = Confusion_Mat.append(pd.Series(name='Random Forest Matrix'))
Confusion_Mat = pd.DataFrame.append(Confusion_Mat,pd.crosstab(Output['Manual output'],Output['Random Forest output'],margins= True))

# Output file
Output.to_csv("D://Installs/Pycharm/Programs/Office/Projects/Spend ML/Master Code/Traditional Result.csv")
Confusion_Mat.to_excel("D://Installs/Pycharm/Programs/Office/Projects/Spend ML/Master Code/Confusion Matrix.xlsx")
pd.DataFrame(probab).to_excel("D://Installs/Pycharm/Programs/Office/Projects/Spend ML/Master Code/Probability_Apoorva.xlsx")

## Timer Stop
t2 = datetime.now()
print("Total Time for Excecution: {}".format(t2-t1))
print("Program succesfully completed")
