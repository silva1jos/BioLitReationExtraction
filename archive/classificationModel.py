import sklearn;
from sklearn.linear_model import LogisticRegression;
from sklearn.model_selection import train_test_split;
from sklearn.metrics import precision_score;
from sklearn.metrics import recall_score;
from sklearn.metrics import f1_score;
from sklearn.metrics import precision_recall_fscore_support, roc_curve, auc
import os, sys, gensim,nltk,csv;
from collections import namedtuple;
import gensim
import matplotlib.pyplot as plt

def getData(foldername):
	# listOfFiles = os.listdir(foldername);
	data = dict();
	EachInstance = namedtuple("EachInstance", "ID gene disease sentence association");

	# for eachFile in listOfFiles:
	fileObj = open(foldername);#+'/'+eachFile);
	csvReader = csv.reader(fileObj);
	first = 1;
	for row in csvReader:
		if first:
			first = 0;
			continue;
		if row[1] == 'Y' or row[1] == 'N':
			tempObj = EachInstance(ID=row[0], gene=row[2], disease=row[3], sentence=row[4], association='P');
		else:
			tempObj = EachInstance(ID=row[0], gene=row[2], disease=row[3], sentence=row[4], association=row[1]);
		data[row[0]] = tempObj;
	fileObj.close();
	return data;

def getData_directcsv(foldername):
	# listOfFiles = os.listdir(foldername);
	data = dict();
	EachInstance = namedtuple("EachInstance", "ID gene disease sentence association");

	# for eachFile in listOfFiles:
	fileObj = open(foldername);#+'/'+eachFile);
	csvReader = csv.reader(fileObj);
	first = 1;
	for row in csvReader:
		if first:
			first = 0;
			continue;
<<<<<<< HEAD
#		if row[1] == 'Y' or row[1] == 'N':
#			tempObj = EachInstance(ID=row[0], gene=row[2], disease=row[3], sentence=row[4], association='P');
#		else:
		tempObj = EachInstance(ID=row[0], gene=row[2], disease=row[3], sentence=row[4], association=row[1]);
=======
		if row[1] == 'Y' or row[1] == 'N':
			tempObj = EachInstance(ID=row[0], gene=row[2], disease=row[3], sentence=row[4], association='P');
		else:
			tempObj = EachInstance(ID=row[0], gene=row[2], disease=row[3], sentence=row[4], association=row[1]);
>>>>>>> d922a2a2706321d1703664ed6874e938e1b550d3
		data[row[0]] = tempObj;
	fileObj.close();
	return data;

def getTrainTest(ynData, fData):
	trainDataYN, testDataYN = train_test_split(list(ynData.values()),test_size=0.15,random_state=7);
	trainDataF, testDataF = train_test_split(list(fData.values()),test_size=0.15,random_state=7);
	trainData = trainDataYN + trainDataF;
	testData = testDataYN + testDataF;
	print('Train data:',len(trainData));
	print('Test data:',len(testData));
	print('Train data YN:',len(trainDataYN));
	print('Test data YN:',len(testDataYN));
	print('Train data F:',len(trainDataF));
	print('Test data F:',len(testDataF),'\n');
	# sys.exit()
	trainDataOutput = dict();
	testDataOutput = dict();
	for i in trainData:
		trainDataOutput[i.ID] = i;
	for i in testData:
		testDataOutput[i.ID] = i;

	return trainDataOutput, testDataOutput;

def vecSum(arr1,arr2):
    new_arr = [];
    for i in range(len(arr1)):
        new_arr.append(arr1[i] + arr2[i]);
    return new_arr;

def vecDivide(arr1,n):
    new_arr = [];
    for i in range(len(arr1)):
        new_arr.append(arr1[i] / n);
    return new_arr;

def wordVecFeature(model,data):
	feat = dict();
	for eachFile in data:
		eachSentence = data[eachFile].sentence;
		sentEmb = [];
		count = 0;
		for eachWord in nltk.tokenize.word_tokenize(eachSentence):
			try:
				if len(sentEmb) == 0:
					## Made changes for disease gene tag
					# if eachWord == data[eachFile].gene:
					# 	sentEmb = model.wv.word_vec('_GENE_');
					# elif eachWord == data[eachFile].disease:
					# 	sentEmb = model.wv.word_vec('_DISEASE_');
					# else:
					# 	sentEmb = model.wv.word_vec(eachWord.lower());
					sentEmb = model.wv.word_vec(eachWord.lower());
				else:
					# if eachWord == data[eachFile].gene:
					# 	sentEmb = vecSum(model.wv.word_vec('_GENE_'), sentEmb);
					# elif eachWord == data[eachFile].disease:
					# 	sentEmb = vecSum(model.wv.word_vec('_DISEASE_'), sentEmb);
					# else:
					# 	sentEmb = vecSum(model.wv.word_vec(eachWord.lower()), sentEmb);
						# sentEmb = model.wv.word_vec(eachWord.lower());
					sentEmb = vecSum(model.wv.word_vec(eachWord.lower()), sentEmb);
					
				count += 1;
			except:
				pass;
		if count == 0:
			print('No word embedding: ', eachFile);
		else:
			feat[eachFile] = vecDivide(sentEmb,count);
	print('Word2Vec feature generation done!');
	sys.stdout.flush();
	return feat;

def getSyntacticParseFeatures(filename):
	fileObj = open(filename);
	csvReader = csv.reader(fileObj);
	data = dict();
	first = 1;
	for row in csvReader:
		if first:
			first = 0;
			continue;
		data[row[0]] = [float(x) for x in row[1:]];
	fileObj.close();
	return data;

def getCombinedFeatures(featArr1,featArr2):
	data = dict();
	for eachId in featArr1:
		# print(type(featArr2))
		# print(type(featArr1))
		data[eachId] = featArr1[eachId] + featArr2[eachId];
	return data;

def getRuleBasedFeatures(filename):
	fileObj = open(filename);
	csvReader = csv.reader(fileObj);
	data = dict();
	for row in csvReader:
		data[row[0]] = [float(x) for x in row[1:]];
	fileObj.close();
	return data;

def leftSide(syntacticFeatureFilename,ruleBasedFeatureFilename):
	ruleBasedFeatures = getRuleBasedFeatures(ruleBasedFeatureFilename);
	# print(ruleBasedFeatures);
	syntacticParseFeatures = getSyntacticParseFeatures(syntacticFeatureFilename);
	# print(syntacticParseFeatures)
	combinedFeatured = getCombinedFeatures(syntacticParseFeatures,ruleBasedFeatures);
	return combinedFeatured;

def getModelInputFormat(leftSideFeats, word2VecFeats,trainTestData):
	data = [];
	label = [];
	for instanceId in word2VecFeats:
		featVect = word2VecFeats[instanceId] + leftSideFeats[instanceId];
		# if not len(featVect) == 100:
			# print(len(featVect));
			# print(instanceId)
		data.append(featVect);
		label.append(trainTestData[instanceId].association);
		# print(len(data[0]));
		# sys.exit();	
	# print(label);
	# sys.exit()

	return data, label;

def plot_roc(model, test_X, test_y):
    # Compute ROC curve and ROC area
    probs = model.predict_proba(test_X)
    y_pred = probs[:,1]
    fpr, tpr, threshold = roc_curve(test_y, y_pred)
    roc_auc = auc(fpr, tpr)
    
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    
#/home/asani07/Documents/nlp/project/CS6120Project
def main():

	trainFoldername = '../dataset/final_train.csv';
	testFoldername = '../dataset/final_test.csv';

	trainData = getData_directcsv(trainFoldername);
	testData = getData_directcsv(testFoldername);


	# ynDataFile = '../dataset/GAD_Y_N_wPubmedID_annotated_cap_preprocessed.csv';
	# fDataFile = '../dataset/GAD_newF2_wPubmedID_cap_preprocessed.csv';
	# fDataFile = '../dataset/GAD_newF_wPubmedID_preprocessed.csv';

	word2VecFilename = 'skipGram_model.model';
	# word2VecFilename = 'skipGram_model_taggedGeneDisease_2.model';

	# ynData = getData(ynDataFile);
	# fData = getData(fDataFile);	

	syntacticFeatureFilenameTrain = '../dataset/train_original_with_id.csv';
	ruleBasedFeatureFilenameTrain = '../dataset/rule_feature_train.csv';

	syntacticFeatureFilenameTest = '../dataset/test_original_with_id.csv';
	ruleBasedFeatureFilenameTest = '../dataset/rule_feature_test.csv';

	# trainData, testData = getTrainTest(ynData, fData);
	word2VecModel = gensim.models.Word2Vec.load(word2VecFilename);

	trainFeaturesWord2Vec = wordVecFeature(word2VecModel,trainData);
	testFeaturesWord2Vec = wordVecFeature(word2VecModel,testData);

	trainFeaturesLeftSide = leftSide(syntacticFeatureFilenameTrain,ruleBasedFeatureFilenameTrain); # dict: ID -> []
	testFeaturesLeftSide = leftSide(syntacticFeatureFilenameTest,ruleBasedFeatureFilenameTest);

	key = {'F':0, 'N':1, 'Y':1} # two classes
#	key = {'F':0, 'N':-1, 'Y':1} # three classes
    
	X,y = getModelInputFormat(trainFeaturesLeftSide, trainFeaturesWord2Vec,trainData);
	Xtest, ytest = getModelInputFormat(testFeaturesLeftSide, testFeaturesWord2Vec,testData);

	y = [key[i] for i in y];
	ytest = [key[i] for i in ytest];

	logRegModel = LogisticRegression();
	logRegModel.fit(X,y);

	print('Accuracy:', logRegModel.score(Xtest,ytest));
	print('Precision:', precision_score(ytest,logRegModel.predict(Xtest),average='weighted'));
	print('Recall:', recall_score(ytest,logRegModel.predict(Xtest),average='weighted'));
	print('F1:', f1_score(ytest,logRegModel.predict(Xtest),average='weighted'));
    
    # ROC curve
	plot_roc(logRegModel, Xtest, ytest);

	return;


if __name__ == '__main__':
	main();
	print('Program terminated!');