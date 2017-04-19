import json
import tf_glove
import matplotlib
import random
from nltk.tokenize import RegexpTokenizer

### Convert from JSON to glove vector

### Convert from JSON into TXT 
random.seed(123)

def read_json_data(json_data):
	for i in range(1):#len(json_data)):
		#print(json_data[i]["title"])
		for j in range(1):#len(json_data[i]["paragraphs"])):
			#print(json_data[i]["paragraphs"][j]["context"])
			#print(tokenizer.tokenize(json_data[i]["paragraphs"][j]["context"]))
			paragraphs.append(tokenizer.tokenize(json_data[i]["paragraphs"][j]["context"]))
			para_count.append(len(json_data[i]["paragraphs"][j]["qas"]))
			
			for k in range(len(json_data[i]["paragraphs"][j]["qas"])):
				questions.append(tokenizer.tokenize(str(json_data[i]["paragraphs"][j]["qas"][k]["question"])))
				answers.append(tokenizer.tokenize(str(json_data[i]["paragraphs"][j]["qas"][k]["answers"][0]["text"])))
				qas_count.append(j)
			#	print("\t" + str(json_data[i]["paragraphs"][j]["qas"][k]["question"]))
			#	print("\t" + str(json_data[i]["paragraphs"][j]["qas"][k]["answers"][0]["text"]))
			#	print()
	#for data in json_data:
	#	print(data)
		
tokenizer = RegexpTokenizer(r'\w+')
json_file = "../train-v1.1.json"
json_data_raw = open(json_file,'r')
json_data = json.load(json_data_raw)
#print(json_data["data"][0]["title"])
#print(json_data["data"][0]["paragraphs"][0]["context"])
paragraphs = []
para_count = []
questions = []
answers = []
qas_count = []
read_json_data(json_data["data"])
glovemodel_paragraphs = tf_glove.GloVeModel(embedding_size=300, context_size=10)
glovemodel_paragraphs.fit_to_corpus(paragraphs)
glovemodel_paragraphs.train(num_epochs=100)

glovemodel_paragraphs.generate_tsne("../tsne.png")
#print(glovemodel_paragraphs.embeddings[glovemodel_paragraphs.id_for_word(paragraphs[0][0])])


if(None):
	glovemodel_para_count = tf_glove.GloVeModel(embedding_size=300, context_size=10)
	glovemodel_para_count.fit_to_corpus(para_count)
	glovemodel_para_count.train(num_epochs=10)

	glovemodel_questions = tf_glove.GloVeModel(embedding_size=300, context_size=10)
	glovemodel_questions.fit_to_corpus(questions)
	glovemodel_questions.train(num_epochs=10)

	glovemodel_answers = tf_glove.GloVeModel(embedding_size=300, context_size=10)
	glovemodel_answers.fit_to_corpus(answers)
	glovemodel_answers.train(num_epochs=10)

	glovemodel_qas_count = tf_glove.GloVeModel(embedding_size=300, context_size=10)
	glovemodel_qas_count.fit_to_corpus(qas_count)
	glovemodel_qas_count.train(num_epochs=10)
#print(questions[0])

