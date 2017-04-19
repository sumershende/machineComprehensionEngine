import json
from nltk.tokenize import RegexpTokenizer

### Convert from JSON to glove vector

### Convert from JSON into TXT 

def read_json_data(json_data):
	for i in range(len(json_data)):
		print(json_data[i]["title"])
		for j in range(len(json_data[i]["paragraphs"])):
			#print(json_data[i]["paragraphs"][j]["context"])
			print(tokenizer.tokenize(json_data[i]["paragraphs"][j]["context"]))
			paragraphs.append(tokenizer.tokenize(json_data[i]["paragraphs"][j]["context"]))
			para_count.append(len(json_data[i]["paragraphs"][j]["qas"]))
			
			for k in range(len(json_data[i]["paragraphs"][j]["qas"])):
				questions.append(str(json_data[i]["paragraphs"][j]["qas"][k]["question"]))
				answers.append(str(json_data[i]["paragraphs"][j]["qas"][k]["answers"][0]["text"]))
			#	print("\t" + str(json_data[i]["paragraphs"][j]["qas"][k]["question"]))
			#	print("\t" + str(json_data[i]["paragraphs"][j]["qas"][k]["answers"][0]["text"]))
			#	print()
	#for data in json_data:
	#	print(data)
		
tokenizer = RegexpTokenizer(r'\w+')
json_file = "../train-v1.1.json"
json_data_raw = open(json_file)
json_data = json.load(json_data_raw)
#print(json_data["data"][0]["title"])
#print(json_data["data"][0]["paragraphs"][0]["context"])
paragraphs = []
para_count = []
questions = []
answers = []
qas_
read_json_data(json_data["data"])

