from transformers import pipeline,logging
from datasets import load_dataset,Dataset, Audio
import asyncio
import sounddevice as sd
from scipy.io.wavfile import write
from ollama import chat, AsyncClient
from ollama import ChatResponse
import requests
from duckduckgo_search import DDGS
from flask import Flask,request,stream_with_context, render_template
import time
from urllib.request import urlopen
from bs4 import BeautifulSoup
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
embeddings=HuggingFaceEmbeddings()


##flask --app whisperlive run

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
@app.route('/message/', methods=['GET', 'POST'])
def message():
	
	if request.method == 'GET':
		def generate():
			print(request.args.get('input'))
			yield asyncio.run(stalling(request.args.get('input')))
			yield hint(request.args.get('input'))
		return stream_with_context(generate())
	else:
		return ""


@app.route('/app/', methods=['GET', 'POST'])
def _app():
	if request.method == 'GET':
		return hint(request.args.get('input'))
	else:
		return ""


@app.route('/')
def index():
	return render_template('index.html')



async def stalling(text):
	message = [{
		'role': 'system',
		'content': """
			You are an expert at not answering questions.
			You provide a lot of bulletpoints to help talk around questions.
			You do not mention specific facts. Only reasonable assumptions or absolute common knowledge.
			Only answer with German text.
			Techniques you could use: paraphrasing of what was said, repeating or complementing it, Clarifying questions, rhetorical/redirecting questions.
			Make it subtle: Nobody should notice that you are trying to avoid the question.
			Try to be as unspecific as possbile without mentioning facts.
			DO NOT ANSWER THE QUESTIONS!"""
	  },
	  {
		'role': 'user',
		'content': f"Bitte umgehe diese Frage: {text}"
	  }]
	all_ = "";
	async for part in await AsyncClient().chat(model='gemma3:1b', messages=message, stream=True):
		print(part['message']['content'], end='')
		all_ +=  part['message']['content']
	return all_.replace(" * ","<br> * ") + "<br><br>"

logging.set_verbosity_error()

pipe = pipeline(
  "automatic-speech-recognition",
  model="openai/whisper-medium",
  chunk_length_s=30,
  device='mps'
)

def look_up(source_text,query, num = 3):
	if isinstance(source_text,list):
		texts = source_text
	else:
		texts = source_text.split("\n")
	db = FAISS.from_texts(texts, embeddings)
	retriever = db.as_retriever(search_kwargs={"k": num})
	res = retriever.invoke(query)
	docs = [i.page_content for i in res]
	text = "\n ... \n".join(docs)
	return text
	

def transcript_audio(file):
	ds = Dataset.from_dict({"audio": [file]}).cast_column("audio", Audio())
	sample = ds[0]["audio"]

	prediction = pipe(sample.copy(), batch_size=8, generate_kwargs={"language": "english"})["text"] #remove generate_kwargs={"language": "english"} for no translation
	
	return prediction



def record():
	print("start")
	fs = 16000
	seconds = 10

	myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
	sd.wait()
	print("end")
	write('audiofile.wav', fs, myrecording)
	transcript_audio("audiofile.wav")
	
def hint(text):
	
	#print(text)
	
	response: ChatResponse = chat(model='gemma3:1b', messages=[
		{
		'role': 'system',
		'content': """
			You are an expert at rewriting and not answering questions.
			You rewrite questions the user provides you with to be a search query.
			Do NOT answer with a URL. Only a plain Text is acceptable.
			DO NOT ANSWER THE QUESTION!"""
	  },
	  {
		'role': 'user',
		'content': f"Make that question a search query, please. \nQuestion: {text}"
	  }
	])

	search_request = response['message']['content']
	
	try:
		print("searching "+search_request)
		search_result = search(search_request,text)
		print("result"+search_result)
		
	except Exception as e:
		print(e)
		print("Error")
		search_result = "none"
	
	print("generating")
	
	if search_result != "none":
		response: ChatResponse = chat(model='gemma3:4b', messages=[
			{
			'role': 'system',
			'content': """Answer the message as good as possible using the context provided (life search results).
				Always speak German.
				Never refer to the context even though you should answer based on it"""
		  },
		  {
			'role': 'user',
			'content': f"'''\n{search_result}\n'''\n\nBased on this result answer the following task: {text}"
		  }
		])
		print("final answer with context")
		print(response['message']['content'])
		return response['message']['content']
	else:
		response: ChatResponse = chat(model='gemma3:4b', messages=[
			{
			'role': 'system',
			'content': """Answer the message as precisely as possible.
				Always speak German.
				If you cant answer it try to divert the conversation to a field you can answer.
				"""
		  },
		  {
			'role': 'user',
			'content': f"Answer the following task: {text}"
		  }
		])
		print("final answer")
		print(response['message']['content'])
		return response['message']['content']



def search(such,query) :
	r = ""

	results = DDGS().text(such, max_results=4)
	max_each = 10000
	for result in results:
		#r += result["body"]
		try:
			url = result["href"]
			print(url)
			html = urlopen(url).read()
			soup = BeautifulSoup(html, features="html.parser")

			for script in soup(["script", "style"]):
				script.extract()  
			result = soup.get_text()
			result = '\n'.join(filter(lambda text: text!="",result.split("\n")))
			result = result[:max_each] if len(result)>max_each else result
			r += result
			
		except:
			r += result["body"]
	max_ = 50000
	r = r[:max_] if len(r)>max_ else r
	
	res = look_up(r,query)
	
	print(res)
	try:
		return res
	except:
		print(f"didnt work with {such}")
		return "No context"
"""
if __name__ == "__main__":
	app.run(host='0.0.0.0', port=5000)
"""

#text = input("prompt: ")

#hint(input("prompt: "))

"""
while True:
	
	input("[press enter]")
	text = record()
	asyncio.run(stalling(text))
	hint(text)
"""
