import tensorflow as tf
import argparse
import NLP_Tokenization
import numpy as np
from youtube_transcript_api import YouTubeTranscriptApi

_MODEL_INPUT_SIZE = 1024

#Step 0, set up argument parser

parser = argparse.ArgumentParser(description='Pegasus summarization related arguments')
parser.add_argument("--youtube_id", help="viedo id number of given youtube video", default="3Rzi11Hvyh0")
parser.add_argument("--tokenization_type", help="type of tokenization used in this project", default="sentencepiece")
parser.add_argument("--original_summary", help="original_summary of input article", default="original_summary")
parser.add_argument("--model_path", help="path of your Pegusus model", default="model/")
parser.add_argument("--ckpt_path", help="path of your sentencepiece model", default=
                    "c4.unigram.newline.10pct.96000.model")
args = parser.parse_args()


#Step 1, tokenization
text = ''
transcript_list = YouTubeTranscriptApi.list_transcripts(args.youtube_id)
for transcript in transcript_list:
    if transcript.language_code == 'en':
        transcript_dicts = transcript.fetch()
        for transcript_dict in transcript_dicts:
            text +=transcript_dict['text']
#Step 1.5, format video transcript
#TODO: separate sentences
text = text.replace(". ", ".\n\n")
#TODO: captilize sentences
sentences = text.split(".\n\n")
paragraph = []
for sentence in sentences:
    paragraph.append(sentence.capitalize())
text = ".\n\n".join(paragraph)
print(text)

encoder = NLP_Tokenization.create_text_encoder(args.ckpt_path, args.tokenization_type)

input_ids = encoder.encode(text)

#TODO: add feature size controlling
inputs = np.zeros(_MODEL_INPUT_SIZE)
input_size = len(input_ids)
if input_size > _MODEL_INPUT_SIZE:
    input_size = _MODEL_INPUT_SIZE
inputs[:input_size] = input_ids[:input_size]

#Step 2, load model

imported_model = tf.saved_model.load(args.model_path, tags='serve')
example = tf.train.Example()
example.features.feature["inputs"].int64_list.value.extend(inputs.astype(int))

#Step 3, Get summarization output

output = imported_model.signatures["serving_default"](examples=tf.constant([example.SerializeToString()]))

#Step 4, detokenization
id_array = output["outputs"].numpy()
summarization = encoder.decode(id_array.flatten().tolist())
print("\n YOUR SUMMARIZATION: ", summarization)




