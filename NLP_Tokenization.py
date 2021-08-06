from typing import List
import tensorflow as tf
import sentencepiece as sp

_DEFAULT_RESERVED_TOKENS = 103

def create_text_encoder(sp_model_file:str, encoder_type: str):
    if encoder_type == "sentencepiece":
        return SentencepieceEncoder(sp_model_file)
    else:
        raise ValueError("The encoder type %s is not supported!" % encoder_type)

class SentencepieceEncoder(object):
    """SentencepieceEncoder.
    
    Tokenization module based on sentencepiece library
    """
    def __init__(self, sp_model_file:str,
                 reserved_tokens:int = _DEFAULT_RESERVED_TOKENS):
        self._tokenizer = sp.SentencePieceProcessor()
        self._model_file = tf.io.gfile.GFile(sp_model_file,"rb").read()
        self._tokenizer.LoadFromSerializedProto(self._model_file)
        self._reserved_tokens=reserved_tokens
        
    def encode(self, text:str) ->List[int]:
        ids = self._tokenizer.EncodeAsIds(text)
        ids = [i + self._reserved_tokens if i > 1 else i for  i in ids]
        return ids
    
    def decode(self,ids: List[int])->str:
        ids = [
            i - self._reserved_tokens
            if i>1 + self._reserved_tokens else i for i in ids]
        text = self._tokenizer.DecodeIds(ids)
        return text
    
    @property
    def vocab_size(self) -> int:
        return self._tokenizer.GetPieceSize() + self._reserved_tokens