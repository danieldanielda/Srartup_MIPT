import re

class Chunker:
    
    async def split_into_sentences(self, text: str):
        # Split text into sentences #
        return re.split(r'(?<=[.!?])\s+', text)